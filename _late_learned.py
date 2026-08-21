"""Task 2 cheap orthogonality test: does a LEARNED late-offset specialist (small 1D-CNN on the
trailing post-break window, standardized by the series' own history) add anything over the 61-col
champion score at late offset vs REAL negatives? If a rank/logit blend of {champ61, CNN} beats
champ61 alone in OOF AUC, escalate to the full gated-blend multi-seed gate; else it is closed.

Deployable: the CNN sees only the trailing window standardized by history stats -> per-series,
causal, no cross-series state.
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from _exp_fast import hist_context, HIST_CAP
from _night_banks import fit_champ_refs, champ_base_row, fit_dep_refs, dep_feats

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(12)
DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250
L = 250                              # sequence length fed to the CNN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
vals = X["value"].to_numpy(np.float64); per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True); ends = np.append(start[1:], len(ids))
ON, HI = {}, {}
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; v = vals[s:e]; HI[int(u)] = v[p == 1]; ON[int(u)] = v[p == 2]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}
rng = np.random.default_rng(0)

OFF_LO, OFF_HI = 60, 400              # late region
N_EACH = 3000

usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
br = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFF_LO]
nb = [s for s in usable if TAU[s] < 0]
rng.shuffle(br); rng.shuffle(nb); br = br[:N_EACH]


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


def seq_std(w, h):
    """Trailing window standardized by history mean/std, left-padded to length L."""
    m, sd = h.mean(), h.std() + 1e-9
    z = (w - m) / sd
    if len(z) >= L:
        z = z[-L:]
    else:
        z = np.concatenate([np.zeros(L - len(z)), z])
    return z.astype(np.float32)


def sample(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    cref = fit_champ_refs(h); dref = fit_dep_refs(h)
    return champ61(w, ctx, cref, dref), seq_std(w, h)


nb_len = np.array([len(ON[s]) for s in nb])
nb_arr = np.array(nb)
Pc, Ps, Nc, Ns = [], [], [], []
for s in br:
    off = int(rng.integers(OFF_LO, OFF_HI + 1))
    t = TAU[s] + off
    if len(ON[s]) <= t:
        t = len(ON[s]) - 1
    # matched no-break negative: a no-break series ALIVE at the SAME step t (same window length)
    elig = nb_arr[nb_len > t]
    if len(elig) == 0:
        continue
    sn = int(rng.choice(elig))
    cp, qp = sample(s, t); cn, qn = sample(sn, t)
    Pc.append(cp); Ps.append(qp); Nc.append(cn); Ns.append(qn)
n = min(len(Pc), len(Nc))
Pc, Ps = np.array(Pc[:n]), np.array(Ps[:n]); Nc, Ns = np.array(Nc), np.array(Ns)
Xc = np.nan_to_num(np.vstack([Pc, Nc])); Xs = np.vstack([Ps, Ns])
y = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
print(f"n={n}/class  champ61 dim={Xc.shape[1]}  seq len={Xs.shape[1]}  offsets [{OFF_LO},{OFF_HI}]")


class CNN(nn.Module):
    def __init__(self, ch=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch * 2, ch * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch * 2, 1))

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def train_cnn(Xtr, ytr, Xva, epochs=15):
    dev = "cpu"
    m = CNN().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr).unsqueeze(1); yt = torch.tensor(ytr)
    idx = np.arange(len(Xt)); bs = 256
    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            out = m(Xt[b]); loss = lossf(out, yt[b])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(Xva).unsqueeze(1))).numpy()
    return p


# champ61 OOF (HGB)
champ_oof = cross_val_predict(
    HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                   random_state=0),
    Xc, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]

# CNN OOF
cnn_oof = np.zeros(len(y))
skf = StratifiedKFold(5, shuffle=True, random_state=0)
for tr, va in skf.split(Xs, y):
    cnn_oof[va] = train_cnn(Xs[tr], y[tr], Xs[va])


def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r / (len(a) - 1)


auc_champ = roc_auc_score(y, champ_oof)
auc_cnn = roc_auc_score(y, cnn_oof)
blend = 0.5 * rank(champ_oof) + 0.5 * rank(cnn_oof)
auc_blend = roc_auc_score(y, blend)
# logistic stack of the two OOF scores
stack = cross_val_predict(LogisticRegression(max_iter=1000),
                          np.c_[rank(champ_oof), rank(cnn_oof)], y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=1),
                          method="predict_proba")[:, 1]
auc_stack = roc_auc_score(y, stack)
print(f"\nAUC champ61        = {auc_champ:.4f}")
print(f"AUC CNN            = {auc_cnn:.4f}")
print(f"AUC 0.5/0.5 rank   = {auc_blend:.4f}   (delta {auc_blend-auc_champ:+.4f})")
print(f"AUC logit stack    = {auc_stack:.4f}   (delta {auc_stack-auc_champ:+.4f})")
print(f"corr(champ,cnn)    = {np.corrcoef(rank(champ_oof), rank(cnn_oof))[0,1]:.3f}")
