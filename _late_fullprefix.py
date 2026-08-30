"""Learned FULL-online-prefix + history rep, late-offset-specific.

Motivation (from the ledger): the champion is a trailing-250 contrast that FORGETS
the pre-break baseline as the online window fills -> per-offset AUC decays late,
yet late offset holds ~71% of the metric weight. A learned rep that sees the WHOLE
online prefix (baseline + break + post) standardized by history should retain that
baseline at late offset and may add orthogonal signal.

This mirrors _late_learned.py (same step-matched no-break negatives so there is no
window-length/padding confound, same OOF-AUC additivity screen vs champ61) but the
sequence input is the ADAPTIVE-POOLED FULL PREFIX (optionally + a trailing channel)
and the head also consumes a small history-summary vector. Necessary-not-sufficient
OOF-AUC screen; escalate to the multi-seed streaming gate only if it clears here.

ENV: REP=full|fulltrail (default full), SEED=0, N_EACH=3000, EPOCHS=15, LP=256,
     OFF_LO=60, OFF_HI=400, THREADS=12, AUX=1.
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

SEED    = int(os.environ.get("SEED", 0))
REP     = os.environ.get("REP", "full")          # full | fulltrail
N_EACH  = int(os.environ.get("N_EACH", 3000))
EPOCHS  = int(os.environ.get("EPOCHS", 15))
LP      = int(os.environ.get("LP", 256))         # pooled sequence length
OFF_LO  = int(os.environ.get("OFF_LO", 60))
OFF_HI  = int(os.environ.get("OFF_HI", 400))
AUX     = int(os.environ.get("AUX", 1))
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get("THREADS", 12)))

DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
WINDOW = 250
IN_CH = 2 if REP == "fulltrail" else 1

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
rng = np.random.default_rng(SEED)

usable = [s for s in TAU if len(HI[s]) > 40 and len(ON[s]) > 60]
br = [s for s in usable if TAU[s] >= 15 and len(ON[s]) > TAU[s] + OFF_LO]
nb = [s for s in usable if TAU[s] < 0]
rng.shuffle(br); rng.shuffle(nb); br = br[:N_EACH]


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    dep = np.asarray(dep_feats(w, ctx, dref), dtype=np.float64)
    return np.concatenate([base, dep[DEP_KEEP]])


def pool_to(z, Lp):
    """Mean-pool a variable-length standardized series into Lp bins; left-pad if short."""
    n = len(z)
    if n == 0:
        return np.zeros(Lp, np.float32)
    if n <= Lp:
        return np.concatenate([np.zeros(Lp - n), z]).astype(np.float32)
    edges = (np.arange(Lp) * n) // Lp
    sums = np.add.reduceat(z, edges)
    counts = np.diff(np.append(edges, n))
    return (sums / counts).astype(np.float32)


def seq_rep(x_prefix, h):
    """FULL online prefix standardized by history, pooled to LP (+trailing channel)."""
    m, sd = h.mean(), h.std() + 1e-9
    z = np.clip((x_prefix - m) / sd, -8, 8)
    ch0 = pool_to(z, LP)                              # whole-trajectory (keeps baseline)
    if REP != "fulltrail":
        return ch0[None, :]
    w = x_prefix[max(0, len(x_prefix) - WINDOW):]
    zw = np.clip((w - m) / sd, -8, 8)
    ch1 = pool_to(zw, LP)                             # recent detail
    return np.stack([ch0, ch1])


def hist_aux(ctx):
    sd = ctx["sd"] + 1e-9
    return np.array([ctx["skew"], ctx["kurt"], ctx["ac1"], ctx["ac2"],
                     ctx["iqr"] / sd, ctx["rng"] / sd], dtype=np.float32)


def sample(s, t):
    x = ON[s][:t + 1]; w = x[max(0, len(x) - WINDOW):]
    h = HI[s]; ctx = hist_context(h, HIST_CAP)
    cref = fit_champ_refs(h); dref = fit_dep_refs(h)
    return champ61(w, ctx, cref, dref), seq_rep(x, h), hist_aux(ctx)


nb_len = np.array([len(ON[s]) for s in nb]); nb_arr = np.array(nb)
Pc, Ps, Pa, Nc, Ns, Na = [], [], [], [], [], []
for s in br:
    off = int(rng.integers(OFF_LO, OFF_HI + 1))
    t = TAU[s] + off
    if len(ON[s]) <= t:
        t = len(ON[s]) - 1
    elig = nb_arr[nb_len > t]                         # step-matched no-break negative
    if len(elig) == 0:
        continue
    sn = int(rng.choice(elig))
    cp, qp, ap = sample(s, t); cn, qn, an = sample(sn, t)
    Pc.append(cp); Ps.append(qp); Pa.append(ap); Nc.append(cn); Ns.append(qn); Na.append(an)
n = min(len(Pc), len(Nc))
Pc, Ps, Pa = np.array(Pc[:n]), np.array(Ps[:n]), np.array(Pa[:n])
Nc, Ns, Na = np.array(Nc), np.array(Ns), np.array(Na)
Xc = np.nan_to_num(np.vstack([Pc, Nc])); Xs = np.vstack([Ps, Ns])
Xa = np.nan_to_num(np.vstack([Pa, Na]))
y = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
print(f"REP={REP} n={n}/class champ61={Xc.shape[1]} seq=({IN_CH},{LP}) aux={Xa.shape[1]} "
      f"off[{OFF_LO},{OFF_HI}] seed={SEED}")


class SeqNet(nn.Module):
    def __init__(self, in_ch, aux_dim, ch=24, use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, ch, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(ch * 2, ch * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        hid = ch * 2 + (16 if use_aux else 0)
        if use_aux:
            self.auxnet = nn.Sequential(nn.Linear(aux_dim, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hid, 1))

    def forward(self, x, a):
        z = self.net(x)
        if self.use_aux:
            z = torch.cat([z, self.auxnet(a)], dim=1)
        return self.head(z).squeeze(-1)


def train_net(Xtr, Atr, ytr, Xva, Ava, epochs=EPOCHS):
    m = SeqNet(IN_CH, Xa.shape[1], use_aux=bool(AUX))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr); At = torch.tensor(Atr); yt = torch.tensor(ytr)
    idx = np.arange(len(Xt)); bs = 256
    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(Xt[b], At[b]), yt[b])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(Xva), torch.tensor(Ava))).numpy()
    return p


champ_oof = cross_val_predict(
    HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=31,
                                   random_state=SEED),
    Xc, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]

net_oof = np.zeros(len(y))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for tr, va in skf.split(Xs, y):
    net_oof[va] = train_net(Xs[tr], Xa[tr], y[tr], Xs[va], Xa[va])


def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r / (len(a) - 1)


auc_champ = roc_auc_score(y, champ_oof)
auc_net = roc_auc_score(y, net_oof)
blend = 0.5 * rank(champ_oof) + 0.5 * rank(net_oof)
auc_blend = roc_auc_score(y, blend)
stack = cross_val_predict(LogisticRegression(max_iter=1000),
                          np.c_[rank(champ_oof), rank(net_oof)], y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED + 1),
                          method="predict_proba")[:, 1]
auc_stack = roc_auc_score(y, stack)
print(f"\nAUC champ61        = {auc_champ:.4f}")
print(f"AUC net (full-pfx) = {auc_net:.4f}")
print(f"AUC 0.5/0.5 rank   = {auc_blend:.4f}   (delta {auc_blend-auc_champ:+.4f})")
print(f"AUC logit stack    = {auc_stack:.4f}   (delta {auc_stack-auc_champ:+.4f})")
print(f"corr(champ,net)    = {np.corrcoef(rank(champ_oof), rank(net_oof))[0,1]:.3f}")
