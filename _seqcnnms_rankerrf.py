"""MULTI-SCALE RAW-WINDOW SEQUENCE / 1-D CNN probe: a LEARNED 3rd member on the E78 ranker+rf base79
champion, extending the single-scale E83 probe (_seqcnn_rankerrf) to THREE window scales.

Origin: 2026-08-29. E83 fed ONE fixed-length (L=128) raw affine-normalized trailing window to a small
1-D CNN 3rd member and landed NULL -- a redundant member, near-exact replay of E82 (the AR-MDN quantile
MLP): negative every seed, monotone worse with weight, adding no orthogonal break signal beyond the
saturated base79 (whose ACF/CUSUM/supF/mosum already encode the recent window). The single 128-point
window is the ONLY axis E83 left untested: a break may sit inside a much SHORTER recent stretch (a fast
mean/variance jump the 128-window averages out) or need a LONGER context (a slow drift the 128-window
truncates). This probe tests that directly by giving the CNN THREE scales at once.

MECHANISM (only change vs E83 is the representation + a multi-branch CNN; the base79 ranker+rf halves,
the 3-way fixed-CDF blend, the arm sweep, the believe-rule and the cache/fold harness are identical):

  SEQ_SCALES = (40, 128, 250). Per online step t, causal (history x_h + x_on[:t+1] only):
    1. history location/scale ONCE per series:  mu_h = mean(x_h), sd_h = std(x_h)  (stored in refs).
    2. trailing window  w = x_on[max(0, t-WINDOW+1):t+1]; then for EACH scale Ls take its most recent
       up-to-Ls points  w[-Ls:].  (WINDOW=250 >= max(SEQ_SCALES)=250, so w[-Ls:] is exactly
       x_on[max(0, t-Ls+1):t+1] for every scale -- verified in _affine_check / _seqrow comment.)
    3. NORMALIZE BY HISTORY STATS (not the window's own):  z = (w[-Ls:] - mu_h)/sqrt(sd_h**2 + 1e-12).
       Normalizing by the HISTORY mean/sd keeps a mean/variance/AR change inside the window VISIBLE
       (a window-self normalization would erase exactly the break we detect), while making z invariant
       to x -> a*x+b (a>0): mu_h -> a*mu_h+b, sd_h -> a*sd_h => z invariant. This holds PER SCALE.
    4. LEFT-PAD each scale to length Ls, right-aligned, with a companion 0/1 mask marking valid
       positions; clip to [-8,8]. Degenerate windows (len<2, sd_h==0, non-finite) emit an all-zero
       seq with all-zero mask for every scale.

  So each row's learned-member embedding is THREE (seq, msk) pairs of lengths 40/128/250, cached as
  float32 per scale (a valid normalized value can be exactly 0.0, so the mask is stored EXPLICITLY,
  never reconstructed as seq!=0). A multi-branch torch 1-D CNN (ONE conv branch per scale -> concat
  pooled features -> shared head) is trained on (seqs, msks, y) aligned row-for-row with the base79
  training rows; its per-step P(break) becomes a 3rd CDF-normalized member of the frozen ranker+rf
  blend. The three branches ARE learned matched filters at three temporal resolutions.

AFFINE-INVARIANCE (hard constraint, per scale). z is history-standardized, hence invariant to
x -> a*x+b to float32 precision (only perturbation is the 1e-12 epsilon under the sqrt). The SMOKE
self-test builds (seq,msk) at every scale for a random window and for 3.7*w+5.2 and asserts
max|delta_seq| < 1e-6 AND mask equality for EACH scale (looser than 1e-8 because seq is stored float32).

BASE (always in, = E78 ranker+rf base79, byte-identical to the shipped CHAMPION halves):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
  The ranker+rf halves are the imported _wloss build machinery. The RF is fit with NO sample_weight
  -> provably identical to the champion. Only the 3rd member and the blend arm are new.

BLEND (3-way fixed-CDF percentile, single arm w_mdn in [0,1)):
  wr = W_RANK*(1-w_mdn); wf = W_RF*(1-w_mdn); wm = w_mdn
  blend = wr*pct(srk,ref_rk) + wf*pct(srf,ref_rf) + wm*pct(smdn,ref_mdn)
  w_mdn = 0  ==  the 2-member ranker+rf proxy champion baseline, byte-identical (drops the CNN).
Arms: w_mdn in {0.10, 0.20, 0.30}.

BELIEVE-RULE (the noise gate; prior LOW). The learned-representation fork has repeatedly died:
E83 (single-scale L=128 CNN) NULL, and before it the learned reps E31/E45/E47 and the shape features
E59/E79/E82 all NULL. Believe an arm only if mean_d >= +0.0015 AND positive on ALL screen seeds
(npos == total), then confirm on fresh seeds {31337,2025,555,99}. A member that only helps on some
seeds is E45 all over again; the believe-rule gates that noise.

Uses its OWN cache dir _seqcnnms_cache (the sd dict carries THREE per-scale seq/mask matrices, NOT
compatible with _seqcnn_cache / _armdn_cache / _struct79_cache). Do NOT delete those. The cache is
~2-3x larger than _seqcnn (three scales) -- watch disk. RUN ONE heavy harness at a time (this one
holds a torch multi-branch CNN).
Run:
  SMOKE=1 PYTHONWARNINGS=ignore .venv/bin/python -u _seqcnnms_rankerrf.py     # tiny self-test + affine
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _seqcnnms_rankerrf.py \
      > _seqcnnms_rankerrf.out 2>&1 & disown
"""
import math
import os
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier

import _ensemble_tune as ET
from _ensemble_tune import WINDOW, load_pool_strat, select_picks
from _exp_fast import HIST_CAP, hist_context, split_series
from _overnight import MIN_N, N_JOBS
from _residual_overnight import _active
# base79 build machinery imported so the ranker+rf half is provably identical to the champion.
from _wloss_rankerrf import (W_RANK, W_RF, RF_TREES, RF_KW, pct, _proba, _auc,
                             make_folds_meta, sort_into_groups_w, fit_ranker_w,
                             _san, _row, _fit_span, _offsets,
                             _refs as _refs_base, build_train as _build_train_base,
                             SHIP_SCALES, NCOL)

SMOKE = os.environ.get("SMOKE") == "1"
SEARCH_SEEDS = [42, 7] if SMOKE else [42, 7, 123]

DENSE_K = 20 if SMOKE else 40           # ranker rows per series (matches _wloss_rankerrf)
SEQ_SCALES = (40, 128, 250)             # multi-scale raw affine-normalized window lengths (short/mid/long)
L_MAX = max(SEQ_SCALES)                 # 250 == WINDOW, so w[-Ls:] == x_on[t-Ls+1:t+1] for every scale
SEQCNN_EPOCHS = 3 if SMOKE else 12      # fixed-epoch CNN training (no early stop)
W_MDN_ARMS = [0.20] if SMOKE else [0.10, 0.20, 0.30]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_seqcnnms_cache")   # OWN cache (sd carries 3 per-scale seq/mask matrices)
RAW = os.path.join(HERE, "_seqcnnms_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_seqcnnms_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_seqcnnms_rankerrf.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- the multi-scale 1-D CNN: one conv branch per scale -> concat pooled feats -> shared head -------
class _SeqBranch(nn.Module):
    """One conv branch over a single scale (channels = [normalized value, mask], length Ls) -> 32-dim."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.body(x)                                        # (n, 32)


class _SeqCNNMSNet(nn.Module):
    def __init__(self, scales):
        super().__init__()
        self.scales = list(scales)
        self.branches = nn.ModuleList([_SeqBranch() for _ in self.scales])
        d = 32 * len(self.scales)                                  # 96 for three scales
        self.head = nn.Sequential(
            nn.Linear(d, 48), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(48, 1),
        )

    def forward(self, xs):                                         # xs = list of (n,2,Ls), parallel to scales
        feats = [b(x) for b, x in zip(self.branches, xs)]
        h = torch.cat(feats, dim=1)                               # (n, 96)
        return self.head(h).squeeze(-1)


class SeqCNNMS:
    """sklearn-like wrapper so prep_scores stays parallel to E83. .fit(seqs,msks,y) /
    .predict_proba(seqs,msks) -> (n,2) with column 1 = P(break). seqs/msks are dicts keyed by scale
    (each value an (n,Ls) float32 array). Deterministic on CPU."""

    def __init__(self, seed, epochs=SEQCNN_EPOCHS, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, n_jobs=None):
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = int(batch_size)
        self.n_jobs = int(n_jobs or N_JOBS or 12)
        self.scales = list(SEQ_SCALES)
        self.net = None

    @staticmethod
    def _cast(seqs, msks, scales):
        """clip seq to [-8,8] and cast both to contiguous float32 dicts (safety clip; seq already clipped)."""
        seqc = {Ls: np.clip(np.asarray(seqs[Ls], dtype=np.float32), -8.0, 8.0) for Ls in scales}
        mskc = {Ls: np.asarray(msks[Ls], dtype=np.float32) for Ls in scales}
        return seqc, mskc

    def _batch(self, seqc, mskc, idx):
        """build the per-scale list of (b,2,Ls) tensors for a row-index slice."""
        xs = []
        for Ls in self.scales:
            x = np.stack([seqc[Ls][idx], mskc[Ls][idx]], axis=1)   # (b, 2, Ls)
            xs.append(torch.from_numpy(np.ascontiguousarray(x)))
        return xs

    def fit(self, seqs, msks, y):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_jobs)
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        self.net = _SeqCNNMSNet(self.scales)
        if n == 0:
            return self
        seqc, mskc = self._cast(seqs, msks, self.scales)
        npos = float(y.sum())
        nneg = float(n - npos)
        pw = torch.tensor([nneg / max(npos, 1.0)], dtype=torch.float32)   # guard divide-by-zero
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ep in range(self.epochs):
            perm = torch.randperm(n, generator=gen).numpy()
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xs = self._batch(seqc, mskc, idx)
                yb = torch.from_numpy(np.ascontiguousarray(y[idx]))
                opt.zero_grad()
                loss = loss_fn(self.net(xs), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, seqs, msks):
        n = int(len(np.asarray(seqs[self.scales[0]])))
        if n == 0 or self.net is None:
            return np.zeros((0, 2), dtype=np.float64)
        seqc, mskc = self._cast(seqs, msks, self.scales)
        self.net.eval()
        out = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, n, 4096):                             # minibatched to bound memory
                idx = slice(i, i + 4096)
                xs = self._batch(seqc, mskc, idx)
                p = torch.sigmoid(self.net(xs)).numpy().astype(np.float64)
                out[i:i + len(p)] = p
        return np.column_stack([1.0 - out, out])


# --- extended per-series refs: base79 refs + history location/scale for affine normalization ------
def _refs_ext(x_h):
    r = _refs_base(x_h)
    xh = np.asarray(x_h, dtype=np.float64)
    r["mu_h"] = float(np.mean(xh)) if len(xh) else 0.0
    r["sd_h"] = float(np.std(xh)) if len(xh) else 0.0
    return r


def _empty_seqdict():
    return {Ls: np.zeros((0, Ls), dtype=np.float32) for Ls in SEQ_SCALES}


# --- the learned member's per-step representation: 3 history-standardized window sequences ----------
def _seqrow(w, ctx, refs):
    """Right-aligned, history-standardized, temporally-ordered trailing window at EACH scale Ls in
    SEQ_SCALES (up to Ls points) plus a 0/1 valid-position mask. Returns (seqs, msks) dicts keyed by
    scale, each an float32 array of length Ls; all-zero for degenerate cases. `ctx` is unused (kept
    for signature parity with _row). Since WINDOW>=L_MAX, w[-Ls:] == x_on[max(0,t-Ls+1):t+1]."""
    seqs = {Ls: np.zeros(Ls, dtype=np.float32) for Ls in SEQ_SCALES}
    msks = {Ls: np.zeros(Ls, dtype=np.float32) for Ls in SEQ_SCALES}
    mu_h, sd_h = refs["mu_h"], refs["sd_h"]
    w = np.asarray(w, dtype=np.float64)
    if len(w) < 2 or sd_h == 0.0 or not np.isfinite(sd_h) or not np.isfinite(mu_h):
        return seqs, msks
    denom = math.sqrt(sd_h * sd_h + 1e-12)                    # HISTORY sd => affine-invariant per scale
    for Ls in SEQ_SCALES:
        ww = w[-Ls:]                                          # most recent up-to-Ls online points
        z = (ww - mu_h) / denom
        z = np.clip(z, -8.0, 8.0).astype(np.float32)
        if not np.all(np.isfinite(z)):
            continue                                          # leave all-zero seq+msk for this scale
        n = len(z)
        seqs[Ls][-n:] = z                                     # right-aligned (left-padded with zeros)
        msks[Ls][-n:] = 1.0
    return seqs, msks


def _seqsan(a, Ls):
    """Sanitize a stack of seq/mask rows to a finite (n, Ls) float32 matrix (mirrors _san)."""
    if not len(a):
        return np.zeros((0, Ls), dtype=np.float32)
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# --- extended train/val builders that ALSO emit per-scale seq/mask aligned row-for-row with base79 --
def _train_series_q(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return (np.zeros((0, NCOL)), [], [], [], _empty_seqdict(), _empty_seqdict())
    refs = _refs_ext(x_h)
    base = active[0]
    rows, labels, tvals = [], [], []
    seqs = {Ls: [] for Ls in SEQ_SCALES}
    msks = {Ls: [] for Ls in SEQ_SCALES}
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))              # base79 row (identical to _wloss)
        s, m = _seqrow(w, ctx, refs)                 # aligned per-scale raw-window sequences + masks
        for Ls in SEQ_SCALES:
            seqs[Ls].append(s[Ls])
            msks[Ls].append(m[Ls])
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    SEQ = {Ls: _seqsan(seqs[Ls], Ls) for Ls in SEQ_SCALES}
    MSK = {Ls: _seqsan(msks[Ls], Ls) for Ls in SEQ_SCALES}
    return (_san(rows), labels, tvals, [int(sid)] * len(labels), SEQ, MSK)


def _val_series_q(sid, x_h, x_on, oi):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _refs_ext(x_h)
    active = _active(x_on)
    rows = []
    seqs = {Ls: [] for Ls in SEQ_SCALES}
    msks = {Ls: [] for Ls in SEQ_SCALES}
    for t in active:
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))
        s, m = _seqrow(w, ctx, refs)
        for Ls in SEQ_SCALES:
            seqs[Ls].append(s[Ls])
            msks[Ls].append(m[Ls])
    feat = _san(rows)
    seq = {Ls: _seqsan(seqs[Ls], Ls) for Ls in SEQ_SCALES}
    msk = {Ls: _seqsan(msks[Ls], Ls) for Ls in SEQ_SCALES}
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat, seq, msk


def build_train_q(X, yi, sid_to_steps):
    """base79 Xtr + row-aligned per-scale SEQ/MSK dicts (ranker rows Rr built separately, no seq)."""
    jobs = []
    for sid, steps in sid_to_steps.items():
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, np.asarray(x_h, dtype=np.float64), onl.to_numpy(dtype=np.float64),
                     int(yi.loc[sid, "tau_index"]), np.asarray(steps)))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_train_series_q)(sid, x_h, x_on, tau, steps)
            for (sid, x_h, x_on, tau, steps) in jobs)
    keep = [p for p in parts if len(p[0])]
    if not keep:
        return (np.zeros((0, NCOL)), np.zeros(0, np.int64), np.zeros(0, np.int64),
                np.zeros(0, np.int64), _empty_seqdict(), _empty_seqdict())
    feat = np.vstack([r for r, _l, _t, _s, _q, _m in keep])
    lab = np.concatenate([l for _r, l, _t, _s, _q, _m in keep])
    tv = np.concatenate([t for _r, _l, t, _s, _q, _m in keep])
    sids = np.concatenate([s for _r, _l, _t, s, _q, _m in keep])
    SEQ = {Ls: np.vstack([q[Ls] for _r, _l, _t, _s, q, _m in keep]) for Ls in SEQ_SCALES}
    MSK = {Ls: np.vstack([m[Ls] for _r, _l, _t, _s, _q, m in keep]) for Ls in SEQ_SCALES}
    return feat, lab.astype(np.int64), tv.astype(np.int64), sids.astype(np.int64), SEQ, MSK


def build_val_q(X, fold_ids):
    """Fold cache dict = the _wloss schema (index/feat/active) PLUS per-scale 'seq' and 'msk' dicts."""
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series_q)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, active_all, offset = [], [], [], 0
    seqs = {Ls: [] for Ls in SEQ_SCALES}
    msks = {Ls: [] for Ls in SEQ_SCALES}
    for midx, active, feat, seq, msk in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOL)))
        for Ls in SEQ_SCALES:
            seqs[Ls].append(seq[Ls] if seq[Ls].shape[0] else np.zeros((0, Ls), np.float32))
            msks[Ls].append(msk[Ls] if msk[Ls].shape[0] else np.zeros((0, Ls), np.float32))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOL))
    seq = {Ls: (np.vstack(seqs[Ls]) if seqs[Ls] else np.zeros((0, Ls), np.float32)) for Ls in SEQ_SCALES}
    msk = {Ls: (np.vstack(msks[Ls]) if msks[Ls] else np.zeros((0, Ls), np.float32)) for Ls in SEQ_SCALES}
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active, "seq": seq, "msk": msk}


def prepare_seed(seed):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"seed_{seed}{'_smoke' if SMOKE else ''}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    ET.SEED = seed
    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    rng = np.random.default_rng(seed + 202)
    rank_steps = {}
    for sid in train_ids:
        L = len(split_series(X.loc[sid])[1].to_numpy())
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        rank_steps[sid] = rng.choice(cand, size=min(DENSE_K, len(cand)), replace=False)
    Xtr, ytr, Ttr, Str, SEQtr, MSKtr = build_train_q(X, yi, picks)
    Rr, Rl, Rt, Rs = _build_train_base(X, yi, rank_steps)   # ranker rows (base79, no seq)
    tau_map = {int(s): int(yi.loc[s, "tau_index"]) for s in train_ids}
    off_tr = _offsets(Ttr, Str, tau_map)
    off_r = _offsets(Rt, Rs, tau_map)
    caches = [build_val_q(X, f) for f in folds]
    sd = dict(Xtr=Xtr, ytr=ytr, Ttr=Ttr, off_tr=off_tr, Str_seq=SEQtr, Str_msk=MSKtr,
              Rr=Rr, Rl=Rl, Rt=Rt, off_r=off_r, caches=caches, y_val=y_val)
    with open(path, "wb") as f:
        pickle.dump(sd, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_seq = SEQtr[SEQ_SCALES[0]].shape[0]
    log(f"  [cache] seed {seed} built in {time.time()-t0:.0f}s  "
        f"Xtr{Xtr.shape} SEQrows={n_seq} scales={SEQ_SCALES} Rr{Rr.shape} -> {path}")
    return sd


# --- fit the 3 members ONCE per seed, then evaluate every w_mdn arm from cached percentiles -------
def prep_scores(sd, metas, seed):
    """Fit ranker+rf (uniform weights == champion) + the learned multi-scale CNN member, and return
    per-fold (active, groups, prk, prf, pmdn) fixed-CDF percentiles at full length."""
    Xtr, ytr = sd["Xtr"], sd["ytr"]

    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr, ytr)                                  # NO sample_weight -> identical to shipped champion
    ref_rf = np.sort(_proba(rf, Xtr))

    rows, lab, grp, _swo = sort_into_groups_w(sd["Rr"], sd["Rl"], sd["Rt"], None)
    rk = fit_ranker_w("rank_xendcg", rows, lab, grp, None)
    ref_rk = np.sort(_proba(rk, sd["Rr"]))

    cnn = SeqCNNMS(seed, epochs=SEQCNN_EPOCHS, n_jobs=N_JOBS)
    cnn.fit(sd["Str_seq"], sd["Str_msk"], ytr)
    ref_mdn = np.sort(cnn.predict_proba(sd["Str_seq"], sd["Str_msk"])[:, 1])

    folds = []
    for (L, active, groups), c in zip(metas, sd["caches"]):
        prk = np.zeros(L)
        prf = np.zeros(L)
        pmdn = np.zeros(L)
        if len(c["feat"]):
            srf = np.zeros(L)
            srk = np.zeros(L)
            smdn = np.zeros(L)
            srf[active] = _proba(rf, c["feat"])
            srk[active] = _proba(rk, c["feat"])
            smdn[active] = cnn.predict_proba(c["seq"], c["msk"])[:, 1]
            prf = pct(srf, ref_rf)
            prk = pct(srk, ref_rk)
            pmdn = pct(smdn, ref_mdn)
        folds.append((active, groups, prk, prf, pmdn))
    return folds


def score_wmdn(folds, w_mdn):
    """3-way fixed-CDF blend at arm w_mdn; w_mdn=0 reproduces the 2-member champion exactly."""
    wr = W_RANK * (1.0 - w_mdn)
    wf = W_RF * (1.0 - w_mdn)
    wm = w_mdn
    aucs = []
    for _active, groups, prk, prf, pmdn in folds:
        blend = wr * prk + wf * prf + wm * pmdn
        wsum = tot = 0.0
        for r, lb, w in groups:
            a = _auc(lb, blend[r])
            if a is None:
                continue
            wsum += w * a
            tot += w
        aucs.append(wsum / tot if tot > 0 else 0.5)
    return float(np.mean(aucs))


def _affine_check():
    """seq(x) must equal seq(a*x+b) for a>0 to float32 precision at EVERY scale (asserts max|delta| <
    1e-6 and identical masks per scale)."""
    rng = np.random.default_rng(0)
    xh = rng.standard_normal(600) * 3.0 + 5.0
    for i in range(2, len(xh)):                       # mild AR structure so the window is non-trivial
        xh[i] += 0.3 * xh[i - 1] - 0.1 * xh[i - 2]
    w = xh[-260:].copy()
    ctx = hist_context(xh, HIST_CAP)
    seq1, msk1 = _seqrow(w, ctx, _refs_ext(xh))
    a, b = 3.7, 5.2
    xh2, w2 = a * xh + b, a * w + b
    ctx2 = hist_context(xh2, HIST_CAP)
    seq2, msk2 = _seqrow(w2, ctx2, _refs_ext(xh2))
    worst = 0.0
    for Ls in SEQ_SCALES:
        md = float(np.max(np.abs(seq1[Ls].astype(np.float64) - seq2[Ls].astype(np.float64))))
        mask_ok = bool(np.array_equal(msk1[Ls], msk2[Ls]))
        nz = int(np.sum(np.abs(seq1[Ls]) > 1e-9))
        log(f"[affine-check] scale {Ls:>3}: seq maxdiff under x->{a}x+{b}: {md:.2e}  mask_eq={mask_ok}  "
            f"(nonzero {nz}/{Ls})")
        assert mask_ok, f"AFFINE INVARIANCE FAILED (scale {Ls}): masks differ under x->a*x+b"
        assert md < 1e-6, (f"AFFINE INVARIANCE FAILED (scale {Ls}): max|delta|={md:.2e} >= 1e-6 "
                           f"(normalization is wrong)")
        worst = max(worst, md)
    return worst


def main():
    open(LOG, "w").close()
    log(f"=== MULTI-SCALE RAW-WINDOW SEQUENCE / 1-D CNN 3rd-member probe on ranker+rf base79  "
        f"{datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
        f"ship_scales={SHIP_SCALES}")
    log(f"NCOL={NCOL} seq_scales={SEQ_SCALES} cnn_epochs={SEQCNN_EPOCHS} torch={torch.__version__} "
        f"arms={W_MDN_ARMS}")
    _affine_check()

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tarm\tbase_auc\tcand_auc\tdelta\n")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        folds = prep_scores(sd, metas, seed)
        base_auc = score_wmdn(folds, 0.0)
        log(f"[seed {seed}] base79 (w_mdn=0) ranker+rf = {base_auc:.4f}  ({time.time()-t0:.0f}s)")
        res = {}
        for w_mdn in W_MDN_ARMS:
            tc = time.time()
            auc = score_wmdn(folds, w_mdn)
            d = auc - base_auc
            res[w_mdn] = d
            arm = f"wmdn{w_mdn:.2f}"
            with open(RAW, "a") as f:
                f.write(f"{seed}\t{arm}\t{base_auc:.6f}\t{auc:.6f}\t{d:+.6f}\n")
            log(f"[seed {seed}] {arm} {auc:.4f}  d={d:+.4f}  ({time.time()-tc:.0f}s)")
        per_seed[seed] = res

    agg = []
    for w_mdn in W_MDN_ARMS:
        ds = [per_seed[s][w_mdn] for s in SEARCH_SEEDS if w_mdn in per_seed[s]]
        if not ds:
            continue
        md = float(np.mean(ds))
        npos = sum(1 for d in ds if d > 0)
        agg.append((f"wmdn{w_mdn:.2f}", md, min(ds), max(ds), npos, len(ds)))
    agg.sort(key=lambda r: -r[1])
    with open(SUMMARY, "w") as f:
        f.write(f"# multi-scale raw-window sequence / 1-D CNN 3rd-member probe on ranker+rf base79  "
                f"{datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
                f"ship={SHIP_SCALES} seq_scales={SEQ_SCALES} cnn_epochs={SEQCNN_EPOCHS}\n")
        f.write(f"# 3rd member = torch multi-branch 1-D CNN over 3 history-standardized window scales\n")
        f.write(f"# (40/128/250; extends the single-scale E83 NULL); w_mdn=0 == the champion baseline.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds\n")
        f.write(f"# {{31337,2025,555,99}}. Prior LOW: E83 single-scale NULL + learned-rep E31/E45/E47 +\n")
        f.write(f"# shape E59/E79/E82 all NULL.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for arm, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md >= 0.0015 and npos == tot) else ""
            f.write(f"{arm:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
