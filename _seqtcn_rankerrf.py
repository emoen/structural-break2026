"""DILATED CAUSAL TCN (WaveNet-family) probe (P_D): a LEARNED 3rd member on the E78 ranker+rf base79
champion. Clone of _seqcnn_rankerrf.py (E83); the ONLY change is the neural model -- its pooled 1-D
CNN is replaced by a dilated causal Temporal Convolutional Network (TCN, Bai et al. 2018, the
modernized WaveNet architecture). Everything else (base79 ranker+rf machinery, the causal
history-normalized sequence input, the affine-invariance self-test, the blend arms, the believe-rule,
the cache format) is byte-identical to _seqcnn_rankerrf.py.

WHY THIS IS THE ONE LEARNED-REP VARIANT WORTH TRYING. The project has FIVE converging nulls
(E59/E79/E82/E83/E84): every per-step, single-series, recent-window learned representation has been
redundant with the saturated base79 hand features. E83 (single-scale CNN) and E84 (multi-scale CNN)
both used POOLING (MaxPool / AdaptiveAvgPool) which AVERAGES AWAY WHERE in the window the break sits.
The E82 diagnostic pinpointed that the failures discard temporal ORDER and changepoint LOCATION. A
dilated CAUSAL TCN is the one variant that structurally fixes this:
  * causal convolutions with exponentially growing dilation give a FULL-window receptive field while
    PRESERVING temporal order (no permutation/sort/pool step anywhere), and
  * reading the LAST (most-recent) timestep's activation as the per-step logit -- with NO global
    pooling -- keeps changepoint-location information intact.
So it directly attacks the E82 diagnosis rather than repeating the pooled-CNN mistake of E83/E84.

INPUT (unchanged from E83). Per online step t, causal (history x_h + x_on[:t+1] only):
    1. history location/scale ONCE per series:  mu_h = mean(x_h), sd_h = std(x_h)  (stored in refs).
    2. trailing window  w = x_on[max(0, t-WINDOW+1):t+1]; take its most recent up-to-L_SEQ points
       (WINDOW=250 >= L_SEQ=128, so this is exactly x_on[max(0,t-L_SEQ+1):t+1]).
    3. NORMALIZE BY HISTORY STATS (not the window's own):  z = (w - mu_h) / sqrt(sd_h**2 + 1e-12).
       Normalizing by the HISTORY mean/sd keeps a mean/variance/AR change inside the window VISIBLE
       (a window-self normalization would erase exactly the break we detect), while making z invariant
       to x -> a*x+b (a>0): mu_h -> a*mu_h+b, sd_h -> a*sd_h => z invariant.
    4. LEFT-PAD to L_SEQ, right-aligned, with a companion 0/1 mask marking valid positions; clip to
       [-8,8]. Degenerate windows (len<2, sd_h==0, non-finite) emit an all-zero seq with all-zero mask.
  The net's 2 input channels are [normalized value z, valid-mask], length L_SEQ, right-aligned so the
  CURRENT step t is always at index -1.

THE TCN MODEL (the only real change vs _seqcnn_rankerrf.py). A stack of canonical Bai-2018
TemporalBlocks. Each TemporalBlock =
    [ weight_norm(Conv1d(k, dilation=d, padding=(k-1)*d)) -> Chomp1d((k-1)*d) -> ReLU -> Dropout ] x2,
    + a 1x1 conv residual connection (identity if channel count already matches), final ReLU.
  The Chomp1d removes the (k-1)*d right-hand (FUTURE) padding so the block stays STRICTLY CAUSAL:
  Conv1d pads (k-1)*d zeros on BOTH sides (out len = L + (k-1)*d), Chomp slices off the last (k-1)*d
  (out len = L), leaving ONLY left padding -> output position i depends on inputs [i-(k-1)*d .. i].
  Dilations DOUBLE per block: 1,2,4,8,16,32 with kernel_size k=3, hidden=32 channels/block, dropout
  0.1, weight_norm on every conv (canonical). RECEPTIVE FIELD (2 convs/block):
      RF = 1 + 2*(k-1)*(1+2+4+8+16+32) = 1 + 4*63 = 253  >= L_SEQ=128  (fully covers the window;
      the single-conv-per-level figure 1 + 2*(1+2+4+8+16+32) = 127 matches Bai's dilation diagram).
  HEAD: take the activation at the LAST timestep (index -1, the most recent point = current step t;
  because the input is right-aligned this is always step t), Linear(hidden->1) -> 1 logit.
  *** NO global/adaptive pooling *** -- that is the whole point (vs E83/E84 which pooled away location).

TRAINING CONFIG (identical to E83): BCEWithLogitsLoss(pos_weight=n_neg/n_pos), Adam lr1e-3 wd1e-4,
batch 512, epochs (smoke 3 / full 12), torch.manual_seed, same device/thread setup. w_mdn=0 still
reproduces the champion EXACTLY (the TCN member only enters the blend at w>0).

AFFINE-INVARIANCE (hard constraint). z is history-standardized, hence invariant to x -> a*x+b to
float32 precision (the only perturbation is the 1e-12 epsilon under the sqrt). The SMOKE self-test
builds (seq,msk) for a random window and for 3.7*w+5.2 and asserts max|delta_seq| < 1e-6 and mask
equality (looser than build A's 1e-8 because seq is stored float32). The input builder is byte-copied
from E83, so this is exactly E83's affine guarantee.

CAUSALITY (hard constraint). No timestep sees future values: (i) every conv is chomped to keep only
left padding, so output i depends on inputs up to i; (ii) the head reads index -1 (step t) whose
receptive field is [t-252 .. t] -- entirely at or before t. Strictly causal.

WHY GENUINELY DIFFERENT FROM THE DEAD LEARNED MEMBERS:
  * E83/E84 (pooled CNN): plain conv + MaxPool/AdaptiveAvgPool discarded WHERE in the window the break
    sits. The TCN keeps a full-window receptive field WITHOUT pooling and reads the current step -- the
    exact axis (temporal order + changepoint location) E82 flagged.
  * E45/E47 (trajectory GRU): a STATEFUL sequence model carrying per-series hidden state that
    anti-aligned across pools. The TCN is per-STEP and STATELESS (a fixed causal conv over one window)
    -- no temporal state, no per-series persistence to disagree about.
  * E31 (CNN) / build A quantile MLP: fed a summary that discards temporal order. The TCN feeds the RAW
    ordered window and its dilated causal convs ARE learned matched filters over that order.
  * E80/E81 (re-weighting): re-weighted the SAME base79 rows and starved early rows. The TCN adds NO
    re-weighting -- it fits the frozen champion members unchanged (w_mdn=0 reproduces the champion
    EXACTLY) and introduces a NEW representation the base never reads.

BASE (always in, = E78 ranker+rf base79, byte-identical to the shipped CHAMPION):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
  The ranker+rf halves are the imported _wloss build machinery. The RF is fit with NO sample_weight
  -> provably identical to the champion. Only the 3rd member and the blend arm are new.

BLEND (3-way fixed-CDF percentile, single arm w_mdn in [0,1)):
  wr = W_RANK*(1-w_mdn); wf = W_RF*(1-w_mdn); wm = w_mdn
  blend = wr*pct(srk,ref_rk) + wf*pct(srf,ref_rf) + wm*pct(smdn,ref_mdn)
  w_mdn = 0  ==  the 2-member champion (baseline), byte-identical.
Arms: w_mdn in {0.10, 0.20, 0.30}.

BELIEVE-RULE (low prior; the learned-representation fork mostly died -- E31 CNN, E45/E47 GRU, the
shape features E59/E79/E82, and the pooled seq-CNNs E83/E84 all NULL). Believe an arm only if
mean_d >= +0.0015 AND positive on all screen seeds (npos == total), then confirm on fresh seeds
{31337,2025,555,99}. The believe-rule is the noise gate: a learned member that only helps on some
seeds is E45 all over again.

Uses its OWN cache dir _seqtcn_cache (the sd dict carries seq/mask matrices, NOT compatible with
_seqcnn_cache / _struct79_cache / _armdn_cache). Do NOT delete those. RUN ONE 15-job harness at a time
(this one also holds a torch TCN -- keep it the only heavy job).
Run:
  SMOKE=1 PYTHONWARNINGS=ignore .venv/bin/python -u _seqtcn_rankerrf.py     # tiny self-test + affine check
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _seqtcn_rankerrf.py \
      > _seqtcn_rankerrf.out 2>&1 & disown
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
from torch.nn.utils import weight_norm
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
L_SEQ = 128                             # fixed length of the raw affine-normalized window sequence
SEQTCN_EPOCHS = 3 if SMOKE else 12      # fixed-epoch TCN training (no early stop)
W_MDN_ARMS = [0.20] if SMOKE else [0.10, 0.20, 0.30]

TCN_HIDDEN = 32                         # channels per TemporalBlock (comparable to E83's ~32-wide CNN)
TCN_LEVELS = 6                          # dilations 1,2,4,8,16,32 -> RF=253 >= L_SEQ=128 (see below)
TCN_KERNEL = 3
TCN_DROPOUT = 0.1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_seqtcn_cache")   # OWN cache (sd carries seq/mask matrices)
RAW = os.path.join(HERE, "_seqtcn_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_seqtcn_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_seqtcn_rankerrf.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- the dilated causal TCN (Bai et al. 2018) over channels=[normalized value, mask], length L_SEQ --
class Chomp1d(nn.Module):
    """Remove the last `chomp_size` timesteps = the RIGHT (future) padding a Conv1d adds, so each
    TemporalBlock is STRICTLY CAUSAL (only left padding remains -> output i depends on inputs <= i)."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Canonical Bai-2018 residual block: [weight_norm Conv1d(dilation d, pad (k-1)*d) -> Chomp ->
    ReLU -> Dropout] x2 + 1x1 residual (identity if channels match) + final ReLU."""

    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size,
                                           padding=pad, dilation=dilation))
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size,
                                           padding=pad, dilation=dilation))
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.drop1,
                                 self.conv2, self.chomp2, self.relu2, self.drop2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _SeqTCNNet(nn.Module):
    """Stack of TemporalBlocks with doubling dilation, then a per-step head on the LAST timestep.
    Input (n, 2, L_SEQ) right-aligned -> logit (n,). NO global/adaptive pooling: the head reads the
    current step t (index -1) so changepoint-location information is preserved (the E82 fix)."""

    def __init__(self, in_ch=2, hidden=TCN_HIDDEN, levels=TCN_LEVELS,
                 kernel_size=TCN_KERNEL, dropout=TCN_DROPOUT):
        super().__init__()
        blocks = []
        for i in range(levels):
            d = 2 ** i                                     # dilations 1,2,4,8,16,32
            cin = in_ch if i == 0 else hidden
            blocks.append(TemporalBlock(cin, hidden, kernel_size, d, dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.tcn(x)                                    # (n, hidden, L_SEQ) -- length preserved
        last = h[:, :, -1]                                 # (n, hidden) -- most-recent step = current t
        return self.head(last).squeeze(-1)                 # (n,) -- 1 logit per step, no pooling


class SeqTCN:
    """sklearn-like wrapper so prep_scores stays parallel to E83. .fit(seq,msk,y) /
    .predict_proba(seq,msk) -> (n,2) with column 1 = P(break). Deterministic on CPU."""

    def __init__(self, seed, epochs=SEQTCN_EPOCHS, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, n_jobs=None):
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = int(batch_size)
        self.n_jobs = int(n_jobs or N_JOBS or 12)
        self.net = None

    def _tensor(self, seq, msk):
        seq = np.clip(np.asarray(seq, dtype=np.float32), -8.0, 8.0)  # safety clip (seq already clipped)
        msk = np.asarray(msk, dtype=np.float32)
        x = np.stack([seq, msk], axis=1)                            # (n, 2, L_SEQ)
        return torch.from_numpy(np.ascontiguousarray(x))

    def fit(self, seq, msk, y):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_jobs)
        seq = np.asarray(seq, dtype=np.float32)
        msk = np.asarray(msk, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        self.net = _SeqTCNNet()
        if n == 0:
            return self
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
                xb = self._tensor(seq[idx], msk[idx])
                yb = torch.from_numpy(np.ascontiguousarray(y[idx]))
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, seq, msk):
        seq = np.asarray(seq, dtype=np.float32)
        msk = np.asarray(msk, dtype=np.float32)
        n = len(seq)
        if n == 0 or self.net is None:
            return np.zeros((0, 2), dtype=np.float64)
        self.net.eval()
        out = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, n, 4096):                              # minibatched to bound memory
                xb = self._tensor(seq[i:i + 4096], msk[i:i + 4096])
                p = torch.sigmoid(self.net(xb)).numpy().astype(np.float64)
                out[i:i + len(p)] = p
        return np.column_stack([1.0 - out, out])


# --- extended per-series refs: base79 refs + history location/scale for affine normalization ------
def _refs_ext(x_h):
    r = _refs_base(x_h)
    xh = np.asarray(x_h, dtype=np.float64)
    r["mu_h"] = float(np.mean(xh)) if len(xh) else 0.0
    r["sd_h"] = float(np.std(xh)) if len(xh) else 0.0
    return r


# --- the learned member's per-step representation (raw affine-normalized window sequence) ----------
def _seqrow(w, ctx, refs):
    """Right-aligned, history-standardized, temporally-ordered trailing window of up to L_SEQ points,
    plus a 0/1 valid-position mask. Returns (seq, msk) float32 length L_SEQ; all-zero for degenerate
    cases. `ctx` is unused (kept for signature parity with _row)."""
    seq = np.zeros(L_SEQ, dtype=np.float32)
    msk = np.zeros(L_SEQ, dtype=np.float32)
    mu_h, sd_h = refs["mu_h"], refs["sd_h"]
    w = np.asarray(w, dtype=np.float64)
    if len(w) < 2 or sd_h == 0.0 or not np.isfinite(sd_h) or not np.isfinite(mu_h):
        return seq, msk
    ww = w[-L_SEQ:]                                          # most recent up-to-L_SEQ online points
    z = (ww - mu_h) / math.sqrt(sd_h * sd_h + 1e-12)         # HISTORY-standardized => affine-invariant
    z = np.clip(z, -8.0, 8.0).astype(np.float32)
    if not np.all(np.isfinite(z)):
        return np.zeros(L_SEQ, dtype=np.float32), np.zeros(L_SEQ, dtype=np.float32)
    n = len(z)
    seq[-n:] = z                                            # right-aligned (left-padded with zeros)
    msk[-n:] = 1.0
    return seq, msk


def _seqsan(a):
    """Sanitize a stack of seq/mask rows to a finite (n, L_SEQ) float32 matrix (mirrors _san)."""
    if not len(a):
        return np.zeros((0, L_SEQ), dtype=np.float32)
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# --- extended train/val builders that ALSO emit seq/mask aligned row-for-row with the base79 rows --
def _train_series_q(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return (np.zeros((0, NCOL)), [], [], [],
                np.zeros((0, L_SEQ), np.float32), np.zeros((0, L_SEQ), np.float32))
    refs = _refs_ext(x_h)
    base = active[0]
    rows, labels, tvals, seqs, msks = [], [], [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))              # base79 row (identical to _wloss)
        s, m = _seqrow(w, ctx, refs)                 # aligned raw-window sequence + mask
        seqs.append(s)
        msks.append(m)
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (_san(rows), labels, tvals, [int(sid)] * len(labels),
            _seqsan(seqs), _seqsan(msks))


def _val_series_q(sid, x_h, x_on, oi):
    ctx = hist_context(x_h, HIST_CAP)
    refs = _refs_ext(x_h)
    active = _active(x_on)
    rows, seqs, msks = [], [], []
    for t in active:
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))
        s, m = _seqrow(w, ctx, refs)
        seqs.append(s)
        msks.append(m)
    feat = _san(rows)
    seq = _seqsan(seqs)
    msk = _seqsan(msks)
    midx = pd.MultiIndex.from_product([[sid], oi], names=["id", "time"])
    return midx, active, feat, seq, msk


def build_train_q(X, yi, sid_to_steps):
    """base79 Xtr + row-aligned SEQ/MSK (the ranker rows Rr are built separately, no seq needed)."""
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
                np.zeros(0, np.int64), np.zeros((0, L_SEQ), np.float32),
                np.zeros((0, L_SEQ), np.float32))
    feat = np.vstack([r for r, _l, _t, _s, _q, _m in keep])
    lab = np.concatenate([l for _r, l, _t, _s, _q, _m in keep])
    tv = np.concatenate([t for _r, _l, t, _s, _q, _m in keep])
    sids = np.concatenate([s for _r, _l, _t, s, _q, _m in keep])
    SEQ = np.vstack([q for _r, _l, _t, _s, q, _m in keep])
    MSK = np.vstack([m for _r, _l, _t, _s, _q, m in keep])
    return feat, lab.astype(np.int64), tv.astype(np.int64), sids.astype(np.int64), SEQ, MSK


def build_val_q(X, fold_ids):
    """Fold cache dict = the _wloss schema (index/feat/active) PLUS row-aligned 'seq' and 'msk'."""
    jobs = [(sid, np.asarray(split_series(X.loc[sid])[0], dtype=np.float64),
             split_series(X.loc[sid])[1].to_numpy(dtype=np.float64),
             split_series(X.loc[sid])[1].index) for sid in fold_ids]
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_val_series_q)(sid, x_h, x_on, oi) for (sid, x_h, x_on, oi) in jobs)
    indices, feats, seqs, msks, active_all, offset = [], [], [], [], [], 0
    for midx, active, feat, seq, msk in parts:
        indices.append(midx)
        feats.append(feat if feat.shape[0] else np.zeros((0, NCOL)))
        seqs.append(seq if seq.shape[0] else np.zeros((0, L_SEQ), np.float32))
        msks.append(msk if msk.shape[0] else np.zeros((0, L_SEQ), np.float32))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOL))
    seq = np.vstack(seqs) if seqs else np.zeros((0, L_SEQ), np.float32)
    msk = np.vstack(msks) if msks else np.zeros((0, L_SEQ), np.float32)
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
    log(f"  [cache] seed {seed} built in {time.time()-t0:.0f}s  "
        f"Xtr{Xtr.shape} SEQ{SEQtr.shape} Rr{Rr.shape} -> {path}")
    return sd


# --- fit the 3 members ONCE per seed, then evaluate every w_mdn arm from cached percentiles -------
def prep_scores(sd, metas, seed):
    """Fit ranker+rf (uniform weights == champion) + the learned TCN member, and return per-fold
    (active, groups, prk, prf, pmdn) where prk/prf/pmdn are the fixed-CDF percentiles at full length."""
    Xtr, ytr = sd["Xtr"], sd["ytr"]

    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr, ytr)                                  # NO sample_weight -> identical to shipped champion
    ref_rf = np.sort(_proba(rf, Xtr))

    rows, lab, grp, _swo = sort_into_groups_w(sd["Rr"], sd["Rl"], sd["Rt"], None)
    rk = fit_ranker_w("rank_xendcg", rows, lab, grp, None)
    ref_rk = np.sort(_proba(rk, sd["Rr"]))

    tcn = SeqTCN(seed, epochs=SEQTCN_EPOCHS, n_jobs=N_JOBS)
    tcn.fit(sd["Str_seq"], sd["Str_msk"], ytr)
    ref_mdn = np.sort(tcn.predict_proba(sd["Str_seq"], sd["Str_msk"])[:, 1])

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
            smdn[active] = tcn.predict_proba(c["seq"], c["msk"])[:, 1]
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
    """seq(x) must equal seq(a*x+b) for a>0 to float32 precision (asserts max|delta| < 1e-6 and
    identical masks)."""
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
    md = float(np.max(np.abs(seq1.astype(np.float64) - seq2.astype(np.float64))))
    mask_ok = bool(np.array_equal(msk1, msk2))
    nz = int(np.sum(np.abs(seq1) > 1e-9))
    log(f"[affine-check] seq maxdiff under x->{a}x+{b}: {md:.2e}  mask_eq={mask_ok}  "
        f"(nonzero {nz}/{L_SEQ})")
    assert mask_ok, "AFFINE INVARIANCE FAILED: masks differ under x->a*x+b"
    assert md < 1e-6, f"AFFINE INVARIANCE FAILED: max|delta|={md:.2e} >= 1e-6 (normalization is wrong)"
    return md


def _rf_check():
    """Compute and log the TCN receptive field so the covers-L_SEQ constraint is auditable."""
    dils = [2 ** i for i in range(TCN_LEVELS)]
    rf = 1 + 2 * (TCN_KERNEL - 1) * sum(dils)          # 2 convs/block
    log(f"[tcn-arch] levels={TCN_LEVELS} kernel={TCN_KERNEL} hidden={TCN_HIDDEN} "
        f"dropout={TCN_DROPOUT} weight_norm=yes dilations={dils} RF={rf} "
        f"(>= L_SEQ={L_SEQ}: {rf >= L_SEQ}) head=last-step(no pooling)")
    return rf


def main():
    open(LOG, "w").close()
    log(f"=== DILATED CAUSAL TCN 3rd-member probe on ranker+rf base79  "
        f"{datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
        f"ship_scales={SHIP_SCALES}")
    log(f"NCOL={NCOL} L_SEQ={L_SEQ} tcn_epochs={SEQTCN_EPOCHS} torch={torch.__version__} "
        f"arms={W_MDN_ARMS}")
    _rf_check()
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
        f.write(f"# dilated causal TCN 3rd-member probe on ranker+rf base79  "
                f"{datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
                f"ship={SHIP_SCALES} L_SEQ={L_SEQ} tcn_epochs={SEQTCN_EPOCHS}\n")
        f.write(f"# 3rd member = dilated causal TCN (Bai 2018) on the raw history-standardized trailing\n")
        f.write(f"# window; full-window RF, NO pooling, last-step head keeps changepoint LOCATION\n")
        f.write(f"# (the E82 fix that pooled CNNs E83/E84 lacked); w_mdn=0 == the champion.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds\n")
        f.write(f"# {{31337,2025,555,99}}. Prior LOW: learned-rep E31/E45/E47 + shape E59/E79/E82 +\n")
        f.write(f"# pooled seq-CNN E83/E84 all NULL.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for arm, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md >= 0.0015 and npos == tot) else ""
            f.write(f"{arm:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
