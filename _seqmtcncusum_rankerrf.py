"""MODERNTCN + CUSUM-PATH INPUT LENS probe: a LEARNED 3rd member on the E78 ranker+rf base79 champion.
NEAR-BYTE-IDENTICAL CLONE of _seqmtcn_rankerrf.py (E88 ModernTCN); the ONLY change is the network
INPUT LENS -- instead of feeding the raw history-standardized value window z, the value channel now
feeds its causal CUSUM PATH  s = cumsum(z)/sqrt(n)  (the classic cumulative-sum change-point statistic).
The ModernTCN model, the base79 ranker+rf machinery, the mask channel, the affine-invariance self-test,
the blend arms, the believe-rule, the cache format are ALL byte-identical to _seqmtcn_rankerrf.py.

WHY THE CUSUM LENS. All four WaveNet-family probes (E87 TCN, E88 ModernTCN, E89 Hyena, E90 S4) fed the
SAME raw z window and were ALL null -- base79 saturates the raw-window info. The one axis never moved is
the INPUT LENS. CUSUM is the canonical change-point statistic (running sum of standardized increments);
feeding the CUSUM *path* (its trajectory shape) to ModernTCN tests whether accumulated-evidence shape
carries break signal beyond base79's scalar CUSUM features. Prior LOW (~expect null ~0.63); this is a
CLOSURE experiment -- the real test of the "change the input" hypothesis.

WHY THIS ARCHITECTURE. E87 (the dilated causal TCN) was the most defensible learned-rep variant -- it
preserved temporal ORDER and changepoint LOCATION (the axis the E82 diagnostic flagged) and still went
NULL (6th converging null: E59/E79/E82/E83/E84/E87). ModernTCN is the natural next long-conv family: it
keeps a full-window receptive field and the per-step (last-token) readout, but swaps the WaveNet-style
narrow dilated convs for the modern large-kernel depthwise-separable ConvNet block (DWConv token mixing
+ pointwise ConvFFN channel mixing) operating on PATCH tokens rather than raw timesteps. Prior is VERY
LOW (the bottleneck looks like base79 information saturation, not architecture) -- this is a SMOKE-first
completeness probe of the WaveNet-family backlog, not a high-prior bet.

INPUT (unchanged from E83/E87). Per online step t, causal (history x_h + x_on[:t+1] only):
    1. history location/scale ONCE per series:  mu_h = mean(x_h), sd_h = std(x_h)  (stored in refs).
    2. trailing window  w = x_on[max(0, t-WINDOW+1):t+1]; take its most recent up-to-L_SEQ points.
    3. NORMALIZE BY HISTORY STATS:  z = (w - mu_h) / sqrt(sd_h**2 + 1e-12)  (affine-invariant to
       x -> a*x+b, a>0, and keeps a mean/variance/AR change inside the window VISIBLE).
    4. CUSUM PATH (the ONLY lens change vs E88):  c = cumsum(z) (strictly causal running sum),
       s = c / sqrt(arange(1, n+1))  (sqrt(n)-standardized classic CUSUM statistic path). s is LINEAR
       in z with no dependence on a or b -> STILL affine-invariant; cumsum is strictly causal.
    5. LEFT-PAD to L_SEQ, right-aligned, with a companion 0/1 mask marking valid positions; clip
       [-8,8]. Degenerate windows emit an all-zero seq + all-zero mask.
  The net's 2 input channels are [CUSUM-path s, valid-mask], length L_SEQ, right-aligned so the
  CURRENT step t is always at index -1.

THE MODERNTCN MODEL (the only real change vs _seqtcn_rankerrf.py):
  PATCH STEM (causal patchify): Conv1d(in_ch=2, D, kernel_size=P, stride=P) with NO padding maps the
    L=128 sequence to N=L/P=16 non-overlapping PATCH TOKENS of width D=32. Because kernel==stride and
    there is no padding, patch token j reads ONLY inputs [P*j, P*j+P-1] -- its own chunk, never a later
    one -- so the stem is STRICTLY CAUSAL (token j summarizes inputs at or before P*j+P-1; the last
    token j=N-1 summarizes inputs [L-P, L-1] = [120, 127] = at or before the current step t=127).
  MODERNTCN BLOCKS (x MTCN_BLOCKS=4): each block =
      x -> [ causal large-kernel DWConv (Conv1d groups=D, k=13, left-pad (k-1)*d + Chomp1d) -> BN ]
             + residual
        -> [ pointwise ConvFFN (1x1 Conv D->2D -> GELU -> 1x1 Conv 2D->D -> Dropout) ] + residual.
    The DWConv mixes across the token/time dim; the 1x1 ConvFFN mixes channels per-token. STRICT
    CAUSALITY: the DWConv left-pads (k-1)*d and Chomp1d slices off the (k-1)*d RIGHT (future) padded
    positions (Conv1d pads both sides -> out len N+(k-1)*d; Chomp -> out len N, left padding only), so
    token i depends only on tokens <= i. The 1x1 ConvFFN is per-token (no time mixing) -> trivially
    causal. BatchNorm normalizes over batch+length per channel, not across future time within a
    sample's forward path, so it does not leak the future into the current-step readout.
  HEAD: read the activation at the LAST token (index -1, = most-recent step t because the input is
    right-aligned) -> Linear(D->1) -> 1 logit. *** NO global/adaptive pooling *** (keeps changepoint
    LOCATION, exactly like E87 and unlike the pooled CNNs E83/E84).
  RECEPTIVE FIELD: token-space RF over 4 blocks (k=13, d=1) = 1 + 4*(13-1) = 49 tokens >= N=16, so the
    last token aggregates ALL 16 tokens = the full 128-input window; input-space coverage >> L_SEQ.

TRAINING CONFIG (identical to E83/E87): BCEWithLogitsLoss(pos_weight=n_neg/n_pos), Adam lr1e-3 wd1e-4,
batch 512, epochs (smoke 3 / full 12), torch.manual_seed, same device/thread setup. w_mdn=0 still
reproduces the champion EXACTLY (the ModernTCN member only enters the blend at w>0).

AFFINE-INVARIANCE (hard constraint). z is history-standardized, hence invariant to x -> a*x+b to
float32 precision; the CUSUM path s = cumsum(z)/sqrt(n) is a LINEAR transform of z (no a/b dependence)
so s is affine-invariant too. The SMOKE self-test builds (seq,msk) for a random window and for 3.7*w+5.2
and asserts max|delta_seq| < 1e-6 and mask equality.

CAUSALITY (hard constraint). No timestep sees future values: (i) the patch stem uses kernel==stride,
no padding -> patch token j reads only inputs [P*j, P*j+P-1]; (ii) every DWConv is chomped to keep only
left padding -> token i depends on tokens <= i; (iii) the 1x1 ConvFFN is per-token; (iv) the head reads
the last token (step t) whose receptive field is entirely at or before t. An empirical check in the
smoke perturbs a later-patch input and asserts every EARLIER patch token's activation is unchanged.

BASE (always in, = E78 ranker+rf base79, byte-identical to the shipped CHAMPION):
  champ51 (AR8 full + AR8 trail120) + dep10 + ship12 + [AR8 trail60, AR8 trail200] x 3 scales.
  The ranker+rf halves are the imported _wloss build machinery. The RF is fit with NO sample_weight
  -> provably identical to the champion. Only the 3rd member and the blend arm are new.

BLEND (3-way fixed-CDF percentile, single arm w_mdn in [0,1)):
  wr = W_RANK*(1-w_mdn); wf = W_RF*(1-w_mdn); wm = w_mdn
  blend = wr*pct(srk,ref_rk) + wf*pct(srf,ref_rf) + wm*pct(smdn,ref_mdn)
  w_mdn = 0  ==  the 2-member champion (baseline), byte-identical.
Arms: w_mdn in {0.10, 0.20, 0.30}.

BELIEVE-RULE (low prior). Believe an arm only if mean_d >= +0.0015 AND positive on all screen seeds
(npos == total), then confirm on fresh seeds {31337,2025,555,99}.

Uses its OWN cache dir _seqmtcncusum_cache (the sd dict carries seq/mask matrices, NOT compatible with
_seqmtcn_cache / _seqtcn_cache / _seqcnn_cache / _struct79_cache / _armdn_cache). Do NOT delete those.
RUN ONE heavy job at a time (this one holds a torch ModernTCN -- keep it the only heavy job).
Run:
  SMOKE=1 PYTHONWARNINGS=ignore .venv/bin/python -u _seqmtcncusum_rankerrf.py   # tiny self-test + checks
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _seqmtcncusum_rankerrf.py \
      > _seqmtcncusum_rankerrf.out 2>&1 & disown
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
SEQMTCN_EPOCHS = 3 if SMOKE else 12     # fixed-epoch ModernTCN training (no early stop)
W_MDN_ARMS = [0.20] if SMOKE else [0.10, 0.20, 0.30]

MTCN_PATCH = 8                          # patch size = stride -> N = L_SEQ/8 = 16 non-overlapping tokens
MTCN_DIM = 32                           # token/channel width D (comparable to E83/E87's ~32 channels)
MTCN_BLOCKS = 4                         # number of ModernTCN (DWConv + ConvFFN) blocks
MTCN_KERNEL = 13                        # large-kernel depthwise Conv1d over the token/time dim (causal)
MTCN_FFN_MULT = 2                       # ConvFFN expansion: 1x1 D -> 2D -> GELU -> 1x1 2D -> D
MTCN_DROPOUT = 0.1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_seqmtcncusum_cache")   # OWN cache (sd carries seq/mask matrices)
RAW = os.path.join(HERE, "_seqmtcncusum_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_seqmtcncusum_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_seqmtcncusum_rankerrf.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- the ModernTCN (ICLR 2024) over channels=[normalized value, mask], length L_SEQ ---------------
class Chomp1d(nn.Module):
    """Remove the last `chomp_size` timesteps = the RIGHT (future) padding a Conv1d adds, so each
    depthwise conv is STRICTLY CAUSAL (only left padding remains -> output i depends on inputs <= i)."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class ModernTCNBlock(nn.Module):
    """ModernTCN 'DWConv + ConvFFN' block over the token/time dim (n, D, N).
      x -> [ causal large-kernel depthwise Conv1d (groups=D, left-pad (k-1)*d -> Chomp1d) -> BN ] + x
        -> [ pointwise ConvFFN: 1x1 Conv D->ffn_mult*D -> GELU -> 1x1 Conv ->D -> Dropout ]        + x
    The DWConv mixes across tokens (large kernel, depthwise); the 1x1 ConvFFN mixes channels per-token
    (no time mixing -> trivially causal). Left-pad+Chomp keeps the DWConv causal (token i sees tokens
    <= i)."""

    def __init__(self, dim, kernel_size, dilation, ffn_mult, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation, groups=dim)
        self.chomp = Chomp1d(pad)                       # trims the (k-1)*d RIGHT (future) padding
        self.bn = nn.BatchNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, ffn_mult * dim, 1)    # pointwise ConvFFN up-projection
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(ffn_mult * dim, dim, 1)    # pointwise ConvFFN down-projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.bn(self.chomp(self.dw(x)))             # causal depthwise token mixing
        x = x + y                                       # residual around DWConv
        z = self.drop(self.pw2(self.act(self.pw1(x))))  # pointwise ConvFFN channel mixing
        return x + z                                    # residual around ConvFFN


class _SeqMTCNNet(nn.Module):
    """ModernTCN encoder over channels=[normalized value z, mask], length L_SEQ, then a per-step head
    on the LAST token. Input (n, 2, L_SEQ) right-aligned -> logit (n,). Causal patch stem
    (Conv1d kernel=stride=P, no padding) -> N=L_SEQ/P non-overlapping patch tokens -> a stack of
    ModernTCNBlocks (causal DWConv + ConvFFN) -> head reads token -1 (= current step t). NO
    global/adaptive pooling: changepoint-location information is preserved (the E82 fix)."""

    def __init__(self, in_ch=2, dim=MTCN_DIM, blocks=MTCN_BLOCKS, patch=MTCN_PATCH,
                 kernel_size=MTCN_KERNEL, ffn_mult=MTCN_FFN_MULT, dropout=MTCN_DROPOUT):
        super().__init__()
        self.patch = int(patch)
        # causal patchify: kernel==stride, NO padding -> token j reads ONLY inputs [P*j, P*j+P-1]
        self.stem = nn.Conv1d(in_ch, dim, kernel_size=self.patch, stride=self.patch)
        self.blocks = nn.Sequential(*[
            ModernTCNBlock(dim, kernel_size, 1, ffn_mult, dropout) for _ in range(blocks)])
        self.head = nn.Linear(dim, 1)

    def encode(self, x):
        h = self.stem(x)                                # (n, D, N=L_SEQ/patch) non-overlapping tokens
        return self.blocks(h)                           # (n, D, N) length preserved (causal)

    def forward(self, x):
        h = self.encode(x)                              # (n, D, N)
        last = h[:, :, -1]                              # (n, D) -- most-recent token = current step t
        return self.head(last).squeeze(-1)              # (n,) -- 1 logit per step, no pooling


class SeqMTCN:
    """sklearn-like wrapper so prep_scores stays parallel to E83/E87. .fit(seq,msk,y) /
    .predict_proba(seq,msk) -> (n,2) with column 1 = P(break). Deterministic on CPU."""

    def __init__(self, seed, epochs=SEQMTCN_EPOCHS, lr=1e-3, weight_decay=1e-4,
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
        self.net = _SeqMTCNNet()
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
    """Right-aligned, temporally-ordered CAUSAL CUSUM PATH of the history-standardized trailing window
    of up to L_SEQ points, plus a 0/1 valid-position mask. The value channel is s = cumsum(z)/sqrt(n)
    where z = (ww - mu_h)/sd_h are affine-invariant standardized increments; s is the classic
    sqrt(n)-normalized CUSUM change-point statistic (linear in z -> still affine-invariant & causal).
    Returns (seq, msk) float32 length L_SEQ; all-zero for degenerate cases. `ctx` is unused (kept for
    signature parity with _row)."""
    seq = np.zeros(L_SEQ, dtype=np.float32)
    msk = np.zeros(L_SEQ, dtype=np.float32)
    mu_h, sd_h = refs["mu_h"], refs["sd_h"]
    w = np.asarray(w, dtype=np.float64)
    if len(w) < 2 or sd_h == 0.0 or not np.isfinite(sd_h) or not np.isfinite(mu_h):
        return seq, msk
    ww = w[-L_SEQ:]                                          # most recent up-to-L_SEQ online points
    z = (ww - mu_h) / math.sqrt(sd_h * sd_h + 1e-12)         # HISTORY-standardized => affine-invariant
    z = np.clip(z, -8.0, 8.0)                                # bounded standardized increments (affine-invariant)
    c = np.cumsum(z)                                        # STRICTLY CAUSAL running sum (elem t = sum z[0..t])
    s = c / np.sqrt(np.arange(1, len(z) + 1))              # sqrt(n)-standardized CUSUM path (classic CP statistic)
    s = np.clip(s, -8.0, 8.0).astype(np.float32)           # same [-8,8] clip as z, for numerical parity
    if not np.all(np.isfinite(s)):
        return np.zeros(L_SEQ, dtype=np.float32), np.zeros(L_SEQ, dtype=np.float32)
    n = len(s)
    seq[-n:] = s                                           # right-aligned CUSUM path (left-padded with zeros)
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
    """Fit ranker+rf (uniform weights == champion) + the learned ModernTCN member, and return per-fold
    (active, groups, prk, prf, pmdn) where prk/prf/pmdn are the fixed-CDF percentiles at full length."""
    Xtr, ytr = sd["Xtr"], sd["ytr"]

    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr, ytr)                                  # NO sample_weight -> identical to shipped champion
    ref_rf = np.sort(_proba(rf, Xtr))

    rows, lab, grp, _swo = sort_into_groups_w(sd["Rr"], sd["Rl"], sd["Rt"], None)
    rk = fit_ranker_w("rank_xendcg", rows, lab, grp, None)
    ref_rk = np.sort(_proba(rk, sd["Rr"]))

    tcn = SeqMTCN(seed, epochs=SEQMTCN_EPOCHS, n_jobs=N_JOBS)
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


def _arch_check():
    """Log the ModernTCN wiring + receptive field so the covers-L_SEQ constraint is auditable."""
    n_tok = L_SEQ // MTCN_PATCH
    rf_tok = 1 + MTCN_BLOCKS * (MTCN_KERNEL - 1)                 # dilation=1, token-space RF (tokens)
    rf_in = min(L_SEQ, (rf_tok - 1) * MTCN_PATCH + MTCN_PATCH)   # input-space coverage (capped at L)
    log(f"[mtcn-arch] input=cusum-path patch={MTCN_PATCH} stride={MTCN_PATCH} tokens={n_tok} dim={MTCN_DIM} "
        f"blocks={MTCN_BLOCKS} dw_kernel={MTCN_KERNEL} ffn_mult={MTCN_FFN_MULT} "
        f"dropout={MTCN_DROPOUT} token_RF={rf_tok}(>=tokens {n_tok}: {rf_tok >= n_tok}) "
        f"input_cover={rf_in}(>=L_SEQ {L_SEQ}: {rf_in >= L_SEQ}) head=last-token(no pooling)")
    return rf_in


def _causal_check():
    """Empirical strict-causality test: perturb a LATER-patch input and assert every EARLIER patch
    token's encoder activation is EXACTLY unchanged (an earlier token must not see a later input)."""
    torch.manual_seed(0)
    net = _SeqMTCNNet()
    net.eval()                                        # eval BN (running stats) -> per-sample deterministic
    x = torch.randn(4, 2, L_SEQ)
    ppos = 64                                         # perturb input position 64 = patch 64//8 = 8
    ptok = ppos // MTCN_PATCH                         # tokens < ptok cannot see position ppos
    with torch.no_grad():
        h0 = net.encode(x)                            # (4, D, N) token feature map
        x2 = x.clone()
        x2[:, :, ppos] += 7.0                         # inject a large "future" perturbation
        h1 = net.encode(x2)
    early = float((h1[:, :, :ptok] - h0[:, :, :ptok]).abs().max())   # must be 0 (causal)
    late = float((h1[:, :, ptok:] - h0[:, :, ptok:]).abs().max())    # may be >0
    lastd = float((h1[:, :, -1] - h0[:, :, -1]).abs().max())         # last token DOES react (alive)
    log(f"[causal-check] perturb input@{ppos} (patch {ptok}): earlier-token(<{ptok}) maxdiff={early:.2e} "
        f"later-token(>={ptok}) maxdiff={late:.2e} last-token maxdiff={lastd:.2e} "
        f"(earlier must be 0)")
    assert early < 1e-8, f"CAUSALITY FAILED: an earlier patch token saw a later input (maxdiff={early:.2e})"
    return early


def main():
    open(LOG, "w").close()
    log(f"=== MODERNTCN 3rd-member probe on ranker+rf base79  "
        f"{datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
        f"ship_scales={SHIP_SCALES}")
    log(f"NCOL={NCOL} L_SEQ={L_SEQ} mtcn_epochs={SEQMTCN_EPOCHS} torch={torch.__version__} "
        f"arms={W_MDN_ARMS}")
    _arch_check()
    _causal_check()
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
        f.write(f"# ModernTCN 3rd-member probe on ranker+rf base79  "
                f"{datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
                f"ship={SHIP_SCALES} L_SEQ={L_SEQ} mtcn_epochs={SEQMTCN_EPOCHS}\n")
        f.write(f"# 3rd member = ModernTCN (ICLR 2024): causal patch stem + large-kernel depthwise\n")
        f.write(f"# Conv + pointwise ConvFFN over the raw history-standardized trailing window; NO\n")
        f.write(f"# pooling, last-token head keeps changepoint LOCATION; w_mdn=0 == the champion.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds\n")
        f.write(f"# {{31337,2025,555,99}}. Prior VERY LOW: learned-rep E31/E45/E47 + shape E59/E79/E82 +\n")
        f.write(f"# pooled seq-CNN E83/E84 + dilated causal TCN E87 all NULL.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for arm, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md >= 0.0015 and npos == tot) else ""
            f.write(f"{arm:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
