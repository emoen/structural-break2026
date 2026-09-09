"""MODERNTCN + CONFIGURABLE MULTI-LENS INPUT probe: a LEARNED 3rd member on the E78 ranker+rf base79
champion. GENERALIZATION of _seqmtcncusum_rankerrf.py (E91 ModernTCN+CUSUM). The ModernTCN model, the
base79 ranker+rf machinery, the blend arms, the believe-rule, the seeds, the epochs are ALL identical
to _seqmtcncusum_rankerrf.py / _seqmtcn_rankerrf.py (E88). The ONLY thing generalized is the network's
INPUT CHANNELS: instead of a single hardcoded value lens (raw z in E88, cusum-path in E91), the value
channels are selected by the env var LENSES from a registry of causal, affine-invariant lenses. This
lets us run each input lens INDIVIDUALLY (one value channel + mask) then STACK several (multi-channel)
to test whether aggregating input lenses helps the blend where any single lens did not.

WHY MULTI-LENS. All WaveNet-family nets (E87 TCN, E88 ModernTCN, E89 Hyena, E90 S4) fed the SAME raw z
window and were ALL null; E91 fed the cusum path and (expected) null. base79 saturates the raw-window
info. The one axis never systematically swept is the INPUT LENS BASIS. This probe makes the lens set a
config knob so each causal/affine-invariant lens (raw value, cusum path, energy, innovations, |innov|,
online rank/PIT) can be tested alone AND stacked, on the identical ModernTCN + base79 harness.

THE LENS REGISTRY (each a function of the already-history-standardized, clipped z; length-n float32;
strictly causal -- value at position i depends ONLY on z[0..i] -- and affine-invariant to x->a*x+b, a>0,
because z is affine-invariant and every lens is a deterministic function of z):
  z     : identity, the raw normalized value.            [== E88 baseline]  clip[-8,8] (z already clipped)
  cusum : cumsum(z)/sqrt(arange(1,n+1)), clip[-8,8].     [== E91]           classic sqrt(n)-CUSUM path
  z2    : z*z, clip[0,64].                                energy/variance path (pointwise fn of z)
  dz    : backward diff z[i]-z[i-1], dz[0]=0, clip[-8,8]. innovations (linear in z)
  adz   : abs(dz), clip[0,8].                             volatility/ARCH magnitude
  rank  : online expanding PIT (#{j<=i:z[j]<=z[i]}-0.5)/(i+1), mapped [0,1]->[-1,1].  ordinal => invariant
  All are EXPANDING/backward (never use z[j] for j>i) so the causal-check earlier-token maxdiff is 0.

Parse:  LENSES = os.environ.get("LENSES","z").split(",")  (default "z" == E88 baseline).
  LENSES=z          -> in_ch 2 ([z, mask])              == E88 exactly
  LENSES=z2         -> in_ch 2 ([z2, mask])             z2 alone
  LENSES=z,z2,dz,rank -> in_ch 5 (4 value lenses + mask)  stacked
The model input is (B, len(LENSES)+1, L_SEQ): one right-aligned channel per lens + a final valid mask.

CACHE / OUTPUT ISOLATION. The cache dir + raw/summary/log ALL encode the lens set, e.g.
  _seqmtcnmulti_cache_z2,  _seqmtcnmulti_cache_z-dz,  ... so LENSES=z2 and LENSES=z,dz never collide.

THE MODERNTCN MODEL (unchanged from E88/E91 except in_ch). PATCH STEM Conv1d(in_ch=len(LENSES)+1, D=32,
kernel=stride=P=8, no padding) -> N=L_SEQ/8=16 causal patch tokens; MTCN_BLOCKS=4 (causal large-kernel
depthwise Conv1d k=13 + Chomp1d, BN, + pointwise ConvFFN D->2D->GELU->D); last-token head, NO pooling
(keeps changepoint LOCATION). Everything but in_ch is byte-identical to E88.

CAUSALITY (hard constraint). (i) patch stem kernel==stride, no padding -> token j reads only inputs
[P*j, P*j+P-1]; (ii) each DWConv chomped to left-padding only; (iii) 1x1 ConvFFN is per-token; (iv)
head reads the last token. The empirical [causal-check] perturbs input@64 and asserts every earlier
patch token (< token 8) is EXACTLY unchanged. NOTE: cusum & online-rank make lens value i depend on
z[0..i], so perturbing z@64 changes those lens values at positions >=64 -- that is fine; the net-side
check perturbs the built input tensor, so earlier tokens must be exactly 0 regardless of lens. A
[lens-causal] check additionally perturbs z@p and asserts every lens output at positions < p is 0.

AFFINE-INVARIANCE (hard constraint). z is history-standardized (invariant to x->a*x+b). Every lens is a
deterministic function of z, so seq(x) == seq(a*x+b) to float32 precision. The [affine-check] builds
(seq,msk) for a random window and for 3.7*w+5.2 and asserts per-lens max|delta| < 1e-6 and mask equality
(exercise it with a MULTI-lens set to cover every lens).

BASE (always in, = E78 ranker+rf base79, byte-identical to the shipped CHAMPION). RF fit with NO
sample_weight -> provably identical to the champion. w_mdn=0 reproduces the 2-member champion exactly
(the ModernTCN member only enters the blend at w>0), so base_auc is invariant to the lens set.

BLEND (3-way fixed-CDF percentile, single arm w_mdn in [0,1)):
  wr = W_RANK*(1-w_mdn); wf = W_RF*(1-w_mdn); wm = w_mdn
  blend = wr*pct(srk,ref_rk) + wf*pct(srf,ref_rf) + wm*pct(smdn,ref_mdn);  w_mdn=0 == champion.
Arms: w_mdn in {0.10, 0.20, 0.30}.

BELIEVE-RULE (low prior). Believe an arm only if mean_d >= +0.0015 AND positive on all screen seeds,
then confirm on fresh seeds {31337,2025,555,99}.

RUN ONE heavy job at a time (holds a torch ModernTCN). Run:
  SMOKE=1 LENSES=z            PYTHONWARNINGS=ignore .venv/bin/python -u _seqmtcnmulti_rankerrf.py  # == E88
  SMOKE=1 LENSES=z,z2,dz,adz,rank PYTHONWARNINGS=ignore .venv/bin/python -u _seqmtcnmulti_rankerrf.py
  nohup env PYTHONWARNINGS=ignore LENSES=z2 N_JOBS=15 .venv/bin/python -u _seqmtcnmulti_rankerrf.py \
      > _seqmtcnmulti_rankerrf_z2.out 2>&1 & disown
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
L_SEQ = 128                             # fixed length of the right-aligned window sequence
SEQMTCN_EPOCHS = 3 if SMOKE else int(os.environ.get("MTCN_EPOCHS", "12"))     # fixed-epoch training (env-tunable)
W_MDN_ARMS = [0.20] if SMOKE else [0.10, 0.20, 0.30]

MTCN_PATCH = 8                          # patch size = stride -> N = L_SEQ/8 = 16 non-overlapping tokens
MTCN_DIM = int(os.environ.get("MTCN_DIM", "32"))      # token/channel width D (env-tunable for capacity probes)
MTCN_BLOCKS = int(os.environ.get("MTCN_BLOCKS", "4")) # number of ModernTCN (DWConv + ConvFFN) blocks (env-tunable)
MTCN_KERNEL = 13                        # large-kernel depthwise Conv1d over the token/time dim (causal)
MTCN_FFN_MULT = 2                       # ConvFFN expansion: 1x1 D -> 2D -> GELU -> 1x1 2D -> D
MTCN_DROPOUT = 0.1

# OPTIONAL synthetic-pretrain -> fine-tune path (default OFF -> byte-identical to before).
PRETRAIN = os.environ.get("PRETRAIN") == "1"            # 1 -> pretrain ModernTCN on synth, warm-start real fit
SYNTH_DIR = os.environ.get("SYNTH_DIR", "synthetic")    # competition-format synthetic dataset dir
PRE_EPOCHS = 2 if SMOKE else int(os.environ.get("PRE_EPOCHS", "8"))       # pretrain epochs
PRE_STEPS = 4 if SMOKE else int(os.environ.get("PRE_STEPS", "20"))        # windows sampled per synth series
PRE_MAXSERIES = 50 if SMOKE else int(os.environ.get("PRE_MAXSERIES", "0"))  # 0 = all synth series; else cap


# ================================ THE LENS REGISTRY ==============================================
# Each lens maps the history-standardized, clipped z (1-D float64, length n) to a length-n array.
# STRICTLY CAUSAL: value at position i uses only z[0..i]. AFFINE-INVARIANT: z is invariant to
# x->a*x+b (a>0) and each lens is a deterministic function of z. Per-lens clipping is done HERE (so
# _tensor must NOT blanket-clip -- that would destroy z2's [0,64] range).
def _lens_z(z):
    """Identity: the raw history-standardized value. [== E88 baseline]  z already in [-8,8]."""
    return np.clip(z, -8.0, 8.0)


def _lens_cusum(z):
    """Classic sqrt(n)-standardized CUSUM PATH s = cumsum(z)/sqrt(arange(1,n+1)). [== E91]
    Linear in z (affine-invariant); cumsum is strictly causal."""
    c = np.cumsum(z)                                     # elem i = sum z[0..i] (causal running sum)
    s = c / np.sqrt(np.arange(1, len(z) + 1))            # 1-based sqrt(n) standardization (no div0)
    return np.clip(s, -8.0, 8.0)


def _lens_z2(z):
    """Energy/variance path z*z. Pointwise fn of z -> causal + affine-invariant. clip[0,64]
    (z in [-8,8] -> z2 in [0,64])."""
    return np.clip(z * z, 0.0, 64.0)


def _lens_dz(z):
    """Innovations = backward first difference z[i]-z[i-1] with dz[0]=0. Causal (uses z[i],z[i-1]
    only), affine-invariant (linear in z). clip[-8,8]."""
    dz = np.empty_like(z)
    dz[0] = 0.0
    dz[1:] = z[1:] - z[:-1]
    return np.clip(dz, -8.0, 8.0)


def _lens_adz(z):
    """Volatility/ARCH magnitude = |backward diff|. Causal + affine-invariant. clip[0,8]."""
    dz = np.empty_like(z)
    dz[0] = 0.0
    dz[1:] = z[1:] - z[:-1]
    return np.clip(np.abs(dz), 0.0, 8.0)


def _lens_rank(z):
    """Online EXPANDING rank / empirical-CDF percentile (PIT): for each i,
    rank[i] = (#{j<=i : z[j] <= z[i]} - 0.5)/(i+1)  in (0,1], mapped to [-1,1]. STRICTLY EXPANDING
    (only z[0..i]) -> causal. Ordinal -> invariant to any increasing transform (a>0). O(n^2) but
    vectorized over the lower triangle (n<=128)."""
    n = len(z)
    cmp = (z[None, :] <= z[:, None])                     # cmp[i,j] = (z[j] <= z[i])
    tri = np.tril(np.ones((n, n), dtype=bool))           # tri[i,j] = 1 if j <= i (expanding window)
    cnt = (cmp & tri).sum(axis=1).astype(np.float64)     # #{j<=i : z[j] <= z[i]}  (>=1, includes j=i)
    pit = (cnt - 0.5) / np.arange(1, n + 1)              # online PIT in (0,1]
    return 2.0 * pit - 1.0                               # center/scale to [-1,1]


LENS_FUNCS = {
    "z": _lens_z,
    "cusum": _lens_cusum,
    "z2": _lens_z2,
    "dz": _lens_dz,
    "adz": _lens_adz,
    "rank": _lens_rank,
}

LENSES = [s.strip() for s in os.environ.get("LENSES", "z").split(",") if s.strip()]
if not LENSES:
    LENSES = ["z"]
for _ln in LENSES:
    if _ln not in LENS_FUNCS:
        raise ValueError(f"unknown lens {_ln!r}; available: {sorted(LENS_FUNCS)}")
N_LENS = len(LENSES)                    # number of value channels
IN_CH = N_LENS + 1                      # value channels + 1 mask channel = net input channels
LENS_TAG = "-".join(LENSES)             # encodes the lens set into cache/output paths


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, f"_seqmtcnmulti_cache_{LENS_TAG}")   # per-lens-set cache (sd carries seq/mask)
RAW = os.path.join(HERE, f"_seqmtcnmulti_rankerrf_raw_{LENS_TAG}.tsv")
SUMMARY = os.path.join(HERE, f"_seqmtcnmulti_rankerrf_summary_{LENS_TAG}.txt")
LOG = os.path.join(HERE, f"_seqmtcnmulti_rankerrf_{LENS_TAG}.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- the ModernTCN (ICLR 2024) over channels=[value lenses..., mask], length L_SEQ ----------------
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
    """ModernTCN encoder over channels=[value lenses..., mask], length L_SEQ, then a per-step head on
    the LAST token. Input (n, IN_CH, L_SEQ) right-aligned -> logit (n,). Causal patch stem
    (Conv1d kernel=stride=P, no padding) -> N=L_SEQ/P non-overlapping patch tokens -> a stack of
    ModernTCNBlocks (causal DWConv + ConvFFN) -> head reads token -1 (= current step t). NO
    global/adaptive pooling: changepoint-location information is preserved (the E82 fix)."""

    def __init__(self, in_ch=IN_CH, dim=MTCN_DIM, blocks=MTCN_BLOCKS, patch=MTCN_PATCH,
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
    .predict_proba(seq,msk) -> (n,2) with column 1 = P(break). Deterministic on CPU. seq is
    (n, N_LENS, L_SEQ) (one channel per lens); msk is (n, L_SEQ)."""

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
        # seq (n, N_LENS, L_SEQ) already per-lens clipped in the registry; msk (n, L_SEQ). No blanket
        # clip here (would destroy z2's [0,64] range). Append the mask as the final channel.
        seq = np.asarray(seq, dtype=np.float32)
        msk = np.asarray(msk, dtype=np.float32)
        x = np.concatenate([seq, msk[:, None, :]], axis=1)         # (n, N_LENS+1, L_SEQ)
        return torch.from_numpy(np.ascontiguousarray(x))

    def fit(self, seq, msk, y, init_state=None):
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_jobs)
        seq = np.asarray(seq, dtype=np.float32)
        msk = np.asarray(msk, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        self.net = _SeqMTCNNet(in_ch=IN_CH)
        if init_state is not None:
            self.net.load_state_dict(init_state)     # WARM-START from pretrained weights (fine-tune)
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


# --- the learned member's per-step representation (multi-lens history-normalized window sequence) --
def _seqrow(w, ctx, refs):
    """Right-aligned, temporally-ordered CAUSAL multi-lens view of the history-standardized trailing
    window of up to L_SEQ points, plus a 0/1 valid-position mask. z = (ww - mu_h)/sd_h are the
    affine-invariant standardized increments; each selected lens is a causal, affine-invariant
    deterministic function of z (see the LENS REGISTRY). Returns (seq, msk): seq is (N_LENS, L_SEQ)
    float32 (one right-aligned row per lens), msk is (L_SEQ,) float32; all-zero for degenerate cases.
    `ctx` is unused (kept for signature parity with _row)."""
    seq = np.zeros((N_LENS, L_SEQ), dtype=np.float32)
    msk = np.zeros(L_SEQ, dtype=np.float32)
    mu_h, sd_h = refs["mu_h"], refs["sd_h"]
    w = np.asarray(w, dtype=np.float64)
    if len(w) < 2 or sd_h == 0.0 or not np.isfinite(sd_h) or not np.isfinite(mu_h):
        return seq, msk
    ww = w[-L_SEQ:]                                          # most recent up-to-L_SEQ online points
    z = (ww - mu_h) / math.sqrt(sd_h * sd_h + 1e-12)         # HISTORY-standardized => affine-invariant
    z = np.clip(z, -8.0, 8.0)                                # bounded standardized increments (base for lenses)
    n = len(z)
    for li, name in enumerate(LENSES):
        v = np.asarray(LENS_FUNCS[name](z), dtype=np.float32)    # causal + affine-invariant lens of z
        if not np.all(np.isfinite(v)):
            return np.zeros((N_LENS, L_SEQ), dtype=np.float32), np.zeros(L_SEQ, dtype=np.float32)
        seq[li, -n:] = v                                    # right-aligned (left-padded with zeros)
    msk[-n:] = 1.0
    return seq, msk


def _seqsan(a):
    """Sanitize a stack of seq rows to a finite (n, N_LENS, L_SEQ) float32 tensor (mirrors _san)."""
    if not len(a):
        return np.zeros((0, N_LENS, L_SEQ), dtype=np.float32)
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _msksan(a):
    """Sanitize a stack of mask rows to a finite (n, L_SEQ) float32 matrix."""
    if not len(a):
        return np.zeros((0, L_SEQ), dtype=np.float32)
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# --- extended train/val builders that ALSO emit seq/mask aligned row-for-row with the base79 rows --
def _train_series_q(sid, x_h, x_on, tau, steps):
    ctx = hist_context(x_h, HIST_CAP)
    active = _active(x_on)
    if len(active) == 0:
        return (np.zeros((0, NCOL)), [], [], [],
                np.zeros((0, N_LENS, L_SEQ), np.float32), np.zeros((0, L_SEQ), np.float32))
    refs = _refs_ext(x_h)
    base = active[0]
    rows, labels, tvals, seqs, msks = [], [], [], [], []
    for t in steps:
        i = t - base
        if i < 0 or i >= len(active):
            continue
        w = x_on[max(0, t - WINDOW + 1):t + 1]
        rows.append(_row(w, ctx, refs))              # base79 row (identical to _wloss)
        s, m = _seqrow(w, ctx, refs)                 # aligned multi-lens sequence + mask
        seqs.append(s)
        msks.append(m)
        labels.append(1 if (tau >= 0 and t >= tau) else 0)
        tvals.append(int(t))
    return (_san(rows), labels, tvals, [int(sid)] * len(labels),
            _seqsan(seqs), _msksan(msks))


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
    msk = _msksan(msks)
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
                np.zeros(0, np.int64), np.zeros((0, N_LENS, L_SEQ), np.float32),
                np.zeros((0, L_SEQ), np.float32))
    feat = np.vstack([r for r, _l, _t, _s, _q, _m in keep])
    lab = np.concatenate([l for _r, l, _t, _s, _q, _m in keep])
    tv = np.concatenate([t for _r, _l, t, _s, _q, _m in keep])
    sids = np.concatenate([s for _r, _l, _t, s, _q, _m in keep])
    SEQ = np.vstack([q for _r, _l, _t, _s, q, _m in keep])      # (n, N_LENS, L_SEQ)
    MSK = np.vstack([m for _r, _l, _t, _s, _q, m in keep])      # (n, L_SEQ)
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
        seqs.append(seq if seq.shape[0] else np.zeros((0, N_LENS, L_SEQ), np.float32))
        msks.append(msk if msk.shape[0] else np.zeros((0, L_SEQ), np.float32))
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    feat = np.vstack(feats) if feats else np.zeros((0, NCOL))
    seq = np.vstack(seqs) if seqs else np.zeros((0, N_LENS, L_SEQ), np.float32)
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


# --- OPTIONAL: pretrain the ModernTCN member on the synthetic labeled dataset (returns a warm-start
#     state_dict reused across all seeds; a no-op when PRETRAIN is unset since main() never calls it) --
def pretrain_synth_mtcn(seed):
    """Pretrain a fresh ModernTCN (identical architecture -- same IN_CH/MTCN_DIM/MTCN_BLOCKS globals)
    on synthetic multi-lens window sequences, then return its CPU state_dict for warm-starting the real
    per-seed fit. Labels come from build_train_q (_train_series_q labels a step 1 iff tau>=0 and t>=tau,
    so no-break synth series (tau_index=-1) contribute ALL-ZERO labels -- pos_rate is the sanity check)."""
    Xs = pd.read_parquet(os.path.join(SYNTH_DIR, "X_train.parquet"))
    yis = pd.read_parquet(os.path.join(SYNTH_DIR, "y_train_index.parquet"))
    synth_ids = list(yis.index)
    if PRE_MAXSERIES > 0:
        synth_ids = synth_ids[:PRE_MAXSERIES]              # deterministic cap for speed
    rng = np.random.default_rng(seed + 909)
    synth_steps = {}
    for sid in synth_ids:                                  # same rank_steps pattern as prepare_seed
        L = len(split_series(Xs.loc[sid])[1].to_numpy())
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        synth_steps[sid] = rng.choice(cand, size=min(PRE_STEPS, len(cand)), replace=False)
    _feat, lab, _tv, _sids, SEQ, MSK = build_train_q(Xs, yis, synth_steps)
    if len(lab) == 0:
        log(f"[pretrain] WARNING: 0 windows from synth={SYNTH_DIR} -> no pretrain")
        return None
    net = SeqMTCN(seed, epochs=PRE_EPOCHS, n_jobs=N_JOBS)
    net.fit(SEQ, MSK, lab)
    log(f"[pretrain] synth={SYNTH_DIR} series={len(synth_steps)} windows={len(lab)} "
        f"pos_rate={float(np.mean(lab)):.3f} pre_epochs={PRE_EPOCHS} dim={MTCN_DIM} "
        f"blocks={MTCN_BLOCKS} in_ch={IN_CH}")
    return {k: v.detach().cpu().clone() for k, v in net.net.state_dict().items()}


# --- fit the 3 members ONCE per seed, then evaluate every w_mdn arm from cached percentiles -------
def prep_scores(sd, metas, seed, pre_state=None):
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
    tcn.fit(sd["Str_seq"], sd["Str_msk"], ytr, init_state=pre_state)
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
    """seq(x) must equal seq(a*x+b) for a>0 to float32 precision, per lens (asserts max|delta| < 1e-6
    and identical masks). Run with a MULTI-lens set to exercise every lens."""
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
    mask_ok = bool(np.array_equal(msk1, msk2))
    md_all = float(np.max(np.abs(seq1.astype(np.float64) - seq2.astype(np.float64)))) if N_LENS else 0.0
    for li, name in enumerate(LENSES):
        mdl = float(np.max(np.abs(seq1[li].astype(np.float64) - seq2[li].astype(np.float64))))
        nz = int(np.sum(np.abs(seq1[li]) > 1e-9))
        log(f"[affine-check] lens {name:<6} maxdiff under x->{a}x+{b}: {mdl:.2e}  (nonzero {nz}/{L_SEQ})")
    log(f"[affine-check] ALL lenses={LENS_TAG} maxdiff={md_all:.2e}  mask_eq={mask_ok}")
    assert mask_ok, "AFFINE INVARIANCE FAILED: masks differ under x->a*x+b"
    assert md_all < 1e-6, f"AFFINE INVARIANCE FAILED: max|delta|={md_all:.2e} >= 1e-6 (a lens is wrong)"
    return md_all


def _lens_causal_check():
    """Lens-level strict causality: perturb z at position p and assert EVERY lens output at positions
    < p is EXACTLY unchanged (position i must depend only on z[0..i])."""
    rng = np.random.default_rng(3)
    z = np.clip(rng.standard_normal(100), -8.0, 8.0)
    p = 40
    z2 = z.copy()
    z2[p] += 5.0                                          # perturb only position p (a "future" input)
    worst = 0.0
    for name in LENSES:
        a = np.asarray(LENS_FUNCS[name](z), dtype=np.float64)
        c = np.asarray(LENS_FUNCS[name](z2), dtype=np.float64)
        d = float(np.max(np.abs(a[:p] - c[:p]))) if p else 0.0
        worst = max(worst, d)
        log(f"[lens-causal] {name:<6} positions<{p} maxdiff={d:.2e}  (must be 0)")
        assert d < 1e-9, f"LENS CAUSALITY FAILED for {name}: earlier position changed (maxdiff={d:.2e})"
    return worst


def _arch_check():
    """Log the ModernTCN wiring + receptive field so the covers-L_SEQ constraint is auditable."""
    n_tok = L_SEQ // MTCN_PATCH
    rf_tok = 1 + MTCN_BLOCKS * (MTCN_KERNEL - 1)                 # dilation=1, token-space RF (tokens)
    rf_in = min(L_SEQ, (rf_tok - 1) * MTCN_PATCH + MTCN_PATCH)   # input-space coverage (capped at L)
    log(f"[mtcn-arch] lenses={LENS_TAG} in_ch={IN_CH} patch={MTCN_PATCH} stride={MTCN_PATCH} "
        f"tokens={n_tok} dim={MTCN_DIM} blocks={MTCN_BLOCKS} dw_kernel={MTCN_KERNEL} "
        f"ffn_mult={MTCN_FFN_MULT} dropout={MTCN_DROPOUT} token_RF={rf_tok}(>=tokens {n_tok}: "
        f"{rf_tok >= n_tok}) input_cover={rf_in}(>=L_SEQ {L_SEQ}: {rf_in >= L_SEQ}) "
        f"head=last-token(no pooling)")
    return rf_in


def _causal_check():
    """Empirical strict-causality test on the NET: perturb a LATER-patch input and assert every
    EARLIER patch token's encoder activation is EXACTLY unchanged (an earlier token must not see a
    later input). Independent of the lens (perturbs the built input tensor directly)."""
    torch.manual_seed(0)
    net = _SeqMTCNNet(in_ch=IN_CH)
    net.eval()                                        # eval BN (running stats) -> per-sample deterministic
    x = torch.randn(4, IN_CH, L_SEQ)
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
    log(f"=== MODERNTCN MULTI-LENS 3rd-member probe on ranker+rf base79  "
        f"{datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"lenses={LENSES} in_ch={IN_CH} cache={os.path.basename(CACHE_DIR)}")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
        f"ship_scales={SHIP_SCALES}")
    log(f"NCOL={NCOL} L_SEQ={L_SEQ} mtcn_epochs={SEQMTCN_EPOCHS} torch={torch.__version__} "
        f"arms={W_MDN_ARMS}")
    log(f"PRETRAIN={int(PRETRAIN)} synth_dir={SYNTH_DIR} pre_epochs={PRE_EPOCHS} "
        f"pre_steps={PRE_STEPS} pre_maxseries={PRE_MAXSERIES}")
    _arch_check()
    _causal_check()
    _lens_causal_check()
    _affine_check()

    if not os.path.exists(RAW):
        with open(RAW, "w") as f:
            f.write("seed\tarm\tbase_auc\tcand_auc\tdelta\n")

    pre_state = None
    if PRETRAIN:
        tp = time.time()
        pre_state = pretrain_synth_mtcn(SEARCH_SEEDS[0])   # pretrain ONCE, reuse warm-start across seeds
        log(f"[pretrain] done in {(time.time()-tp)/60:.1f} min, state={'None' if pre_state is None else 'ok'}")

    per_seed = {}
    for seed in SEARCH_SEEDS:
        sd = prepare_seed(seed)
        metas = make_folds_meta(sd)
        t0 = time.time()
        folds = prep_scores(sd, metas, seed, pre_state)
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
        f.write(f"# ModernTCN MULTI-LENS 3rd-member probe on ranker+rf base79  "
                f"{datetime.now():%F %T}\n")
        f.write(f"# lenses={LENS_TAG} in_ch={IN_CH} seeds={SEARCH_SEEDS} rf_trees={RF_TREES} "
                f"dense_k={DENSE_K} blend={W_RANK}/{W_RF} ship={SHIP_SCALES} L_SEQ={L_SEQ} "
                f"mtcn_epochs={SEQMTCN_EPOCHS}\n")
        f.write(f"# 3rd member = ModernTCN (ICLR 2024): causal patch stem + large-kernel depthwise\n")
        f.write(f"# Conv + pointwise ConvFFN over the selected causal/affine-invariant input lenses;\n")
        f.write(f"# NO pooling, last-token head keeps changepoint LOCATION; w_mdn=0 == the champion.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds\n")
        f.write(f"# {{31337,2025,555,99}}. Prior VERY LOW: learned-rep window nets E83/84/87/88/89/90 +\n")
        f.write(f"# cusum-lens E91 all NULL.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for arm, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md >= 0.0015 and npos == tot) else ""
            f.write(f"{arm:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
