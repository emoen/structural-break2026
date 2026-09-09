"""HYENA-OPERATOR (implicit long conv) probe (P_F): a LEARNED 3rd member on the E78 ranker+rf base79
champion. CLONE of _seqtcn_rankerrf.py (P_D / E87); the ONLY change is the neural model -- the Bai-2018
dilated causal TCN is replaced by a causal Hyena operator (Poli et al. 2023, "Hyena Hierarchy": an
implicit long convolution parameterized by a positional-encoding MLP, gated by short causal depthwise
convs). Everything else (base79 ranker+rf machinery, the causal history-normalized sequence input z +
mask, the affine-invariance self-test, the blend arms, the believe-rule, the cache format) is
byte-identical to _seqtcn_rankerrf.py.

WHY THIS ARCHITECTURE. E87 (the dilated causal TCN) and E88 (ModernTCN) were the most defensible
learned-rep variants -- both preserved temporal ORDER and changepoint LOCATION (the axis the E82
diagnostic flagged) and still went NULL (E59/E79/E82/E83/E84/E87/E88 all converging). The Hyena operator
is the last distinct long-conv family in the WaveNet-family backlog: instead of a stack of narrow
dilated convs (E87) or large-kernel depthwise convs over patch tokens (E88), it learns a SINGLE
implicit long convolution filter of the FULL window length L=128 -- the filter taps are the output of a
small positional MLP, so the receptive field covers the whole window in one layer while remaining
strictly causal. Prior is VERY LOW (the bottleneck looks like base79 information saturation, not
architecture) -- this is a SMOKE-first completeness probe, not a high-prior bet.

INPUT (unchanged from E83/E87/E88). Per online step t, causal (history x_h + x_on[:t+1] only):
    1. history location/scale ONCE per series:  mu_h = mean(x_h), sd_h = std(x_h)  (stored in refs).
    2. trailing window  w = x_on[max(0, t-WINDOW+1):t+1]; take its most recent up-to-L_SEQ points.
    3. NORMALIZE BY HISTORY STATS:  z = (w - mu_h) / sqrt(sd_h**2 + 1e-12)  (affine-invariant to
       x -> a*x+b, a>0, and keeps a mean/variance/AR change inside the window VISIBLE).
    4. LEFT-PAD to L_SEQ, right-aligned, with a companion 0/1 mask marking valid positions; clip
       [-8,8]. Degenerate windows emit an all-zero seq + all-zero mask.
  The net's 2 input channels are [normalized value z, valid-mask], length L_SEQ, right-aligned so the
  CURRENT step t is always at index -1.

THE HYENA MODEL (the only real change vs _seqtcn_rankerrf.py):
  INPUT PROJECTION (causal): Conv1d(in_ch=2, D=32, kernel_size=1) -- pointwise, trivially causal, maps
    the 2-channel input to the model width D=32.
  HYENA BLOCKS (x HY_BLOCKS=2): each block =
      x -> [ HyenaOperator(x) -> BatchNorm1d ] + residual
        -> [ pointwise ConvFFN: 1x1 Conv D->2D -> GELU -> 1x1 Conv 2D->D -> Dropout ] + residual.
    The HyenaOperator is the order-2 implicit-long-conv token mixer; the 1x1 ConvFFN mixes channels
    per-token (no time mixing -> trivially causal).  BatchNorm normalizes per channel over batch+length,
    not across future time within a single sample's forward path, so it does not leak the future into
    the current-step readout.
  HYENA OPERATOR (order HY_ORDER=2, the core):
    * input projection Conv1d(D -> (order+1)*D, 1) -> split into (order+1) depthwise groups
      x1, x2, ..., v (here 3 groups: two gate signals + one value);
    * a SHORT causal depthwise Conv1d (k=3, left-pad (k-1) + Chomp1d) processes those groups;
    * IMPLICIT LONG FILTER h[t] of length L_SEQ=128 per channel: a positional grid t=0..L-1 encoded as
      [t/L, sin(2*pi*k*t/L), cos(2*pi*k*t/L) for k=1..HY_POS_FREQS] is fed through a 2-layer MLP
      (pos_feats -> HY_FILTER_MLP=64 -> GELU -> D) to produce the D-channel filter, multiplied by a
      learnable per-channel exponential decay window exp(-softplus(alpha)*lag/L) (lag 0 = most recent =
      weight 1, decaying with lag) to keep it stable and causal-emphasized;
    * CAUSAL LONG CONVOLUTION via DIRECT F.conv1d (NOT FFT): the depthwise filter is flipped to
      lag-order and applied with left-padding L_SEQ-1 then the trailing (future) L_SEQ-1 positions are
      chomped -- provably strictly causal (output i depends only on inputs <= i), avoiding any FFT
      circular-conv indexing pitfalls;
    * GATING recurrence (order 2): v <- x_o .* longconv(h_o, v) for o=1..order, i.e.
      x2 .* Conv(h2, x1 .* Conv(h1, v)); a final 1x1 out_proj + dropout.
  HEAD: read the activation at the LAST timestep (index -1, = most-recent step t because the input is
    right-aligned) -> Linear(D->1) -> 1 logit. *** NO global/adaptive pooling *** (keeps changepoint
    LOCATION, exactly like E87/E88 and unlike the pooled CNNs E83/E84).
  RECEPTIVE FIELD: the implicit long conv filter has length L_SEQ=128 = the FULL window in a single
    operator, so the last-step readout covers all 128 inputs.

TRAINING CONFIG (identical to E83/E87/E88): BCEWithLogitsLoss(pos_weight=n_neg/n_pos), Adam lr1e-3
wd1e-4, batch 512, epochs (smoke 3 / full 12), torch.manual_seed, same device/thread setup. w_mdn=0
still reproduces the champion EXACTLY (the Hyena member only enters the blend at w>0).

AFFINE-INVARIANCE (hard constraint). z is history-standardized, hence invariant to x -> a*x+b to
float32 precision. The SMOKE self-test builds (seq,msk) for a random window and for 3.7*w+5.2 and
asserts max|delta_seq| < 1e-6 and mask equality. The input builder is byte-copied from E87.

CAUSALITY (hard constraint). No timestep sees future values: (i) the input projection is 1x1
(pointwise); (ii) the short depthwise conv left-pads + chomps -> token i sees tokens <= i; (iii) the
implicit long conv left-pads L_SEQ-1 + chomps the future tail -> output i depends only on inputs <= i;
(iv) gating is elementwise (per-token) and the 1x1 ConvFFN is per-token; (v) the head reads the last
step (step t) whose receptive field is entirely at or before t. An empirical check in the smoke
perturbs a LATER input position and asserts every EARLIER output position is EXACTLY unchanged
(maxdiff 0.00e+00).

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

Uses its OWN cache dir _seqhyena_cache (the sd dict carries seq/mask matrices, NOT compatible with
_seqtcn_cache / _seqmtcn_cache / _seqcnn_cache / _struct79_cache / _armdn_cache). Do NOT delete those.
RUN ONE heavy job at a time (this one holds a torch Hyena net -- keep it the only heavy job).
Run:
  SMOKE=1 PYTHONWARNINGS=ignore .venv/bin/python -u _seqhyena_rankerrf.py    # tiny self-test + checks
  nohup env PYTHONWARNINGS=ignore N_JOBS=15 .venv/bin/python -u _seqhyena_rankerrf.py \
      > _seqhyena_rankerrf.out 2>&1 & disown
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
import torch.nn.functional as F
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
SEQHYENA_EPOCHS = 3 if SMOKE else 12    # fixed-epoch Hyena training (no early stop)
W_MDN_ARMS = [0.20] if SMOKE else [0.10, 0.20, 0.30]

HY_DIM = 32                             # model width D (matches E83/E87/E88's ~32 channels)
HY_BLOCKS = 2                           # number of Hyena (operator + ConvFFN) blocks
HY_ORDER = 2                            # Hyena recurrence order (# implicit long filters + # gates)
HY_SHORT_K = 3                          # short causal depthwise conv kernel on the projected groups
HY_FFN_MULT = 2                         # ConvFFN expansion: 1x1 D -> 2D -> GELU -> 1x1 2D -> D
HY_FILTER_MLP = 64                      # hidden width of the positional filter MLP
HY_POS_FREQS = 8                        # # sin/cos frequency pairs in the positional grid features
HY_DROPOUT = 0.1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "_seqhyena_cache")   # OWN cache (sd carries seq/mask matrices)
RAW = os.path.join(HERE, "_seqhyena_rankerrf_raw.tsv")
SUMMARY = os.path.join(HERE, "_seqhyena_rankerrf_summary.txt")
LOG = os.path.join(HERE, "_seqhyena_rankerrf.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


# --- the causal Hyena operator (Poli et al. 2023) over channels=[normalized value, mask], len L_SEQ --
class Chomp1d(nn.Module):
    """Remove the last `chomp_size` timesteps = the RIGHT (future) padding a Conv1d adds, so a conv is
    STRICTLY CAUSAL (only left padding remains -> output i depends on inputs <= i)."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class HyenaFilter(nn.Module):
    """Implicit long convolution filter h of length seq_len per channel: a positional grid t=0..L-1
    encoded as [t/L, sin(2*pi*k*t/L), cos(2*pi*k*t/L) for k=1..n_freqs] passed through a 2-layer MLP
    (pos_feats -> mlp_hidden -> GELU -> dim), multiplied by a learnable per-channel exponential decay
    window exp(-softplus(alpha)*lag/L) (lag 0 = most recent = weight 1). Returns (dim, L) indexed by
    lag t; the long conv flips it to conv (cross-correlation) order."""

    def __init__(self, dim, seq_len, mlp_hidden=HY_FILTER_MLP, n_freqs=HY_POS_FREQS):
        super().__init__()
        self.dim = int(dim)
        self.seq_len = int(seq_len)
        t = torch.arange(self.seq_len, dtype=torch.float32)
        feats = [(t / self.seq_len).unsqueeze(1)]                 # normalized absolute position
        for k in range(1, n_freqs + 1):
            feats.append(torch.sin(2.0 * math.pi * k * t / self.seq_len).unsqueeze(1))
            feats.append(torch.cos(2.0 * math.pi * k * t / self.seq_len).unsqueeze(1))
        pos = torch.cat(feats, dim=1)                             # (L, 1 + 2*n_freqs)
        self.register_buffer("pos", pos)
        lag = (t / self.seq_len).unsqueeze(0)                     # (1, L) lag grid for decay
        self.register_buffer("lag", lag)
        self.mlp = nn.Sequential(nn.Linear(pos.shape[1], mlp_hidden), nn.GELU(),
                                 nn.Linear(mlp_hidden, self.dim))
        self.log_alpha = nn.Parameter(torch.zeros(self.dim))     # per-channel decay (softplus >= 0)

    def forward(self):
        h = self.mlp(self.pos).transpose(0, 1)                   # (dim, L) filter taps by lag
        alpha = F.softplus(self.log_alpha).unsqueeze(1)          # (dim, 1) >= 0
        decay = torch.exp(-alpha * self.lag)                     # (dim, L) emphasize recent lags
        return h * decay                                         # (dim, L)


class HyenaOperator(nn.Module):
    """Order-`order` causal Hyena operator over (n, dim, L). Projects the input into (order+1) depthwise
    groups (x1..x_order gates + value v), passes them through a SHORT causal depthwise conv, then
    applies the implicit-long-conv recurrence v <- x_o .* longconv(h_o, v). All ops are causal: 1x1
    projections are pointwise, the short conv is left-pad+chomp, the long conv is left-pad(L-1)+chomp
    (direct F.conv1d, provably strictly causal -- no FFT circular-conv risk), gating is elementwise."""

    def __init__(self, dim, seq_len, order=HY_ORDER, short_k=HY_SHORT_K, dropout=HY_DROPOUT):
        super().__init__()
        self.dim = int(dim)
        self.seq_len = int(seq_len)
        self.order = int(order)
        self.short_pad = int(short_k - 1)
        self.in_proj = nn.Conv1d(dim, (self.order + 1) * dim, 1)             # 1x1 -> (order+1) groups
        self.short = nn.Conv1d((self.order + 1) * dim, (self.order + 1) * dim, short_k,
                               padding=self.short_pad, groups=(self.order + 1) * dim)
        self.chomp = Chomp1d(self.short_pad)                                 # keep short conv causal
        self.filters = nn.ModuleList([HyenaFilter(dim, seq_len) for _ in range(self.order)])
        self.out_proj = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def _longconv(self, v, h):
        """Strictly-causal depthwise long convolution of v (n, dim, L) with filter h (dim, L) via
        direct F.conv1d: flip h to cross-correlation order, left-pad L-1, chomp the future tail."""
        weight = h.flip(-1).unsqueeze(1)                                     # (dim, 1, L)
        y = F.conv1d(v, weight, padding=self.seq_len - 1, groups=self.dim)   # (n, dim, 2L-1)
        return y[:, :, :self.seq_len]                                        # chomp future -> causal

    def forward(self, u):
        z = self.chomp(self.short(self.in_proj(u)))                          # (n, (order+1)*dim, L)
        parts = z.chunk(self.order + 1, dim=1)                               # x1, x2, ..., v
        gates, v = parts[:-1], parts[-1]
        for o in range(self.order):
            v = gates[o] * self._longconv(v, self.filters[o]())              # gated causal long conv
        return self.drop(self.out_proj(v))                                   # (n, dim, L)


class HyenaBlock(nn.Module):
    """One Hyena block over (n, dim, L):
      x -> [ HyenaOperator(x) -> BatchNorm1d ] + residual
        -> [ pointwise ConvFFN: 1x1 Conv dim->ffn_mult*dim -> GELU -> 1x1 Conv ->dim -> Dropout ] + x.
    The operator mixes across time (causal), the 1x1 ConvFFN mixes channels per-token (trivially
    causal). Same norm style (BatchNorm1d after the token mixer) as the E88 ModernTCN sibling."""

    def __init__(self, dim, seq_len, ffn_mult=HY_FFN_MULT, dropout=HY_DROPOUT,
                 order=HY_ORDER, short_k=HY_SHORT_K):
        super().__init__()
        self.op = HyenaOperator(dim, seq_len, order, short_k, dropout)
        self.bn = nn.BatchNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, ffn_mult * dim, 1)    # pointwise ConvFFN up-projection
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(ffn_mult * dim, dim, 1)    # pointwise ConvFFN down-projection
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.bn(self.op(x))                         # causal Hyena token mixing
        x = x + y                                       # residual around the operator
        z = self.drop(self.pw2(self.act(self.pw1(x))))  # pointwise ConvFFN channel mixing
        return x + z                                    # residual around ConvFFN


class _SeqHyenaNet(nn.Module):
    """Hyena encoder over channels=[normalized value z, mask], length L_SEQ, then a per-step head on
    the LAST timestep. Input (n, 2, L_SEQ) right-aligned -> logit (n,). Causal 1x1 input projection ->
    a stack of HyenaBlocks (causal implicit long conv + ConvFFN) -> head reads index -1 (= current step
    t). NO global/adaptive pooling: changepoint-location information is preserved (the E82 fix)."""

    def __init__(self, in_ch=2, dim=HY_DIM, blocks=HY_BLOCKS, seq_len=L_SEQ,
                 order=HY_ORDER, short_k=HY_SHORT_K, ffn_mult=HY_FFN_MULT, dropout=HY_DROPOUT):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, dim, kernel_size=1)     # pointwise input projection (causal)
        self.blocks = nn.Sequential(*[
            HyenaBlock(dim, seq_len, ffn_mult, dropout, order, short_k) for _ in range(blocks)])
        self.head = nn.Linear(dim, 1)

    def encode(self, x):
        h = self.stem(x)                                # (n, D, L_SEQ)
        return self.blocks(h)                           # (n, D, L_SEQ) length preserved (causal)

    def forward(self, x):
        h = self.encode(x)                              # (n, D, L_SEQ)
        last = h[:, :, -1]                              # (n, D) -- most-recent step = current step t
        return self.head(last).squeeze(-1)              # (n,) -- 1 logit per step, no pooling


class SeqHyena:
    """sklearn-like wrapper so prep_scores stays parallel to E83/E87/E88. .fit(seq,msk,y) /
    .predict_proba(seq,msk) -> (n,2) with column 1 = P(break). Deterministic on CPU."""

    def __init__(self, seed, epochs=SEQHYENA_EPOCHS, lr=1e-3, weight_decay=1e-4,
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
        self.net = _SeqHyenaNet()
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
    """Fit ranker+rf (uniform weights == champion) + the learned Hyena member, and return per-fold
    (active, groups, prk, prf, pmdn) where prk/prf/pmdn are the fixed-CDF percentiles at full length."""
    Xtr, ytr = sd["Xtr"], sd["ytr"]

    rf = RandomForestClassifier(n_estimators=RF_TREES, n_jobs=N_JOBS, random_state=seed, **RF_KW)
    rf.fit(Xtr, ytr)                                  # NO sample_weight -> identical to shipped champion
    ref_rf = np.sort(_proba(rf, Xtr))

    rows, lab, grp, _swo = sort_into_groups_w(sd["Rr"], sd["Rl"], sd["Rt"], None)
    rk = fit_ranker_w("rank_xendcg", rows, lab, grp, None)
    ref_rk = np.sort(_proba(rk, sd["Rr"]))

    hy = SeqHyena(seed, epochs=SEQHYENA_EPOCHS, n_jobs=N_JOBS)
    hy.fit(sd["Str_seq"], sd["Str_msk"], ytr)
    ref_mdn = np.sort(hy.predict_proba(sd["Str_seq"], sd["Str_msk"])[:, 1])

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
            smdn[active] = hy.predict_proba(c["seq"], c["msk"])[:, 1]
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
    """Log the Hyena wiring + receptive field so the covers-L_SEQ constraint is auditable. The implicit
    long conv filter spans the FULL window (length L_SEQ) in a single operator."""
    rf = L_SEQ                                                    # implicit long conv = full window
    log(f"[hyena-arch] dim={HY_DIM} blocks={HY_BLOCKS} order={HY_ORDER} short_k={HY_SHORT_K} "
        f"filter_len={L_SEQ} filter=implicit-posMLP(hidden={HY_FILTER_MLP},freqs={HY_POS_FREQS})"
        f"+expdecay ffn_mult={HY_FFN_MULT} dropout={HY_DROPOUT} "
        f"longconv=direct(F.conv1d,leftpad{L_SEQ-1}+chomp) RF={rf}(>=L_SEQ {L_SEQ}: {rf >= L_SEQ}) "
        f"head=last-token(no pooling)")
    return rf


def _causal_check():
    """Empirical strict-causality test: perturb a LATER input position and assert every EARLIER output
    position's encoder activation is EXACTLY unchanged (an earlier position must not see a later
    input). Decisive test for the implicit long conv -- must print 0.00e+00 for earlier positions."""
    torch.manual_seed(0)
    net = _SeqHyenaNet()
    net.eval()                                        # eval BN (running stats) -> per-sample deterministic
    x = torch.randn(4, 2, L_SEQ)
    ppos = 64                                         # perturb input position 64 (mid-window)
    with torch.no_grad():
        h0 = net.encode(x)                            # (4, D, L_SEQ) time-space feature map
        x2 = x.clone()
        x2[:, :, ppos] += 7.0                         # inject a large "future" perturbation
        h1 = net.encode(x2)
    early = float((h1[:, :, :ppos] - h0[:, :, :ppos]).abs().max())   # must be EXACTLY 0 (causal)
    late = float((h1[:, :, ppos:] - h0[:, :, ppos:]).abs().max())    # must be > 0 (reacts)
    lastd = float((h1[:, :, -1] - h0[:, :, -1]).abs().max())         # last step DOES react (alive)
    log(f"[causal-check] perturb input@{ppos}: earlier(<{ppos}) maxdiff={early:.2e} "
        f"later(>={ppos}) maxdiff={late:.2e} last-token maxdiff={lastd:.2e} (earlier must be 0)")
    assert early == 0.0, f"CAUSALITY FAILED: an earlier position saw a later input (maxdiff={early:.2e})"
    assert late > 0.0, f"DEAD MODEL: perturbing input@{ppos} changed no later output (maxdiff={late:.2e})"
    return early


def main():
    open(LOG, "w").close()
    log(f"=== HYENA-OPERATOR 3rd-member probe on ranker+rf base79  "
        f"{datetime.now():%F %T}  smoke={SMOKE} ===")
    log(f"seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
        f"ship_scales={SHIP_SCALES}")
    log(f"NCOL={NCOL} L_SEQ={L_SEQ} hyena_epochs={SEQHYENA_EPOCHS} torch={torch.__version__} "
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
        f.write(f"# Hyena-operator 3rd-member probe on ranker+rf base79  "
                f"{datetime.now():%F %T}\n")
        f.write(f"# seeds={SEARCH_SEEDS} rf_trees={RF_TREES} dense_k={DENSE_K} blend={W_RANK}/{W_RF} "
                f"ship={SHIP_SCALES} L_SEQ={L_SEQ} hyena_epochs={SEQHYENA_EPOCHS}\n")
        f.write(f"# 3rd member = causal Hyena operator (Poli 2023): implicit long conv (positional-MLP\n")
        f.write(f"# filter of full window length, exp-decay) + gating over the raw history-standardized\n")
        f.write(f"# trailing window; NO pooling, last-step head keeps changepoint LOCATION; w_mdn=0 ==\n")
        f.write(f"# the champion.\n")
        f.write(f"# BELIEVE only if mean_d>=+0.0015 AND all-seeds-positive, then confirm on fresh seeds\n")
        f.write(f"# {{31337,2025,555,99}}. Prior VERY LOW: learned-rep E31/E45/E47 + shape E59/E79/E82 +\n")
        f.write(f"# pooled seq-CNN E83/E84 + dilated causal TCN E87 + ModernTCN E88 all NULL.\n\n")
        f.write(f"{'arm':<12}{'mean_d':>10}{'min_d':>10}{'max_d':>10}{'npos':>7}\n")
        for arm, md, lo, hi, npos, tot in agg:
            tag = "  <== WIN" if (md >= 0.0015 and npos == tot) else ""
            f.write(f"{arm:<12}{md:+10.5f}{lo:+10.5f}{hi:+10.5f}{npos:>4}/{tot}{tag}\n")
    log("wrote " + SUMMARY)


if __name__ == "__main__":
    main()
