"""Neural two-sample discriminator (window-vs-history) vs the champion HGB.

Why
---
Every hand-feature / paradigm / model lever has flat-lined at ~0.575 LB, and the one
NN attempt (E31, 1D-CNN over the raw window) failed. But E31's CNN read the window as a
lone *signal* -- it never received the historical sample as an explicit reference, and had
no permutation-invariant distributional structure. The problem is structurally a
TWO-SAMPLE TEST (is the trailing window drawn from a different distribution than the
pre-online history?), so we give the network BOTH samples and let it learn the test.

Three architectures (screened in this order per user): 
  2. deepsets  -- permutation-invariant learned two-sample test (ignores order).
  1. hybrid    -- deepsets distributional branch + ordered-window CNN (order + distribution).
  3. seqhist   -- E31's CNN over the ordered window, but WITH the hist set as explicit input.

Fair comparison (identical to E31/_seqmodel): same stratified pool + folds as
_ensemble_tune (8000 train / 2000 val, 10 stratified folds x 200), same labels, same TS-AUC,
same warm-up. We reuse precompute_fold/score_fold by feeding them a PAIR-VECTORISER
(window + history packed into one per-step row), so every net is scored on the exact same
online steps as the champion -> paired-t is apples-to-apples.

Kill gate: screen each arch STANDALONE vs champion on SCREEN_FOLDS folds first. If a net
cannot approach ~0.55 alone it is dead (E28/E31 verdict). Promote survivors to full 10-fold
+ champion blend.

Usage
-----
  SMOKE=1 .venv/bin/python -u _nn_twosample.py                  # ~3 min plumbing check
  SCREEN_FOLDS=3 .venv/bin/python -u _nn_twosample.py           # kill-gate screen (default 3)
  SCREEN_FOLDS=10 .venv/bin/python -u _nn_twosample.py          # full 10-fold
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _overnight import DATA, MIN_N, STEPS_PER_SERIES, build_training, paired  # noqa: F401
from _ensemble_tune import (
    FOLD_SIZE, N_FOLDS, N_TRAIN, SMOKE, WINDOW,
    blend_aucs, champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, report, score_fold, select_picks,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(12, os.cpu_count() or 4))

SEED = int(os.environ.get("SEED", "42"))
CLIP = 8.0
EPOCHS = 2 if SMOKE else 20
BATCH = 512
LR = 1e-3
WD = float(os.environ.get("WD", "1e-3"))     # weight decay (was 1e-4; overfit control)
DROP = float(os.environ.get("DROP", "0.4"))   # dropout (was 0.2)
SET_DIM = 32          # per-point set-encoder embedding width
SEQ_CH = int(os.environ.get("SEQ_CH", "24"))  # conv channels (was 32; capacity control)

WIN_L = WINDOW        # 250
HIST_L = HIST_CAP     # 200
PAIR_DIM = 2 * WIN_L + 2 * HIST_L   # [win_val|win_mask|hist_val|hist_mask]

# folds used for the standalone kill-gate screen (champion computed on the same subset)
SCREEN_FOLDS = 2 if SMOKE else int(os.environ.get("SCREEN_FOLDS", "3"))
STEPS = int(os.environ.get("STEPS", str(STEPS_PER_SERIES)))  # NN training windows per series


def select_picks_n(X, ids, n):
    """Sample n online positions per series for NN training (more windows = more data)."""
    rng = np.random.default_rng(SEED + 1)
    picks = {}
    for sid in ids:
        _, onl = split_series(X.loc[sid])
        L = len(onl)
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        picks[sid] = rng.choice(cand, size=min(n, len(cand)), replace=False)
    return picks


# --- pair vectoriser (drop-in for a feat_fn) -------------------------------
def pair_vec(w, ctx):
    """Standardised trailing window + standardised history sample -> one flat row.

    Both standardised by the PRE-ONLINE history (z=(x-mu_h)/sd_h, clipped +/-CLIP), so a
    mean shift shows as nonzero window mean and a variance change as altered spread.
    Layout: [win_val(WIN_L) | win_mask(WIN_L) | hist_val(HIST_L) | hist_mask(HIST_L)],
    values right-aligned (most-recent at the right edge) with a validity mask.
    """
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / ctx["sd"]
    np.clip(z, -CLIP, CLIP, out=z)
    L = len(z)
    if L > WIN_L:
        z = z[-WIN_L:]
        L = WIN_L
    h = (np.asarray(ctx["sample_ordered"], dtype=np.float64) - ctx["mu"]) / ctx["sd"]
    np.clip(h, -CLIP, CLIP, out=h)
    Lh = len(h)
    if Lh > HIST_L:
        h = h[-HIST_L:]
        Lh = HIST_L
    out = np.zeros(PAIR_DIM, dtype=np.float64)
    out[WIN_L - L:WIN_L] = z
    out[2 * WIN_L - L:2 * WIN_L] = 1.0
    b = 2 * WIN_L
    out[b + HIST_L - Lh:b + HIST_L] = h
    out[b + 2 * HIST_L - Lh:b + 2 * HIST_L] = 1.0
    return out


def build_training_pairs(X, yi, picks, window):
    """Mirror _overnight.build_training exactly, but emit pair vectors not features."""
    rows, labels = [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(pair_vec(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _split_pair(x):
    """(B, PAIR_DIM) flat row -> win (B,2,WIN_L), hist (B,2,HIST_L)."""
    win = torch.stack([x[:, 0:WIN_L], x[:, WIN_L:2 * WIN_L]], dim=1)
    b = 2 * WIN_L
    hist = torch.stack([x[:, b:b + HIST_L], x[:, b + HIST_L:b + 2 * HIST_L]], dim=1)
    return win, hist


# --- building blocks -------------------------------------------------------
class SetEncoder(nn.Module):
    """Permutation-invariant set encoder: pointwise MLP phi then masked mean/max/std pool.

    Input: xset (B, 2, L) with channel 0 = standardised values, channel 1 = validity mask.
    Output: (B, 3*SET_DIM).
    """

    def __init__(self, d=SET_DIM):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Conv1d(1, d, 1), nn.ReLU(),
            nn.Conv1d(d, d, 1), nn.ReLU(),
        )

    def forward(self, xset):
        val = xset[:, 0:1, :]
        m = xset[:, 1:2, :]
        h = self.phi(val)                     # (B, d, L)
        den = m.sum(dim=2).clamp(min=1.0)     # (B, 1)
        mean = (h * m).sum(dim=2) / den
        var = (h * h * m).sum(dim=2) / den - mean * mean
        std = torch.sqrt(var.clamp(min=0.0) + 1e-6)
        mx = h.masked_fill(m == 0, -1e9).max(dim=2).values
        return torch.cat([mean, mx, std], dim=1)   # (B, 3d)


class SeqEncoder(nn.Module):
    """1D-CNN over the ordered (2, WIN_L) window -> (B, 2*SEQ_CH)."""

    def __init__(self, ch=SEQ_CH):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(DROP),
            nn.Conv1d(ch, 2 * ch, 5, padding=2), nn.BatchNorm1d(2 * ch), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )

    def forward(self, win):
        return self.body(win)


def _head(in_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(DROP),
        nn.Linear(64, 1),
    )


# --- architectures (2, 1, 3) ----------------------------------------------
class DeepSetsTS(nn.Module):
    """Arch 2: permutation-invariant learned two-sample test (order-agnostic)."""

    def __init__(self):
        super().__init__()
        self.enc = SetEncoder()
        self.head = _head(4 * 3 * SET_DIM)

    def forward(self, win, hist):
        ew = self.enc(win)
        eh = self.enc(hist)
        feat = torch.cat([ew, eh, ew - eh, (ew - eh).abs()], dim=1)
        return self.head(feat)


class HybridTS(nn.Module):
    """Arch 1: deepsets distributional branch + ordered-window CNN branch."""

    def __init__(self):
        super().__init__()
        self.enc = SetEncoder()
        self.seq = SeqEncoder()
        self.head = _head(4 * 3 * SET_DIM + 2 * SEQ_CH)

    def forward(self, win, hist):
        ew = self.enc(win)
        eh = self.enc(hist)
        dist = torch.cat([ew, eh, ew - eh, (ew - eh).abs()], dim=1)
        seq = self.seq(win)
        return self.head(torch.cat([dist, seq], dim=1))


class SeqHistTS(nn.Module):
    """Arch 3: E31's ordered-window CNN, but WITH the history set as explicit input."""

    def __init__(self):
        super().__init__()
        self.seq = SeqEncoder()
        self.hist_enc = SetEncoder()
        self.head = _head(2 * SEQ_CH + 3 * SET_DIM)

    def forward(self, win, hist):
        return self.head(torch.cat([self.seq(win), self.hist_enc(hist)], dim=1))

    def features(self, win, hist):
        """Penultimate 64-dim embedding (before the final logit layer), for feature fusion."""
        z = torch.cat([self.seq(win), self.hist_enc(hist)], dim=1)
        return self.head[:-1](z)


class PMAPool(nn.Module):
    """Learned-query attention pooling over a masked set of token embeddings."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, heads, dropout=DROP, batch_first=True)

    def forward(self, tokens, key_padding_mask):
        q = self.q.expand(tokens.size(0), -1, -1)
        out, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        return out.squeeze(1)


class SetTransformerTS(nn.Module):
    """Attention two-sample test: self-attend each set, learned-query pool, then compare."""

    def __init__(self, d=48, heads=4):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d,
                                              dropout=DROP, batch_first=True)
        self.pool = PMAPool(d, heads)
        self.head = _head(4 * d)

    def _emb(self, xset):
        kpm = (xset[:, 1, :] == 0)
        h = self.embed(xset[:, 0, :].unsqueeze(-1))
        h = self.enc(h, src_key_padding_mask=kpm)
        return self.pool(h, kpm)

    def forward(self, win, hist):
        ew, eh = self._emb(win), self._emb(hist)
        return self.head(torch.cat([ew, eh, ew - eh, (ew - eh).abs()], dim=1))


class GRUTS(nn.Module):
    """BiGRU over the ordered window (temporal dependence) + hist set encoder."""

    def __init__(self, h=48):
        super().__init__()
        self.gru = nn.GRU(2, h, batch_first=True, bidirectional=True)
        self.hist_enc = SetEncoder()
        self.head = _head(2 * h + 3 * SET_DIM)

    def forward(self, win, hist):
        out, _ = self.gru(win.transpose(1, 2))
        m = win[:, 1, :].unsqueeze(-1)
        pooled = out.masked_fill(m == 0, -1e9).max(dim=1).values
        return self.head(torch.cat([pooled, self.hist_enc(hist)], dim=1))


class SeqHistBigTS(nn.Module):
    """Wider DILATED CNN over the ordered window (multiscale receptive field) + hist set."""

    def __init__(self, ch=48):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, ch, 5, padding=2, dilation=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Dropout(DROP),
            nn.Conv1d(ch, 2 * ch, 5, padding=8, dilation=4), nn.BatchNorm1d(2 * ch), nn.ReLU(),
            nn.Conv1d(2 * ch, 2 * ch, 5, padding=16, dilation=8), nn.BatchNorm1d(2 * ch),
            nn.ReLU(), nn.Dropout(DROP),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
        )
        self.hist_enc = SetEncoder()
        self.head = _head(2 * ch + 3 * SET_DIM)

    def forward(self, win, hist):
        return self.head(torch.cat([self.body(win), self.hist_enc(hist)], dim=1))


ALL_ARCHS = {
    "deepsets": DeepSetsTS, "hybrid": HybridTS, "seqhist": SeqHistTS,
    "seqhistbig": SeqHistBigTS, "settrans": SetTransformerTS, "gru": GRUTS,
}
_sel = os.environ.get("NN_ARCHS", "seqhist,seqhistbig,settrans,gru")
ARCHS = [(n, ALL_ARCHS[n]) for n in _sel.split(",") if n]


# --- train / score ---------------------------------------------------------
class PairProba:
    """Wrap a trained net so score_fold's predict_proba(M)[:,1] works on pair caches."""

    def __init__(self, net, batch=8192):
        self.net = net
        self.batch = batch

    def predict_proba(self, M):
        self.net.eval()
        N = len(M)
        probs = np.empty(N, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, N, self.batch):
                chunk = torch.from_numpy(np.asarray(M[i:i + self.batch], dtype=np.float32))
                win, hist = _split_pair(chunk)
                logit = self.net(win.to(DEVICE), hist.to(DEVICE)).squeeze(1)
                probs[i:i + len(chunk)] = torch.sigmoid(logit).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])

    def predict_embed(self, M):
        """Penultimate 64-dim embedding per row (requires net.features), for embedding fusion."""
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(M), self.batch):
                chunk = torch.from_numpy(np.asarray(M[i:i + self.batch], dtype=np.float32))
                win, hist = _split_pair(chunk)
                out.append(self.net.features(win.to(DEVICE), hist.to(DEVICE)).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 64), dtype=np.float64)


def net_fold_aucs(net, caches, y_val):
    proba = PairProba(net)
    series = [score_fold(proba, c) for c in caches]
    aucs = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in series])
    return aucs, series


def load_labels(ids):
    """Per-(series,step) labels for an arbitrary id set (for the early-stop holdout)."""
    y = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))
    return y[y.index.get_level_values("id").isin(set(ids.tolist()))]


def train_track(model, Xtr, ytr, es_cache, es_y, eval_caches, y_val, epochs, seed):
    """Train, selecting the best epoch by an INDEPENDENT early-stop holdout (es_cache/es_y),
    never the eval folds. Then score the best-epoch net on the eval folds.
    Returns (eval_aucs, eval_series, best_epoch, best_es_auc)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model.to(DEVICE)
    dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=BATCH, shuffle=True, generator=g,
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    best = (-1.0, None, 0)   # (es_auc, state_dict, epoch)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            win, hist = _split_pair(xb)
            opt.zero_grad()
            loss = lossf(model(win, hist).squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        es_aucs, _ = net_fold_aucs(model, es_cache, es_y)
        es_m = float(es_aucs.mean())
        if es_m > best[0]:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best = (es_m, state, ep + 1)
        print(f"    epoch {ep + 1}/{epochs} loss={tot / len(Xtr):.4f} es={es_m:.4f}", flush=True)
    model.load_state_dict(best[1])
    eval_aucs, eval_series = net_fold_aucs(model, eval_caches, y_val)
    return eval_aucs, eval_series, best[2], best[0]


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== nn two-sample ({mode}) start {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"device={DEVICE} N_TRAIN={N_TRAIN} folds={N_FOLDS}x{FOLD_SIZE} "
          f"screen_folds={SCREEN_FOLDS} WINDOW={WINDOW} epochs={EPOCHS}", flush=True)

    X, yi, y_val, train_ids, folds = load_pool_strat()
    folds = folds[:SCREEN_FOLDS]
    # carve an INDEPENDENT early-stop holdout from the train pool (never the eval folds)
    es_n = max(40, int(0.12 * len(train_ids)))
    es_ids = train_ids[-es_n:]
    fit_ids = train_ids[:-es_n]
    picks_c = select_picks(X, fit_ids)           # champion: canonical 10 steps/series
    picks_n = select_picks_n(X, fit_ids, STEPS)  # NN: more windows (STEPS/series)
    print(f"pool: {len(fit_ids)} fit / {len(es_ids)} early-stop, "
          f"eval on {len(folds)} folds x {FOLD_SIZE} | STEPS(nn)={STEPS}", flush=True)

    # champion baseline (its canonical config) on the fit set
    Xtr_c, ytr = build_training(X, yi, picks_c, champion_feat, WINDOW)
    tf = time.time()
    champ_caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    base_aucs, hgb_series = eval_model(make_hgb(), Xtr_c, ytr, champ_caches, y_val)
    print(f"champion cache+fit in {(time.time() - tf) / 60:.1f} min", flush=True)
    report("HGB champion", base_aucs, None)

    # pair training matrix + pair caches (shared across all architectures)
    Xtr_p, ytr_p = build_training_pairs(X, yi, picks_n, WINDOW)
    print(f"pair train {Xtr_p.shape}, positives {ytr_p.mean():.3f}", flush=True)
    tf = time.time()
    pair_caches = [precompute_fold(pair_vec, WINDOW, X, f) for f in folds]
    es_cache = [precompute_fold(pair_vec, WINDOW, X, es_ids)]
    es_y = load_labels(es_ids)
    print(f"pair val+es cache in {(time.time() - tf) / 60:.1f} min", flush=True)

    print(f"config: WD={WD} DROP={DROP} SEQ_CH={SEQ_CH} epochs={EPOCHS} "
          f"(early-stop on independent holdout)", flush=True)
    results = {}
    for name, ctor in ARCHS:
        print(f"\n-- arch: {name} --", flush=True)
        tf = time.time()
        aucs, series, ep, es_best = train_track(
            ctor(), Xtr_p, ytr_p, es_cache, es_y, pair_caches, y_val, EPOCHS, SEED)
        print(f"  trained+scored in {(time.time() - tf) / 60:.1f} min "
              f"(es-picked epoch {ep}, es_auc {es_best:.4f})", flush=True)
        report(f"{name} (es-picked)", aucs, base_aucs)
        best = (None, -1.0)
        for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            a = blend_aucs(hgb_series, series, w, y_val)
            report(f"  blend w_{name}={w:.1f}", a, base_aucs)
            if a.mean() > best[1]:
                best = (w, a.mean())
        results[name] = (float(aucs.mean()), ep, best)

    print(f"\n=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)
    print(f"champion mean={base_aucs.mean():.4f} (on {len(folds)} folds)", flush=True)
    for name, (alone, ep, (bw, bmean)) in results.items():
        print(f"{name:>9}: es-epoch({ep}) alone={alone:.4f} | best blend w={bw} "
              f"mean={bmean:.4f}", flush=True)


if __name__ == "__main__":
    main()
