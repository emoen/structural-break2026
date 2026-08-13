"""Sequence-model (1D-CNN over the raw online window) vs the champion HGB.

Motivation
----------
Every hand-engineered-feature lever (E14-E25), paradigm lever (E26-E29) and the
model/ensemble lever (E30) has flat-lined. The one representation we have NEVER tried is
letting a network read the RAW ordered online window instead of ~45 summary statistics.
A 1D-CNN is translation-invariant: it can localise a level shift / variance change / shape
change anywhere in the trailing window, and pick up temporal structure that per-window
summary stats blur out. This is the genuine "different representation" test.

Fair comparison
---------------
Same stratified pool + folds as _ensemble_tune (8000 train / 2000 val, 10 stratified
folds x 200). Same labels: step t is positive iff (tau>=0 and t>=tau). Same TS-AUC metric,
same warm-up (score 0 until the sliding buffer reaches MIN_N). We reuse _ensemble_tune's
`precompute_fold`/`score_fold` by feeding them a WINDOW-VECTORISER instead of a feature
function, so the CNN is scored on the exact same online steps as the champion -> paired-t
is apples-to-apples.

CNN input: trailing raw window standardised by the pre-online history: z=(x-mu_h)/sd_h,
clipped to +/-8, left-padded to length WINDOW, plus a validity-mask channel -> (2, WINDOW).

Also included (answers the "was HGB really tuned?" question): a RANDOMISED JOINT HGB search
(vs the one-axis-at-a-time E30 grid), reusing the champion feature cache so it is cheap.

Usage
-----
  SMOKE=1 .venv/bin/python -u _seqmodel.py                     # ~3 min plumbing check
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _seqmodel.py \
       > _seqmodel.log 2>&1 & disown                            # real run
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
from _overnight import MIN_N, build_training, paired
from _ensemble_tune import (
    FOLD_SIZE, N_FOLDS, N_TRAIN, SMOKE, WINDOW,
    blend_aucs, champion_feat, eval_model, load_pool_strat, make_hgb,
    precompute_fold, report, score_fold, select_picks,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# CNN training/scoring runs in the MAIN process (feature precompute uses loky separately),
# so give torch real threads despite OMP_NUM_THREADS=1 in the env.
torch.set_num_threads(min(12, os.cpu_count() or 4))
SEED = 42
CLIP = 8.0
EPOCHS = 2 if SMOKE else 14
BATCH = 512
LR = 1e-3
N_HGB_RANDOM = 4 if SMOKE else 24
if os.environ.get("SKIP_HGB") == "1":
    N_HGB_RANDOM = 0   # search already captured; go straight to the CNN


# --- window vectoriser (drop-in for a feat_fn) -----------------------------
def window_vec(w, ctx):
    """Standardised trailing window -> flat length-2*WINDOW vector.

    Layout: [ values(WINDOW) | mask(WINDOW) ], each left-padded (most-recent at the
    right edge). Reshaped to (2, WINDOW) inside the CNN wrapper.
    """
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / ctx["sd"]
    np.clip(z, -CLIP, CLIP, out=z)
    L = len(z)
    if L > WINDOW:
        z = z[-WINDOW:]
        L = WINDOW
    out = np.zeros(2 * WINDOW, dtype=np.float64)
    out[WINDOW - L:WINDOW] = z          # channel 0: standardised values
    out[2 * WINDOW - L:2 * WINDOW] = 1.0  # channel 1: validity mask
    return out


def build_training_windows(X, yi, picks, window):
    """Mirror _overnight.build_training exactly, but emit raw windows not features."""
    rows, labels = [], []
    for sid, ts in picks.items():
        x_h, onl = split_series(X.loc[sid])
        x_on = onl.to_numpy(dtype=np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(yi.loc[sid, "tau_index"])
        for t in ts:
            lo = max(0, t - window + 1)
            rows.append(window_vec(x_on[lo:t + 1], ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


# --- 1D-CNN ----------------------------------------------------------------
class SeqCNN(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2, ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, 2 * ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(2 * ch), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * ch, 2 * ch), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(2 * ch, 1),
        )

    def forward(self, x):
        return self.head(self.body(x))


class CNNProba:
    """Wrap a trained SeqCNN so score_fold's `predict_proba(M)[:,1]` just works.

    M is the (N, 2*WINDOW) cache matrix produced by precompute_fold(window_vec, ...).
    """
    def __init__(self, net, batch=8192):
        self.net = net
        self.batch = batch

    def predict_proba(self, M):
        self.net.eval()
        N = len(M)
        probs = np.empty(N, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, N, self.batch):
                chunk = np.asarray(M[i:i + self.batch], dtype=np.float32)
                x = torch.from_numpy(chunk.reshape(-1, 2, WINDOW))
                logit = self.net(x).squeeze(1)
                probs[i:i + len(chunk)] = torch.sigmoid(logit).numpy()
        return np.column_stack([1.0 - probs, probs])


def train_cnn(Xtr_win, ytr, epochs, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = SeqCNN()
    Xt = torch.from_numpy(Xtr_win.reshape(-1, 2, WINDOW))
    yt = torch.from_numpy(ytr.astype(np.float32))
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=True, generator=g)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(net(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        print(f"  cnn epoch {ep + 1}/{epochs} loss={tot / len(yt):.4f}", flush=True)
    return net


def cnn_fold_aucs(cnn, caches, y_val):
    series = [score_fold(cnn, c) for c in caches]
    aucs = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in series])
    return aucs, series


# --- randomised joint HGB search (vs one-axis E30 grid) --------------------
def sample_hgb(rng):
    lr = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
    return dict(
        learning_rate=round(lr, 4),
        max_iter=int(rng.choice([150, 200, 300, 400])),
        max_leaf_nodes=int(rng.choice([15, 31, 47, 63, 95])),
        min_samples_leaf=int(rng.choice([10, 20, 40, 80])),
        l2_regularization=float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        max_features=float(rng.choice([0.5, 0.7, 1.0])),
    )


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== seqmodel ({mode}) start {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)
    print(f"config: N_TRAIN={N_TRAIN} folds={N_FOLDS}x{FOLD_SIZE} WINDOW={WINDOW} "
          f"epochs={EPOCHS} torch_threads={torch.get_num_threads()}", flush=True)

    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {FOLD_SIZE}", flush=True)

    # champion feature training matrix + fold caches
    Xtr_c, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    print(f"champion train matrix {Xtr_c.shape}, positives {ytr.mean():.3f}", flush=True)
    tf = time.time()
    champ_caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    print(f"champion cache in {(time.time()-tf)/60:.1f} min", flush=True)

    # --- champion baseline ---
    print("\n-- baseline --", flush=True)
    base_aucs, hgb_series = eval_model(make_hgb(), Xtr_c, ytr, champ_caches, y_val)
    report("HGB champion", base_aucs, None)

    # --- randomised joint HGB search (cheap: reuses champion cache) ---
    print(f"\n-- randomised joint HGB search ({N_HGB_RANDOM} configs, vs champion) --", flush=True)
    rng = np.random.default_rng(SEED + 7)
    best_hgb = ("champion", base_aucs.mean(), dict())
    for i in range(N_HGB_RANDOM):
        over = sample_hgb(rng)
        a, _ = eval_model(make_hgb(**over), Xtr_c, ytr, champ_caches, y_val)
        tag = f"hgb#{i:02d}"
        report(f"{tag} {over['learning_rate']}/{over['max_iter']}/{over['max_leaf_nodes']}/"
               f"msl{over['min_samples_leaf']}/l2{over['l2_regularization']}/mf{over['max_features']}",
               a, base_aucs)
        if a.mean() > best_hgb[1]:
            best_hgb = (tag, a.mean(), over)
    print(f"  best random HGB: {best_hgb[0]} mean={best_hgb[1]:.4f} over={best_hgb[2]}", flush=True)

    # --- sequence model (1D-CNN) ---
    print("\n-- sequence model: 1D-CNN over raw standardised window --", flush=True)
    Xtr_w, ytr_w = build_training_windows(X, yi, picks, WINDOW)
    assert np.array_equal(ytr_w, ytr), "window labels must match feature labels"
    print(f"cnn train windows {Xtr_w.shape}, positives {ytr_w.mean():.3f}", flush=True)
    tf = time.time()
    win_caches = [precompute_fold(window_vec, WINDOW, X, f) for f in folds]
    print(f"cnn val cache in {(time.time()-tf)/60:.1f} min", flush=True)

    tf = time.time()
    net = train_cnn(Xtr_w, ytr_w, EPOCHS, SEED)
    print(f"cnn trained in {(time.time()-tf)/60:.1f} min", flush=True)
    cnn = CNNProba(net)
    cnn_aucs, cnn_series = cnn_fold_aucs(cnn, win_caches, y_val)
    report("CNN alone", cnn_aucs, base_aucs)

    # --- CNN + champion HGB probability blend (the real hope: orthogonal signal) ---
    print("\n-- HGB champion + CNN blend (weight on CNN) --", flush=True)
    best_blend = (None, -1.0)
    for w in (0.2, 0.3, 0.4, 0.5, 0.6):
        a = blend_aucs(hgb_series, cnn_series, w, y_val)
        report(f"blend w_cnn={w:.1f}", a, base_aucs)
        if a.mean() > best_blend[1]:
            best_blend = (w, a.mean())

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"champion mean={base_aucs.mean():.4f}", flush=True)
    print(f"best random HGB: {best_hgb[0]} mean={best_hgb[1]:.4f} over={best_hgb[2]}", flush=True)
    print(f"CNN alone mean={cnn_aucs.mean():.4f}", flush=True)
    print(f"best HGB+CNN blend: w_cnn={best_blend[0]} mean={best_blend[1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
