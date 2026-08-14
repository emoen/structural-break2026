"""Trajectory sequence model (GRU) over the per-step multiscale feature vectors.

Two levers combined, both untried:
  1. REPRESENTATION: the champion is a stateless snapshot classifier — it scores each online
     step independently from its 45-dim window-vs-history feature vector. A break makes that
     contrast DRIFT and PERSIST over subsequent steps. A causal GRU consumes the SEQUENCE of
     those 45-dim vectors and can integrate that temporal persistence (a single elevated step
     may be noise; a sustained elevation is a break).
  2. MORE DATA: the champion trains on 10 sampled snapshots/series. The GRU trains on the FULL
     per-step online trajectory (median 502 steps/series) -> ~49x more supervision. This is the
     "we massively under-use the data" lever, and it only pays off for a data-hungry sequence
     model (E7 showed dense sampling HURTS the tabular HGB via correlated rows).

Fairness: the GRU uses the IDENTICAL 45 champion features per step (no new feature info), so any
gain is attributable purely to temporal integration + dense supervision, not features. Champion
HGB and the GRU are scored on the EXACT same val steps -> paired-t on the same stratified folds.

Runs on CPU (torch 2.13.0+cpu) or CUDA if present (auto). This is the one lever that justifies the
GPU box. Longer shot than the two CPU levers; run it after them.

Usage:
  SMOKE=1 .venv/bin/python -u _seqtraj.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _seqtraj.py > _seqtraj.log 2>&1 & disown
  knobs: SEQ_N_TRAIN (train series, default 3000), EPOCHS (12), HIDDEN (64), POOL_SEEDS
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import torch
import torch.nn as nn

import _ensemble_tune as ET
from _ensemble_tune import champion_feat, load_pool_strat, make_hgb, WINDOW
from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _overnight import MIN_N, N_JOBS, paired

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    POOL_SEEDS = [42]
    SEQ_N_TRAIN = 300
    EPOCHS = 3
else:
    POOL_SEEDS = [int(s) for s in os.environ.get("POOL_SEEDS", "42,7").split(",")]
    SEQ_N_TRAIN = int(os.environ.get("SEQ_N_TRAIN", "3000"))
    EPOCHS = int(os.environ.get("EPOCHS", "12"))

HIDDEN = int(os.environ.get("HIDDEN", "64"))
LAYERS = int(os.environ.get("LAYERS", "1"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
FEAT_DIM = 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- per-series dense feature sequence (all scored steps) ------------------
def build_seq_one(sid, x_h, x_on, onl_index, tau):
    ctx = hist_context(x_h, HIST_CAP)
    L = len(x_on)
    feats, labels, active = [], [], []
    for idx in range(L):
        if idx < MIN_N - 1:      # warm-up: unscored (matches champion & ts_auc_from cumcount)
            continue
        lo = max(0, idx - WINDOW + 1)
        feats.append(champion_feat(x_on[lo:idx + 1], ctx))
        labels.append(1.0 if (tau >= 0 and idx >= tau) else 0.0)
        active.append(idx)
    feat = np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, FEAT_DIM), np.float32)
    return {
        "sid": sid,
        "feat": feat,                                   # (T,45)
        "lab": np.asarray(labels, dtype=np.float32),    # (T,)
        "active": np.asarray(active, dtype=int),        # online idx of each seq position
        "onl_index": np.asarray(onl_index),             # full online time index (len L)
        "L": L,
    }


def build_seqs(X, yi, ids):
    jobs = []
    for sid in ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index,
                     int(yi.loc[sid, "tau_index"])))
    with threadpool_limits(limits=1):
        seqs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(build_seq_one)(sid, x_h, x_on, oi, tau)
            for (sid, x_h, x_on, oi, tau) in jobs
        )
    return [s for s in seqs if s["feat"].shape[0] > 0]


# --- model -----------------------------------------------------------------
class GRUScorer(nn.Module):
    def __init__(self, d_in=FEAT_DIM, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                 # x (B,T,45)
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)   # (B,T) logits


def standardize_fit(seqs):
    allf = np.vstack([s["feat"] for s in seqs])
    mu = allf.mean(0)
    sd = allf.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def _pad_batch(batch, mu, sd):
    lengths = [s["feat"].shape[0] for s in batch]
    Tmax = max(lengths)
    B = len(batch)
    X = np.zeros((B, Tmax, FEAT_DIM), np.float32)
    Y = np.zeros((B, Tmax), np.float32)
    M = np.zeros((B, Tmax), np.float32)
    for i, s in enumerate(batch):
        T = lengths[i]
        X[i, :T] = (np.clip(s["feat"], -1e6, 1e6) - mu) / sd
        Y[i, :T] = s["lab"]
        M[i, :T] = 1.0
    return (torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M))


def train_gru(seqs, mu, sd, seed):
    torch.manual_seed(seed)
    model = GRUScorer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    order = np.arange(len(seqs))
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(EPOCHS):
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BATCH):
            idxs = order[b0:b0 + BATCH]
            batch = [seqs[i] for i in idxs]
            X, Y, M = _pad_batch(batch, mu, sd)
            X, Y, M = X.to(DEVICE), Y.to(DEVICE), M.to(DEVICE)
            opt.zero_grad()
            logits = model(X)
            loss = (lossf(logits, Y) * M).sum() / M.sum()
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        print(f"    epoch {ep+1}/{EPOCHS} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def gru_scores(model, seqs, mu, sd):
    """Return one pd.Series of per-step probabilities over the full val (id,time) index."""
    parts = []
    order = sorted(range(len(seqs)), key=lambda i: seqs[i]["feat"].shape[0])
    for b0 in range(0, len(order), BATCH):
        idxs = order[b0:b0 + BATCH]
        batch = [seqs[i] for i in idxs]
        X, _, M = _pad_batch(batch, mu, sd)
        logits = model(X.to(DEVICE))
        prob = torch.sigmoid(logits).cpu().numpy()
        for j, s in enumerate(batch):
            T = s["feat"].shape[0]
            full = np.zeros(s["L"], dtype=np.float64)
            full[s["active"]] = prob[j, :T]
            midx = pd.MultiIndex.from_product([[s["sid"]], s["onl_index"]], names=["id", "time"])
            parts.append(pd.Series(full, index=midx))
    return pd.concat(parts)


def champion_scores(hgb, seqs):
    """Score champion HGB on the SAME val sequence features -> identical steps, fair paired-t."""
    parts = []
    for s in seqs:
        p = hgb.predict_proba(s["feat"])[:, 1]
        full = np.zeros(s["L"], dtype=np.float64)
        full[s["active"]] = p
        midx = pd.MultiIndex.from_product([[s["sid"]], s["onl_index"]], names=["id", "time"])
        parts.append(pd.Series(full, index=midx))
    return pd.concat(parts)


def fold_auc(series, y_val):
    return ts_auc_from(series.to_frame("prediction"), y_val)


def main():
    print(f"=== trajectory GRU {datetime.now():%Y-%m-%d %H:%M:%S} (device={DEVICE}) ===", flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} SEQ_N_TRAIN={SEQ_N_TRAIN} EPOCHS={EPOCHS} HIDDEN={HIDDEN} "
          f"| champion trains on 10-step snapshots, GRU on dense trajectories", flush=True)

    from _ensemble_tune import select_picks
    from _overnight import build_training as bt_champ

    g_deltas, g_t = [], []
    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()

        # champion HGB trained on its native 10-step snapshots
        picks = select_picks(X, train_ids)
        Xc, yc = bt_champ(X, yi, picks, champion_feat, WINDOW)
        with threadpool_limits(limits=1):
            hgb = make_hgb().fit(Xc, yc)

        # dense sequences: train subset (more-data) + all val folds
        tr_ids = train_ids[:SEQ_N_TRAIN]
        train_seqs = build_seqs(X, yi, tr_ids)
        fold_seqs = [build_seqs(X, yi, f) for f in folds]
        del X
        mu, sd = standardize_fit(train_seqs)
        print(f"[seed {seed}] train seqs={len(train_seqs)} "
              f"(steps={sum(s['feat'].shape[0] for s in train_seqs):,}), "
              f"prep {time.time()-t0:.0f}s", flush=True)

        model = train_gru(train_seqs, mu, sd, seed)

        base = np.asarray([fold_auc(champion_scores(hgb, fs), y_val) for fs in fold_seqs])
        gru = np.asarray([fold_auc(gru_scores(model, fs, mu, sd), y_val) for fs in fold_seqs])
        md, sdv, t, _ = paired(gru, base)
        g_deltas.append(md); g_t.append(t)
        verdict = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
            "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
        print(f"[seed {seed}] champion={base.mean():.4f}  GRU={gru.mean():.4f}  "
              f"delta={md:+.4f} t={t:+.2f} -> {verdict}  ({time.time()-t0:.0f}s)\n", flush=True)
        del train_seqs, fold_seqs, model, hgb

    d = np.asarray(g_deltas); tt = np.asarray(g_t)
    print("=== SUMMARY ===", flush=True)
    print(f"GRU vs champion: mean_d={d.mean():+.4f} min_d={d.min():+.4f} mean_t={tt.mean():+.2f} "
          f"pos_all={'YES' if (d>0).all() else 'no'}", flush=True)
    print("If positive on all seeds with mean_d>=0.0015 (ideally t>=2.5) the trajectory+more-data "
          "lever is real -> then design a streaming GRU infer() (carry hidden state across steps).", flush=True)


if __name__ == "__main__":
    main()
