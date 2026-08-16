"""Stronger trajectory GRU as a 3rd CDF member (E44 follow-up).

E44 showed the GRU is the ONE member with a real decorrelation signal (seed 42 mean_d +0.0033,
t +2.11, positive at every weight) but it FAILED the all-seeds gate because a single GRU fit is
high-variance (seed 7 went negative). This run attacks that variance and adds capacity:
  1. SEED-ENSEMBLE: train N_GRU GRUs per seed with different inits, AVERAGE their per-step
     probabilities into one member before the fixed-CDF percentile. Averaging independent fits is
     the direct fix for the seed-7 variance blow-up.
  2. CAPACITY: wider hidden (128 vs 64) + output dropout + small weight-decay for regularization,
     and more epochs (18 vs 12).
Everything else is IDENTICAL to _gru_ens3.py (same tuned-gb + rank_xendcg plumbing, same dense
fold index, same fixed-CDF blend, same weight sets, same hardened gate, paired-t vs the SHIPPED
2-member champion). Only the GRU member changes.

Gate: positive vs the 2-member champion on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5
  -> fresh-seed confirm on [31337,2025,555,99], then build the 3-member notebook.

Usage (ONE 15-job harness at a time; also holds torch GRUs -> keep it the only heavy job):
  SMOKE=1 .venv/bin/python -u _gru_strong.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _gru_strong.py > _gru_strong.log 2>&1 & disown
  SEEDS=31337,2025,555,99 nohup ... (fresh-seed confirm)
  knobs: N_GRU (3), HID (128), EPOCHS (18), DROPOUT (0.2), SEQ_N_TRAIN (via _seqtraj, 3000)
"""
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from threadpoolctl import threadpool_limits

import _ensemble_tune as ET
from _ensemble_tune import (
    champion_feat, load_pool_strat, make_hgb,
    precompute_fold, score_fold, select_picks, WINDOW,
)
from _exp_fast import ts_auc_from
from _lambdarank import build_grouped, sort_into_groups, fit_ranker
from _overnight import build_training, paired

import _seqtraj
from _seqtraj import build_seqs, standardize_fit, gru_scores, _pad_batch, BATCH, DEVICE, FEAT_DIM

SMOKE = os.environ.get("SMOKE") == "1"
_SEEDS_ENV = os.environ.get("SEEDS", "")
if _SEEDS_ENV:
    POOL_SEEDS = [int(s) for s in _SEEDS_ENV.split(",") if s.strip()]
    DENSE_K = 60
elif SMOKE:
    POOL_SEEDS = [42]
    DENSE_K = 30
else:
    POOL_SEEDS = [int(s) for s in os.environ.get("POOL_SEEDS", "42,7").split(",")]
    DENSE_K = 60

N_GRU = int(os.environ.get("N_GRU", "1" if SMOKE else "3"))
HID = int(os.environ.get("HID", "32" if SMOKE else "128"))
LAYERS = int(os.environ.get("LAYERS", "1"))
DROPOUT = float(os.environ.get("DROPOUT", "0.2"))
EPOCHS = int(os.environ.get("EPOCHS", "3" if SMOKE else "18"))
LR = float(os.environ.get("LR", "1e-3"))
WD = float(os.environ.get("WD", "1e-5"))

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

BASE_W = (0.6, 0.4, 0.0)
WEIGHT_SETS = [
    BASE_W,
    (0.50, 0.33, 0.17),
    (0.48, 0.32, 0.20),
    (0.45, 0.30, 0.25),
    (0.42, 0.28, 0.30),
    (0.36, 0.34, 0.30),
    (0.34, 0.33, 0.33),
]


class GRUScorerCfg(nn.Module):
    def __init__(self, d_in=FEAT_DIM, hidden=HID, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.drop(h)).squeeze(-1)


def train_one(train_seqs, mu, sd, seed):
    torch.manual_seed(seed)
    model = GRUScorerCfg().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    order = np.arange(len(train_seqs))
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(EPOCHS):
        rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BATCH):
            batch = [train_seqs[i] for i in order[b0:b0 + BATCH]]
            X, Y, M = _pad_batch(batch, mu, sd)
            X, Y, M = X.to(DEVICE), Y.to(DEVICE), M.to(DEVICE)
            opt.zero_grad()
            loss = (lossf(model(X), Y) * M).sum() / M.sum()
            loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
    model.eval()
    return model


@torch.no_grad()
def active_probs(model, seqs, mu, sd):
    """Concatenated active-step probabilities in length-sorted seq order (deterministic across
    models -> elementwise-averageable for the ensemble reference)."""
    out = []
    order = sorted(range(len(seqs)), key=lambda i: seqs[i]["feat"].shape[0])
    for b0 in range(0, len(order), BATCH):
        batch = [seqs[i] for i in order[b0:b0 + BATCH]]
        Xb, _, _ = _pad_batch(batch, mu, sd)
        prob = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy()
        for j, s in enumerate(batch):
            out.append(prob[j, :s["feat"].shape[0]])
    return np.concatenate(out) if out else np.zeros(1)


def pct(x, ref):
    return np.searchsorted(ref, x, side="right") / len(ref)


def _key(w):
    return f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"


def main():
    print(f"=== stronger GRU 3rd-member {datetime.now():%Y-%m-%d %H:%M:%S} (device={DEVICE}) ===",
          flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K} SEQ_N_TRAIN={_seqtraj.SEQ_N_TRAIN} "
          f"N_GRU={N_GRU} HID={HID} LAYERS={LAYERS} DROPOUT={DROPOUT} EPOCHS={EPOCHS} WD={WD}  "
          f"base=shipped {_key(BASE_W)}  gate: pos_all & mean_d>=0.0015 & mean_t>=2.5", flush=True)

    cand = [w for w in WEIGHT_SETS if w != BASE_W]
    d_store = {_key(w): [] for w in cand}
    t_store = {_key(w): [] for w in cand}

    for seed in POOL_SEEDS:
        t0 = time.time()
        ET.SEED = seed
        X, yi, y_val, train_ids, folds = load_pool_strat()
        caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]

        picks = select_picks(X, train_ids)
        Xtr_c, ytr_c = build_training(X, yi, picks, champion_feat, WINDOW)
        model_gb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            model_gb.fit(Xtr_c, ytr_c)
        ref_gb = np.sort(model_gb.predict_proba(Xtr_c)[:, 1])
        gb_series = [score_fold(model_gb, c) for c in caches]

        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]

        tr_ids = train_ids[:_seqtraj.SEQ_N_TRAIN]
        train_seqs = build_seqs(X, yi, tr_ids)
        fold_seqs = [build_seqs(X, yi, f) for f in folds]
        del X
        mu, sd = standardize_fit(train_seqs)
        print(f"[seed {seed}] gb/rank ready + gru train_seqs={len(train_seqs)} "
              f"(steps={sum(s['feat'].shape[0] for s in train_seqs):,}), prep {time.time()-t0:.0f}s",
              flush=True)

        # seed-ensemble of GRUs: average probabilities before the fixed-CDF percentile
        models = []
        for g in range(N_GRU):
            tg = time.time()
            models.append(train_one(train_seqs, mu, sd, seed * 100 + g))
            print(f"  gru {g+1}/{N_GRU} trained ({time.time()-tg:.0f}s)", flush=True)
        ref_gru = np.sort(np.mean([active_probs(m, train_seqs, mu, sd) for m in models], axis=0))
        gru_series = []
        for fs, c in zip(fold_seqs, caches):
            ens = sum(gru_scores(m, fs, mu, sd).reindex(c["index"]).fillna(0.0)
                      for m in models) / N_GRU
            gru_series.append(ens)

        per = {_key(w): [] for w in WEIGHT_SETS}
        for hs, rs_, gs in zip(gb_series, rank_series, gru_series):
            fh = pct(hs.to_numpy(), ref_gb)
            fr = pct(rs_.to_numpy(), ref_rank)
            fg = pct(gs.to_numpy(), ref_gru)
            for w in WEIGHT_SETS:
                b = pd.Series(w[0] * fh + w[1] * fr + w[2] * fg, index=hs.index)
                per[_key(w)].append(ts_auc_from(b.to_frame("prediction"), y_val))
        base_aucs = np.asarray(per[_key(BASE_W)])

        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        for w in cand:
            a = np.asarray(per[_key(w)])
            md, _s, t, _ = paired(a, base_aucs)
            d_store[_key(w)].append(md); t_store[_key(w)].append(t)
            v = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
                "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
            print(f"  {_key(w):<16} mean={a.mean():.4f} d={md:+.4f} t={t:+.2f} -> {v}", flush=True)
        del caches, gb_series, rank_series, gru_series, Xtr_c, ytr_c, rd, ld, td
        del train_seqs, fold_seqs, models, model_gb, ranker

    print("\n=== SUMMARY (mean over seeds, vs shipped 2-member blend) ===", flush=True)
    print(f"{'w_gb/rank/gru':<16} {'mean_d':>8} {'min_d':>8} {'mean_t':>7}  pos_all  verdict", flush=True)
    for w in cand:
        d = np.asarray(d_store[_key(w)]); t = np.asarray(t_store[_key(w)])
        pos = (d > 0).all()
        confirm = pos and d.mean() >= 0.0015 and t.mean() >= 2.5
        v = "CONFIRM" if confirm else ("promising" if (pos and d.mean() > 0) else "noise")
        print(f"{_key(w):<16} {d.mean():+8.4f} {d.min():+8.4f} {t.mean():+7.2f}   "
              f"{'YES' if pos else ' no'}     {v}", flush=True)
    print("\nA weight set positive on ALL seeds with mean_d>=0.0015 & mean_t>=2.5 BEATS the shipped "
          "2-member champion -> fresh-seed confirm, then build the 3-member notebook.", flush=True)


if __name__ == "__main__":
    main()
