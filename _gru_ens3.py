"""Trajectory GRU as a 3rd member of the SHIPPED CDF-blend ensemble.

The shipped champion (leaderboard 58.78%) = tuned gradient-boosting + LightGBM rank_xendcg,
FIXED train-CDF blend  emit  0.6*F_gb(proba) + 0.4*F_rank(score).

Every member so far (gradient-boosting, ranker, random forest) is a STATELESS SNAPSHOT scorer:
each online step is scored independently from its 45-dim window-vs-history vector. The GRU is the
one genuinely new axis -- it consumes the SEQUENCE of those vectors and integrates the temporal
PERSISTENCE of a break (a single elevated step is noise; a sustained elevation is a break). It is
therefore the one member with a real shot at being DECORRELATED from the tree+ranker, unlike the
random forest (bagging) and scale specialists (feature subspace) that both overlapped the ranker.

This mirrors _rf_ens3.py EXACTLY (same fixed-CDF `pct`, same tuned-gb + rank_xendcg plumbing,
same paired-t vs the SHIPPED 2-member blend) and swaps member 3 (random forest) for the GRU from
_seqtraj.py. The GRU trains on DENSE per-step trajectories (the "we under-use the data" lever);
gb + ranker keep their native 10-step-snapshot / dense-group training. All three are scored on the
IDENTICAL dense fold index (_featmat_one and build_seq_one cover the same warm-up-passing steps),
so the fixed-CDF percentile blend aligns exactly.

Gate: positive vs the 2-member champion on ALL seeds AND mean_d>=0.0015 AND mean_t>=2.5
  -> fresh-seed confirm on [31337,2025,555,99], then build the 3-member notebook.

Usage (ONE 15-job harness at a time; also holds a torch GRU -> keep it the only heavy job):
  SMOKE=1 .venv/bin/python -u _gru_ens3.py
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _gru_ens3.py > _gru_ens3.log 2>&1 & disown
  SEEDS=31337,2025,555,99 nohup ... (fresh-seed confirm)
  knobs (read by _seqtraj): SEQ_N_TRAIN (train series, 3000), EPOCHS (12), HIDDEN (64)
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
from _seqtraj import (
    build_seqs, standardize_fit, train_gru, gru_scores, _pad_batch, BATCH, DEVICE,
)

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

TUNED = dict(
    learning_rate=0.021145795969329348, max_iter=400, max_leaf_nodes=63,
    min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
)

# (w_gb, w_rank, w_gru); first row = shipped 2-member baseline (sanity ~0 vs itself).
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


def pct(x, ref):
    """Fixed-reference percentile: fraction of ref <= x. ref must be pre-sorted."""
    return np.searchsorted(ref, x, side="right") / len(ref)


def _key(w):
    return f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"


@torch.no_grad()
def gru_ref(model, seqs, mu, sd):
    """Sorted GRU probabilities over TRAIN active steps -> fixed-CDF reference (matches how the
    tree members use predict_proba on their training rows). Warm-up steps excluded."""
    vals = []
    order = sorted(range(len(seqs)), key=lambda i: seqs[i]["feat"].shape[0])
    for b0 in range(0, len(order), BATCH):
        batch = [seqs[i] for i in order[b0:b0 + BATCH]]
        Xb, _, _ = _pad_batch(batch, mu, sd)
        prob = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy()
        for j, s in enumerate(batch):
            T = s["feat"].shape[0]
            vals.append(prob[j, :T])
    return np.sort(np.concatenate(vals)) if vals else np.zeros(1)


def main():
    print(f"=== GRU 3rd-member CDF ensemble {datetime.now():%Y-%m-%d %H:%M:%S} (device={DEVICE}) ===",
          flush=True)
    print(f"POOL_SEEDS={POOL_SEEDS} dense_k={DENSE_K} SEQ_N_TRAIN={_seqtraj.SEQ_N_TRAIN} "
          f"EPOCHS={_seqtraj.EPOCHS} HIDDEN={_seqtraj.HIDDEN}  base=shipped 2-member {_key(BASE_W)}  "
          f"gate: pos_all & mean_d>=0.0015 & mean_t>=2.5", flush=True)

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

        # member 1: tuned gradient-boosting
        model_gb = make_hgb(**TUNED, random_state=seed)
        with threadpool_limits(limits=1):
            model_gb.fit(Xtr_c, ytr_c)
        ref_gb = np.sort(model_gb.predict_proba(Xtr_c)[:, 1])
        gb_series = [score_fold(model_gb, c) for c in caches]

        # member 2: rank_xendcg dense ranker
        rng = np.random.default_rng(seed + 202)
        rd, ld, td = build_grouped(X, yi, train_ids, champion_feat, WINDOW, DENSE_K, rng)
        rr, lr, gr = sort_into_groups(rd, ld, td)
        ranker = fit_ranker("rank_xendcg", rr, lr, gr)
        ref_rank = np.sort(ranker.r.predict(rr))
        rank_series = [score_fold(ranker, c) for c in caches]

        # member 3: trajectory GRU (dense per-step supervision)
        tr_ids = train_ids[:_seqtraj.SEQ_N_TRAIN]
        train_seqs = build_seqs(X, yi, tr_ids)
        fold_seqs = [build_seqs(X, yi, f) for f in folds]
        del X
        mu, sd = standardize_fit(train_seqs)
        print(f"[seed {seed}] gb/rank fit + gru train_seqs={len(train_seqs)} "
              f"(steps={sum(s['feat'].shape[0] for s in train_seqs):,}), prep {time.time()-t0:.0f}s",
              flush=True)
        model_gru = train_gru(train_seqs, mu, sd, seed)
        ref_gru = gru_ref(model_gru, train_seqs, mu, sd)
        # align GRU fold scores to the exact cache index (same dense steps, possibly reordered)
        gru_series = [gru_scores(model_gru, fs, mu, sd).reindex(c["index"]).fillna(0.0)
                      for fs, c in zip(fold_seqs, caches)]

        # per-fold fixed-CDF percentiles, then blend under each weight set
        per = {_key(w): [] for w in WEIGHT_SETS}
        for hs, rs_, gs in zip(gb_series, rank_series, gru_series):
            fh = pct(hs.to_numpy(), ref_gb)
            fr = pct(rs_.to_numpy(), ref_rank)
            fg = pct(gs.to_numpy(), ref_gru)
            for w in WEIGHT_SETS:
                b = pd.Series(w[0] * fh + w[1] * fr + w[2] * fg, index=hs.index)
                per[_key(w)].append(ts_auc_from(b.to_frame("prediction"), y_val))
        base_aucs = np.asarray(per[_key(BASE_W)])

        print(f"\n[seed {seed}] 2-member base mean={base_aucs.mean():.4f} "
              f"train={Xtr_c.shape} ({time.time()-t0:.0f}s)", flush=True)
        for w in cand:
            a = np.asarray(per[_key(w)])
            md, _s, t, _ = paired(a, base_aucs)
            d_store[_key(w)].append(md); t_store[_key(w)].append(t)
            v = "SHIP(hard)" if (md > 0 and t >= 2.5) else (
                "marginal" if (md > 0 and t >= 2.0) else ("weak" if (md > 0 and t >= 1.5) else "noise"))
            print(f"  {_key(w):<16} mean={a.mean():.4f} d={md:+.4f} t={t:+.2f} -> {v}", flush=True)
        del caches, gb_series, rank_series, gru_series, Xtr_c, ytr_c, rd, ld, td
        del train_seqs, fold_seqs, model_gru, model_gb, ranker

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
