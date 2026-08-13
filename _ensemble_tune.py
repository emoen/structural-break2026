"""Ensemble (HGB + Logistic Regression) + light hyperparameter tuning harness.

Design goals
------------
1. Add a *decorrelated* second model (logistic regression) to the champion HGB and
   test whether a probability-blend lifts TS-AUC (recovering signal a single tree
   model leaves on the table). RandomForest deferred to a later pass.
2. Tune both members with a small grid, scored on OUR metric (TS-AUC), not log-loss.
3. Fix the validation design: smaller held-out set (20% vs 40%) and STRATIFIED folds
   (equal structural-break rate per fold) so folds are comparable.

Efficiency
----------
The expensive part is computing the 45 multiscale features at every online step of every
validation series. We do that ONCE per fold (`precompute_fold`) and cache the feature
matrix; every candidate model then only calls predict_proba on the cache. So baselines,
the blend weight-sweep, and both tuning grids are cheap after a single feature pass.

Champion parity: champion_feat == base15 (base14 + spec_entropy) on nested trailing
sub-windows for scales (60,130,250) => 45 features, WINDOW=250. Same as the E17 champion
and the E25-E29 notebooks.

Usage
-----
  SMOKE=1 .venv/bin/python -u _ensemble_tune.py          # ~2 min plumbing check
  nohup env PYTHONWARNINGS=ignore .venv/bin/python -u _ensemble_tune.py \
       > _ensemble_tune.log 2>&1 & disown                 # real run
"""
import math
import os
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from _exp_fast import HIST_CAP, hist_context, split_series, ts_auc_from
from _feat_lib import _feat
from _overnight import DATA, MIN_N, N_JOBS, SEED, STEPS_PER_SERIES, build_training, paired

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_TRAIN, N_FOLDS, FOLD_SIZE = 300, 2, 40
else:
    N_TRAIN, N_FOLDS, FOLD_SIZE = 8000, 10, 200   # 20% held out, stratified

SCALES = (60, 130, 250)
WINDOW = 250
BASE_KEYS = ("spec_entropy",)
FEAT_DIM = len(SCALES) * 15   # base15 per scale


def _subwin(w, s):
    return w if s >= len(w) else w[-s:]


def champion_feat(w, ctx):
    out = []
    for s in SCALES:
        out.extend(_feat(_subwin(w, s), ctx, keys=BASE_KEYS))
    return out


HGB_BASE = dict(
    max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, min_samples_leaf=20,
    early_stopping=True, random_state=SEED,
)


def make_hgb(**over):
    p = dict(HGB_BASE)
    p.update(over)
    return HistGradientBoostingClassifier(**p)


def make_lr(C=1.0, class_weight=None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight=class_weight, random_state=SEED),
    )


# --- data: stratified pool -------------------------------------------------
def load_pool_strat():
    """Fixed train set + N_FOLDS disjoint val folds, stratified by break label.

    Series label = (tau_index >= 0) i.e. the series contains a structural break.
    Val positives/negatives are round-robin assigned across folds so every fold has
    (near) identical break-rate -> folds are comparable, reducing fold-to-fold noise.
    """
    rng = np.random.default_rng(SEED)
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    ids = yi.index.to_numpy()
    has_break = (yi["tau_index"].to_numpy() >= 0).astype(int)
    pos = ids[has_break == 1].copy()
    neg = ids[has_break == 0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val = N_FOLDS * FOLD_SIZE
    p = len(pos) / len(ids)
    n_val_pos = int(round(n_val * p))
    n_val_neg = n_val - n_val_pos
    val_pos, val_neg = pos[:n_val_pos], neg[:n_val_neg]

    # stratified folds: round-robin so each fold gets ~equal pos and neg
    folds = [[] for _ in range(N_FOLDS)]
    for i, sid in enumerate(val_pos):
        folds[i % N_FOLDS].append(sid)
    for i, sid in enumerate(val_neg):
        folds[i % N_FOLDS].append(sid)
    folds = [np.sort(np.asarray(f)) for f in folds]
    val_ids = np.concatenate(folds)

    train_pool = np.concatenate([pos[n_val_pos:], neg[n_val_neg:]])
    rng.shuffle(train_pool)
    train_ids = np.sort(train_pool[:N_TRAIN])

    keep = set(train_ids.tolist()) | set(val_ids.tolist())
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))
    y_val = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    return X, yi, y_val, train_ids, folds


def select_picks(X, train_ids):
    rng = np.random.default_rng(SEED + 1)
    picks = {}
    for sid in train_ids:
        _, onl = split_series(X.loc[sid])
        L = len(onl)
        if L <= MIN_N:
            continue
        cand = np.arange(MIN_N, L)
        k = min(STEPS_PER_SERIES, len(cand))
        picks[sid] = rng.choice(cand, size=k, replace=False)
    return picks


# --- validation feature cache (computed ONCE, model-independent) -----------
def _featmat_one(feat_fn, window, sid, x_h, x_on, onl_index):
    ctx = hist_context(x_h, HIST_CAP)
    n = len(x_on)
    rows, active = [], []
    for idx in range(n):
        # Match _stream_one exactly: it scores once the sliding buffer first reaches
        # MIN_N points, i.e. from idx == MIN_N - 1 onward; earlier steps stay 0.
        if idx < MIN_N - 1:
            continue   # warm-up: score stays 0 for every model
        lo = max(0, idx - window + 1)
        rows.append(feat_fn(x_on[lo:idx + 1], ctx))
        active.append(idx)
    midx = pd.MultiIndex.from_product([[sid], onl_index], names=["id", "time"])
    feat = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, FEAT_DIM))
    return midx, np.asarray(active, dtype=int), feat


def precompute_fold(feat_fn, window, X, fold_ids):
    jobs = []
    for sid in fold_ids:
        x_h, onl = split_series(X.loc[sid])
        jobs.append((sid, x_h, onl.to_numpy(dtype=np.float64), onl.index))
    with threadpool_limits(limits=1):
        parts = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(_featmat_one)(feat_fn, window, sid, x_h, x_on, oi)
            for (sid, x_h, x_on, oi) in jobs
        )
    indices, feats, active_all = [], [], []
    offset = 0
    for midx, active, feat in parts:
        indices.append(midx)
        feats.append(feat)
        active_all.append(active + offset)
        offset += len(midx)
    index = indices[0].append(indices[1:]) if len(indices) > 1 else indices[0]
    # Width-robust stack: empty (warm-up-only) series come back with 0 rows; coerce them
    # to the real feature width so any feat_fn (45-wide champion OR 500-wide window vec) stacks.
    nonempty = [f for f in feats if f.shape[0] > 0]
    width = nonempty[0].shape[1] if nonempty else FEAT_DIM
    feats = [f if f.shape[0] > 0 else np.zeros((0, width)) for f in feats]
    feat = np.vstack(feats) if feats else np.zeros((0, width))
    active = np.concatenate(active_all) if active_all else np.zeros(0, dtype=int)
    return {"index": index, "feat": feat, "active": active}


def score_fold(model, cache):
    full = np.zeros(len(cache["index"]))
    if len(cache["feat"]):
        full[cache["active"]] = model.predict_proba(cache["feat"])[:, 1]
    return pd.Series(full, index=cache["index"])


# --- evaluation ------------------------------------------------------------
def eval_model(model, Xtr, ytr, caches, y_val):
    with threadpool_limits(limits=1):
        model.fit(Xtr, ytr)
    series = [score_fold(model, c) for c in caches]
    aucs = np.asarray([ts_auc_from(s.to_frame("prediction"), y_val) for s in series])
    return aucs, series


def blend_aucs(series_a, series_b, w_b, y_val):
    """Weighted probability blend: (1-w)*A + w*B, per-fold TS-AUC."""
    out = []
    for sa, sb in zip(series_a, series_b):
        blended = (1.0 - w_b) * sa + w_b * sb
        out.append(ts_auc_from(blended.to_frame("prediction"), y_val))
    return np.asarray(out)


def report(name, aucs, base_aucs):
    if base_aucs is None:
        print(f"  {name:<28} mean={aucs.mean():.4f}  (baseline)", flush=True)
        return
    md, sd, t, _ = paired(aucs, base_aucs)
    if md > 0 and t >= 2.5:
        verdict = "SHIP(hard)"
    elif md > 0 and t >= 2.0:
        verdict = "marginal(HOLD)"
    elif md > 0 and t >= 1.5:
        verdict = "weak"
    else:
        verdict = "noise"
    print(f"  {name:<28} mean={aucs.mean():.4f}  delta={md:+.4f}  t={t:+.2f}  -> {verdict}",
          flush=True)


HGB_GRID = [
    ("hgb_champion", dict()),
    ("hgb_lr0.03_it300", dict(learning_rate=0.03, max_iter=300)),
    ("hgb_lr0.08", dict(learning_rate=0.08)),
    ("hgb_leaves15", dict(max_leaf_nodes=15)),
    ("hgb_leaves63", dict(max_leaf_nodes=63)),
    ("hgb_l2_2.0", dict(l2_regularization=2.0)),
    ("hgb_msl40", dict(min_samples_leaf=40)),
]
LR_GRID = [
    ("lr_C0.1", dict(C=0.1)),
    ("lr_C0.3", dict(C=0.3)),
    ("lr_C1.0", dict(C=1.0)),
    ("lr_C3.0", dict(C=3.0)),
    ("lr_C1.0_bal", dict(C=1.0, class_weight="balanced")),
]
BLEND_WEIGHTS = [0.2, 0.3, 0.4, 0.5]


def main():
    t0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print(f"=== ensemble+tune ({mode}) start {datetime.now():%Y-%m-%d %H:%M:%S} ===", flush=True)

    X, yi, y_val, train_ids, folds = load_pool_strat()
    picks = select_picks(X, train_ids)
    brk = (yi["tau_index"] >= 0)
    rates = [f"{brk.loc[f].mean():.3f}" for f in folds]
    print(f"pool: {len(train_ids)} train, {len(folds)} folds x {FOLD_SIZE} "
          f"(20% val, stratified). per-fold break-rate: {rates}", flush=True)

    Xtr, ytr = build_training(X, yi, picks, champion_feat, WINDOW)
    print(f"train matrix {Xtr.shape}, positives {ytr.mean():.3f}", flush=True)

    tf = time.time()
    caches = [precompute_fold(champion_feat, WINDOW, X, f) for f in folds]
    print(f"val feature cache built in {(time.time()-tf)/60:.1f} min "
          f"({sum(len(c['feat']) for c in caches)} scored rows)\n", flush=True)

    # --- baselines ---
    print("-- baselines --", flush=True)
    base_aucs, hgb_series = eval_model(make_hgb(), Xtr, ytr, caches, y_val)
    report("HGB champion", base_aucs, None)
    lr_aucs, lr_series = eval_model(make_lr(), Xtr, ytr, caches, y_val)
    report("LR default", lr_aucs, base_aucs)

    # --- HGB + LR blend weight sweep (uses cached series, cheap) ---
    print("\n-- HGB + LR probability blend (weight on LR) --", flush=True)
    best_blend = (None, -1.0, None)
    for w in BLEND_WEIGHTS:
        a = blend_aucs(hgb_series, lr_series, w, y_val)
        report(f"blend w_lr={w:.1f}", a, base_aucs)
        if a.mean() > best_blend[1]:
            best_blend = (w, a.mean(), a)

    # --- HGB tuning grid ---
    print("\n-- HGB tuning (TS-AUC, vs champion) --", flush=True)
    best_hgb = ("hgb_champion", base_aucs.mean(), hgb_series, dict())
    for name, over in HGB_GRID[1:]:
        a, s = eval_model(make_hgb(**over), Xtr, ytr, caches, y_val)
        report(name, a, base_aucs)
        if a.mean() > best_hgb[1]:
            best_hgb = (name, a.mean(), s, over)

    # --- LR tuning grid ---
    print("\n-- LR tuning (TS-AUC, vs HGB champion) --", flush=True)
    best_lr = ("lr_default", lr_aucs.mean(), lr_series, dict())
    for name, over in LR_GRID:
        a, s = eval_model(make_lr(**over), Xtr, ytr, caches, y_val)
        report(name, a, base_aucs)
        if a.mean() > best_lr[1]:
            best_lr = (name, a.mean(), s, over)

    # --- best HGB (x) best LR blend sweep ---
    print(f"\n-- best-HGB ({best_hgb[0]}) + best-LR ({best_lr[0]}) blend --", flush=True)
    for w in BLEND_WEIGHTS:
        a = blend_aucs(best_hgb[2], best_lr[2], w, y_val)
        report(f"best-blend w_lr={w:.1f}", a, base_aucs)

    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"champion mean={base_aucs.mean():.4f}", flush=True)
    print(f"best single HGB: {best_hgb[0]} ({best_hgb[1]:.4f}) over={best_hgb[3]}", flush=True)
    print(f"best single LR:  {best_lr[0]} ({best_lr[1]:.4f}) over={best_lr[3]}", flush=True)
    print(f"best HGB+LR blend (default members): w_lr={best_blend[0]} mean={best_blend[1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
