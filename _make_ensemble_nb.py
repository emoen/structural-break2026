"""Generate streaming_detector_ensemble.ipynb (E36 rank+tuned ship-realistic blend).

Starts from the current champion streaming_detector_tuned.ipynb (so the feature code
-- hist_context / compute_features / compute_features_ms -- is reused VERBATIM and stays
bit-identical to the validated champion_feat), and swaps only:
  * setup cell   : also pip-install lightgbm
  * imports cell : + import lightgbm
  * config cell  : + DENSE_K, RANK_SEED, BLEND_W and LGBM ranker params
  * train()      : fit tuned HGB (pointwise) AND an LGBMRanker(rank_xendcg, group=online step);
                   store each model's sorted TRAINING-score reference (empirical CDF)
  * infer()      : per step emit  w*pct_hgb(proba) + (1-w)*pct_rank(raw)  via fixed train-CDF
  * top markdown : describe the ensemble

The fixed-train-CDF blend is the SHIP-REALISTIC computation confirmed on fresh seeds
{31337,2025,555,99}: w=0.6 mean_d +0.0080 (min +0.0018, t +3.07, positive 4/4).
"""
import nbformat as nbf

SRC = "streaming_detector_tuned.ipynb"
DST = "streaming_detector_ensemble.ipynb"

nb = nbf.read(SRC, as_version=4)

TOP_MD = """# Rank + Tuned Ensemble Streaming Detector (E36)

An ensemble of two decorrelated models over the SAME 45 multi-scale window-vs-history
features (base15 on trailing sub-windows 60/130/250):

1. **Tuned HGB champion** (E32, LB 57.72%) — a pointwise `HistGradientBoostingClassifier`
   optimising log-loss.
2. **LGBM listwise ranker** (`rank_xendcg`, query group = online step) — trained on the
   dense per-step online trajectory, optimising the within-step *ordering* that TS-AUC
   actually scores.

The two are blended by converting each model's output to a **fixed empirical-CDF
percentile** (estimated once from the training scores) and averaging:

`score = w · F̂_hgb(proba) + (1 − w) · F̂_rank(raw)`,  with **w = 0.6**.

Because TS-AUC is a within-step ordering metric, this fixed monotone transform is the
deployable equivalent of a cross-sectional rank — no future/other-series info is used.
The champion anchors the blend; the decorrelated ranker corrects within-step orderings,
a variance-reduction that turned a non-robust standalone-rank signal into a robust
ensemble. Confirmed on fresh disjoint seeds {31337, 2025, 555, 99}: w=0.6 mean Δ **+0.0080**
(min +0.0018, t **+3.07**, positive on all 4) vs the tuned champion."""

SETUP = """%pip install crunch-cli lightgbm --upgrade --quiet --progress-bar off
!crunch setup-notebook structural-break-real-time bf6s5LCM9xL3PRAospKyui4T"""

IMPORTS = """import os

# Keep BLAS / OpenMP single-threaded. infer() runs in several parallel
# processes, and single-row model predictions are dramatically faster without
# per-call thread-pool spin-up. Env vars only work if set before numpy loads
# (not guaranteed on the cloud), so infer() ALSO caps threads at call time via
# threadpool_limits below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
from collections import deque
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score"""

CONFIG = """# @crunch/keep:on
WINDOW           = 250   # buffer = largest MS scale (E17 multi-scale)
MS_SCALES        = (60, 130, 250)  # E17 champion ms_60_130_250 (45 feats); shared by both models
MIN_N            = 15    # warm-up: emit 0.0 until this many online points seen
HIST_CAP         = 200   # max historical points sampled for the KS test
PREDICT_EVERY    = 1     # recompute features + prediction every N steps (1 = no staleness)
STEPS_PER_SERIES = 10    # sampled online steps per training series for the HGB (pointwise)
DENSE_K          = 60    # sampled online steps per series for the ranker (dense trajectory)
MAX_TRAIN_SERIES = 10**9 # consume ALL series the platform provides (no cap)
MAX_ITER         = 400   # E32 tuned HGB winner
SEED             = 42
RANK_SEED        = 42    # ranker model seed
BLEND_W          = 0.6   # score = w*pct_hgb + (1-w)*pct_rank; confirmed w=0.6 (fresh seeds, t=+3.07)

# LGBM listwise ranker (E33/E36): rank_xendcg, query group = online step t
LGB_PARAMS = dict(
    objective="rank_xendcg", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, reg_lambda=1.0, subsample=1.0, colsample_bytree=0.7,
    random_state=RANK_SEED, verbose=-1,
)"""

TRAIN = '''def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    """Fit two decorrelated models on the SAME 45 multi-scale features:

    * a tuned HGB (pointwise, STEPS_PER_SERIES steps/series), and
    * an LGBM ranker (rank_xendcg, DENSE_K steps/series, query group = online step t).

    Each model's SORTED training-score vector is stored as its empirical-CDF reference so
    infer() can convert a live output to a fixed percentile without any cross-series state.
    """
    rng = np.random.default_rng(SEED)
    hgb_rows: List[List[float]] = []
    hgb_labels: List[int] = []
    rk_rows: List[List[float]] = []
    rk_labels: List[int] = []
    rk_t: List[int] = []

    n_used = 0
    for dataset_id, x_hist, x_online, tau in datasets:
        if n_used >= MAX_TRAIN_SERIES:
            break
        x_h = np.asarray(x_hist, dtype=np.float64)
        x_on = np.asarray(list(x_online), dtype=np.float64)  # materialize (iterable once)
        L = len(x_on)
        if L <= MIN_N or len(x_h) < 5:
            continue
        ctx = hist_context(x_h, HIST_CAP)
        cand = np.arange(MIN_N, L)

        # HGB pointwise rows (10 steps/series, matches the champion)
        kh = min(STEPS_PER_SERIES, len(cand))
        for t in rng.choice(cand, size=kh, replace=False):
            lo = max(0, t - WINDOW + 1)
            hgb_rows.append(compute_features_ms(x_on[lo:t + 1], ctx))
            hgb_labels.append(1 if (tau is not None and t >= tau) else 0)

        # Ranker rows (dense: DENSE_K steps/series), recording the group key (online step t)
        kr = min(DENSE_K, len(cand))
        for t in rng.choice(cand, size=kr, replace=False):
            lo = max(0, t - WINDOW + 1)
            rk_rows.append(compute_features_ms(x_on[lo:t + 1], ctx))
            rk_labels.append(1 if (tau is not None and t >= tau) else 0)
            rk_t.append(int(t))
        n_used += 1

    # --- tuned HGB champion (pointwise) ---
    X_hgb = np.asarray(hgb_rows, dtype=np.float64)
    y_hgb = np.asarray(hgb_labels, dtype=np.int64)
    print(f"HGB matrix {X_hgb.shape} from {n_used} series, positives {y_hgb.mean():.3f}")
    hgb = HistGradientBoostingClassifier(
        max_iter=MAX_ITER, learning_rate=0.021145795969329348, max_leaf_nodes=63,
        min_samples_leaf=10, l2_regularization=1.0, max_features=0.5,
        early_stopping=True, random_state=SEED,
    )
    hgb.fit(X_hgb, y_hgb)
    print(f"trained HGB {hgb.n_iter_} trees")
    ref_hgb = np.sort(hgb.predict_proba(X_hgb)[:, 1])

    # --- LGBM listwise ranker (rank_xendcg): sort rows so each online step is a contiguous group ---
    X_rk = np.asarray(rk_rows, dtype=np.float64)
    y_rk = np.asarray(rk_labels, dtype=np.int64)
    t_rk = np.asarray(rk_t, dtype=np.int64)
    order = np.argsort(t_rk, kind="stable")
    X_rk, y_rk, t_rk = X_rk[order], y_rk[order], t_rk[order]
    _, groups = np.unique(t_rk, return_counts=True)
    print(f"ranker matrix {X_rk.shape}, {len(groups)} query groups (online steps)")
    n_jobs_fit = max(1, (os.cpu_count() or 4) - 1)
    ranker = lgb.LGBMRanker(n_jobs=n_jobs_fit, **LGB_PARAMS)
    with threadpool_limits(limits=1):
        ranker.fit(X_rk, y_rk, group=groups)
    ranker.set_params(n_jobs=1)  # single-row streaming predict at inference
    ref_rank = np.sort(ranker.predict(X_rk))

    joblib.dump(
        {
            "hgb": hgb,
            "ranker": ranker,
            "ref_hgb": ref_hgb,
            "ref_rank": ref_rank,
            "blend_w": BLEND_W,
            "config": {
                "WINDOW": WINDOW,
                "MS_SCALES": list(MS_SCALES),
                "MIN_N": MIN_N,
                "HIST_CAP": HIST_CAP,
                "PREDICT_EVERY": PREDICT_EVERY,
            },
        },
        os.path.join(model_directory_path, "model.joblib"),
    )'''

INFER = '''def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    """One score per online point = fixed-CDF percentile blend of the two models.

    score = w * F_hgb(proba) + (1 - w) * F_rank(raw), where F_* are each model's empirical
    CDF from TRAINING scores (searchsorted / N). No cross-series or future info is used, so
    this is exactly deployable; TS-AUC only needs the within-step ordering.
    """
    saved = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    hgb = saved["hgb"]
    ranker = saved["ranker"]
    ref_hgb = np.asarray(saved["ref_hgb"], dtype=np.float64)
    ref_rank = np.asarray(saved["ref_rank"], dtype=np.float64)
    w = float(saved["blend_w"])
    cfg = saved["config"]
    window        = int(cfg["WINDOW"])
    min_n         = int(cfg["MIN_N"])
    hist_cap      = int(cfg["HIST_CAP"])
    predict_every = int(cfg["PREDICT_EVERY"])
    global MS_SCALES
    MS_SCALES = tuple(cfg.get("MS_SCALES", MS_SCALES))
    n_hgb = len(ref_hgb)
    n_rank = len(ref_rank)
    try:
        ranker.set_params(n_jobs=1)
    except Exception:
        pass

    yield  # Signal readiness to the runner.

    # Force single-threaded BLAS/OpenMP at call time (see the champion notebook for why).
    with threadpool_limits(limits=1):
        for x_historical, x_online in datasets:
            x_h = np.asarray(x_historical, dtype=np.float64)
            ctx = hist_context(x_h, hist_cap)

            buf = deque()
            score = 0.0
            for idx, point in enumerate(x_online):
                if len(buf) == window:
                    buf.popleft()
                buf.append(float(point))
                n = len(buf)

                if n < min_n:
                    yield 0.0
                    continue

                if idx % predict_every == 0:
                    feats = np.asarray(compute_features_ms(np.asarray(buf), ctx),
                                       dtype=np.float64).reshape(1, -1)
                    proba = float(hgb.predict_proba(feats)[0, 1])
                    raw = float(ranker.predict(feats)[0])
                    pct_hgb = float(np.searchsorted(ref_hgb, proba, side="right")) / n_hgb
                    pct_rank = float(np.searchsorted(ref_rank, raw, side="right")) / n_rank
                    score = w * pct_hgb + (1.0 - w) * pct_rank

                yield float(score)'''


def replace_cell(nb, matcher, new_src):
    for cell in nb.cells:
        if matcher(cell.source):
            cell.source = new_src
            return True
    raise SystemExit(f"no cell matched for: {new_src[:50]!r}")


replace_cell(nb, lambda s: s.startswith("# Supervised Streaming Structural-Break Detector"), TOP_MD)
replace_cell(nb, lambda s: "crunch setup-notebook" in s, SETUP)
replace_cell(nb, lambda s: "from sklearn.ensemble import HistGradientBoostingClassifier" in s
             and "import joblib" in s, IMPORTS)
replace_cell(nb, lambda s: "WINDOW           = 250" in s, CONFIG)
replace_cell(nb, lambda s: s.lstrip().startswith("def train("), TRAIN)
replace_cell(nb, lambda s: s.lstrip().startswith("def infer("), INFER)

nbf.write(nb, DST)
print(f"wrote {DST} with {len(nb.cells)} cells")
