"""Build streaming_detector_rankerrf.ipynb by trimming streaming_detector_stack4.ipynb
down to the 2-member ranker+rf ship (blend 0.45/0.55) with the confirmed new-feature block
perm_entropy4 + max_split_lr + mosum appended to the champion feature contract.

3-seed isolation on the ranker+rf blend (2026-08-24):
  base                 0.62503
  +perm4               0.62558  (+0.00055, t=4.39, 3/3)
  +mosum               0.62610  (+0.00107, t=0.92, 2/3)   <- unreliable alone
  +perm4+mosum         0.62642  (+0.00139, t=1.67, 3/3)
  +perm4+maxlr+mosum   0.62725  (+0.00222, t=5.88, 3/3)   <- ship
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "streaming_detector_stack4.ipynb")
DST = os.path.join(HERE, "streaming_detector_rankerrf.ipynb")

nb = json.load(open(SRC))
cells = nb["cells"]


def src_of(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)
    if cells[i]["cell_type"] == "code":
        cells[i]["outputs"] = []
        cells[i]["execution_count"] = None


def find(substr):
    for i, c in enumerate(cells):
        if substr in "".join(c["source"]):
            return i
    raise KeyError(substr)


# ---------------------------------------------------------------- 0: title
set_src(0, """# 2-member ship: ranker + RandomForest (drops hgb / xgb / catboost)

The deployed **stack4** (5-member percentile stack) only reached LB **62.59%**. A cached-OOF A/B
showed **ranker + rf alone == the full 5-member stack** (+0.0016 vs champ, 7/7 seeds) — the extra
hgb/xgb/catboost members add nothing; rf carries the gain. So this notebook ships the minimal
**ranker (0.45) + rf (0.55)** fixed-CDF percentile blend.

On top of that base, a 3-seed isolation on the ranker+rf blend (2026-08-24) added three new feature
families to the champion contract:

| combo | score | mean Δ | t | seeds+ |
|---|---|---|---|---|
| base (ranker+rf) | 0.62503 | — | — | 0/3 |
| +perm_entropy4 | 0.62558 | +0.00055 | 4.39 | 3/3 |
| +mosum | 0.62610 | +0.00107 | 0.92 | 2/3 |
| +perm_entropy4+mosum | 0.62642 | +0.00139 | 1.67 | 3/3 |
| **+perm_entropy4+max_split_lr+mosum** | **0.62725** | **+0.00222** | **5.88** | **3/3** |

`mosum` alone is unreliable (2/3 seeds, driven by one seed); but the **three together** win on every
metric (highest score, highest mean, highest t, 3/3). The features are not additive individually —
the combination is what fires. Ship set = **perm_entropy4 + max_split_lr + mosum** (12 extra cols,
3 families x 3 MS scales).

Everything upstream of the models is the unchanged champion pipeline (base-15 x multi-scale +
residual-variance block + dependence/early-onset block = 61 cols) plus the +12 ship-feature block.
""")

# ---------------------------------------------------------------- pip: drop xgb/catboost
i_pip = find("%pip install crunch-cli")
set_src(i_pip, """%pip install crunch-cli lightgbm onnxruntime skl2onnx onnx --upgrade --quiet --progress-bar off
!crunch setup-notebook structural-break-real-time NUTwmjJVnni1JlK9HB9prtDI
""")

# ---------------------------------------------------------------- imports: drop xgb/cat/hgb
i_imp = find("from threadpoolctl import threadpool_limits")
set_src(i_imp, """import os

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
""")

# ---------------------------------------------------------------- config: BLEND + params
i_cfg = find("BLEND = {")
set_src(i_cfg, """# @crunch/keep:on
WINDOW           = 250   # buffer = largest MS scale (E17 multi-scale)
MS_SCALES        = (60, 130, 250)  # E17 champion ms_60_130_250 (45 feats); shared by both models
MIN_N            = 15    # warm-up: emit 0.0 until this many online points seen
HIST_CAP         = 200   # max historical points sampled for the KS test
PREDICT_EVERY    = 1     # recompute features + prediction every N steps (1 = no staleness)
STEPS_PER_SERIES = 10    # sampled online steps per training series for rf (raised to 30 in resid block)
DENSE_K          = 60    # sampled online steps per series for the ranker (dense trajectory)
MAX_TRAIN_SERIES = 10**9 # consume ALL series the platform provides (no cap)
SEED             = 42
RANK_SEED        = 42    # ranker model seed

# 2-member ship blend: ranker + rf in fixed-CDF percentile space (each member's live score ->
# its train-CDF percentile). W_RANK=0.45 / W_RF=0.55 matches the cached-OOF optimum plateau and
# the 3-seed isolation base. Sum = 1.0.
BLEND = {
    "ranker": 0.45,
    "rf":     0.55,
}

# LGBM listwise ranker (E33/E36): rank_xendcg, query group = online step t
LGB_PARAMS = dict(
    objective="rank_xendcg", n_estimators=300, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, reg_lambda=1.0, subsample=1.0, colsample_bytree=0.7,
    random_state=RANK_SEED, verbose=-1,
)

# RandomForest recipe (zoo fine-search winner, on the champion features).
RF_PARAMS  = dict(n_estimators=800, max_features=0.5, min_samples_leaf=20)
""")

# ---------------------------------------------------------------- insert ship-feature cell after dep block
i_dep = find("dependence block (b_no_accf)")
ship_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": ["""# @crunch/keep:on
# =============================================================================
# Ship-feature block (2026-08-24): perm_entropy4 + max_split_lr + mosum
# -----------------------------------------------------------------------------
# 3-seed isolation on the ranker+rf blend: the three families TOGETHER add
# +0.00222 TS-AUC (t=5.88, 3/3 seeds) on top of the 61-col champion base. Each
# family is computed on the same MS_SCALES sub-windows as the champion base and
# appended (+12 cols = 3 families [perm4:1, maxlr:1, mosum:2] x 3 scales).
# Rides along via the compute_features_ms wrapper (late binding), so train() and
# infer() pick it up unchanged.
# =============================================================================
_SHIP_EPS = 1e-12
_SHIP_MSPLIT_MIN_SEG = 5


def _ship_perm_entropy4(w):
    n = len(w)
    if n < 8:
        return [0.0]
    counts = {}
    for i in range(n - 3):
        pat = tuple(np.argsort(w[i:i + 4]))
        counts[pat] = counts.get(pat, 0) + 1
    p = np.array(list(counts.values()), dtype=np.float64)
    p /= p.sum()
    return [float(-(p * np.log(p)).sum()) / math.log(24)]


def _ship_msplit_prefix(w):
    n = len(w)
    lo, hi = _SHIP_MSPLIT_MIN_SEG, n - _SHIP_MSPLIT_MIN_SEG
    if hi <= lo:
        return None
    c1 = np.cumsum(w)
    c2 = np.cumsum(w * w)
    tot1 = c1[-1]
    tot2 = c2[-1]
    ks = np.arange(lo, hi + 1, dtype=np.float64)
    kidx = ks.astype(np.int64) - 1
    s1l = c1[kidx]
    s2l = c2[kidx]
    nl = ks
    nr = n - ks
    s1r = tot1 - s1l
    s2r = tot2 - s2l
    ssl = np.maximum(s2l - s1l * s1l / nl, 0.0)
    ssr = np.maximum(s2r - s1r * s1r / nr, 0.0)
    return n, nl, nr, s1l, s1r, ssl, ssr


def _ship_max_split_lr(w):
    pf = _ship_msplit_prefix(w)
    if pf is None:
        return [0.0]
    n, nl, nr, s1l, s1r, ssl, ssr = pf
    var0 = max(float(np.sum(w * w) / n - (float(np.sum(w)) / n) ** 2), 1e-12)
    varl = np.maximum(ssl / nl, 1e-12)
    varr = np.maximum(ssr / nr, 1e-12)
    llr = 0.5 * (n * math.log(var0) - nl * np.log(varl) - nr * np.log(varr))
    return [float(np.max(np.maximum(llr, 0.0)))]


def _ship_mosum(w, ctx, k=10):
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / max(ctx["sd"], 1e-8)
    sq = z * z
    n = len(sq)
    if n < 2 * k:
        return [0.0, 1.0]
    c = np.concatenate([[0.0], np.cumsum(sq)])
    blocks = (c[k:] - c[:-k]) / k
    mx = float(blocks.max())
    mn = float(blocks.mean())
    return [math.log(mx + _SHIP_EPS), mx / max(mn, _SHIP_EPS)]


def _ship_feats(w, ctx):
    w = np.asarray(w, dtype=np.float64)
    out = []
    for s in MS_SCALES:
        ws = w if s >= len(w) else w[-s:]
        out.extend(_ship_perm_entropy4(ws))
        out.extend(_ship_max_split_lr(ws))
        out.extend(_ship_mosum(ws, ctx))
    return out


# Chain onto the dependence block's wrappers (late binding); train()/infer() call by name.
_ship_base_compute_features_ms = compute_features_ms


def compute_features_ms(w, ctx):
    return _ship_base_compute_features_ms(w, ctx) + _ship_feats(w, ctx)


_n_ship = len(MS_SCALES) * 4
print(f"ship feature block: perm_entropy4 + max_split_lr + mosum x {len(MS_SCALES)} scales -> +{_n_ship} feats")
"""],
}
cells.insert(i_dep + 1, ship_cell)

# ---------------------------------------------------------------- train(): ranker + rf only
i_train = find("def _pointwise_ctor(name, seed):")
set_src(i_train, '''def _pointwise_ctor(name, seed):
    if name == "rf":
        n_jobs_fit = max(1, (os.cpu_count() or 4) - 1)
        return RandomForestClassifier(n_jobs=n_jobs_fit, random_state=seed, **RF_PARAMS)
    raise ValueError(name)


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    rng = np.random.default_rng(SEED)
    pt_rows: List[List[float]] = []
    pt_lab: List[int] = []
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

        # rf rows (STEPS_PER_SERIES steps/series)
        kh = min(STEPS_PER_SERIES, len(cand))
        for t in rng.choice(cand, size=kh, replace=False):
            lo = max(0, t - WINDOW + 1)
            pt_rows.append(compute_features_ms(x_on[lo:t + 1], ctx))
            pt_lab.append(1 if (tau is not None and t >= tau) else 0)

        # ranker rows (dense: DENSE_K steps/series), recording the group key (online step t)
        kr = min(DENSE_K, len(cand))
        for t in rng.choice(cand, size=kr, replace=False):
            lo = max(0, t - WINDOW + 1)
            rk_rows.append(compute_features_ms(x_on[lo:t + 1], ctx))
            rk_labels.append(1 if (tau is not None and t >= tau) else 0)
            rk_t.append(int(t))
        n_used += 1

    X_pt = np.asarray(pt_rows, dtype=np.float64)
    y_pt = np.asarray(pt_lab, dtype=np.int64)
    print(f"pointwise matrix {X_pt.shape} from {n_used} series, {y_pt.mean():.3f} positive")

    models, refs = {}, {}

    # --- RandomForest (the only pointwise member) -> ONNX: single-row streaming predict is
    #     ~300x faster than sklearn's predict_proba, with identical rankings ---
    rf_clf = _pointwise_ctor("rf", SEED)
    with threadpool_limits(limits=None):
        rf_clf.fit(X_pt, y_pt)
    onx = convert_sklearn(
        rf_clf,
        initial_types=[("input", FloatTensorType([None, rf_clf.n_features_in_]))],
        options={type(rf_clf): {"zipmap": False}},
        target_opset=17,
    )
    rf_onnx_bytes = onx.SerializeToString()
    # recompute rf's reference distribution from ONNX so inference percentiles
    # are exactly consistent with what infer() computes
    _sess = ort.InferenceSession(rf_onnx_bytes, providers=["CPUExecutionProvider"])
    _pn = _sess.get_outputs()[1].name
    _rf_pp = _sess.run([_pn], {"input": X_pt.astype(np.float32)})[0][:, 1]
    refs["rf"] = np.sort(_rf_pp.astype(np.float64))
    print(f"  trained rf -> ONNX ({len(rf_onnx_bytes) / 1e6:.1f} MB)")

    # --- LGBM listwise ranker: sort rows so each online step is a contiguous query group ---
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
    models["ranker"] = ranker
    refs["ranker"] = np.sort(ranker.predict(X_rk))

    joblib.dump(
        {
            "members": [("ranker", "ranker"), ("rf", "onnx")],
            "models": models,
            "refs": refs,
            "rf_onnx": rf_onnx_bytes,
            "blend": dict(BLEND),
            "config": {
                "WINDOW": WINDOW,
                "MS_SCALES": list(MS_SCALES),
                "MIN_N": MIN_N,
                "HIST_CAP": HIST_CAP,
                "PREDICT_EVERY": PREDICT_EVERY,
            },
        },
        os.path.join(model_directory_path, "model.joblib"),
        compress=3,  # gzip the ~370MB ONNX RF blob -> ~100MB artifact
    )
    print(f"saved 2 members (ranker + rf ONNX); active blend = "
          f"{ {k: v for k, v in BLEND.items() if v > 0} }")
''')

# ---------------------------------------------------------------- infer parallelism: fix OOM
# MEASURED (2026-08-25, _measure_rf_mem.py, full-data 73-feat RF): each ONNX RF worker
# transiently spikes to ~11-12.5 GB PEAK RSS while onnxruntime builds the TreeEnsemble
# session (settling to ~3 GB steady). crunch spawns all workers at once -> simultaneous
# build peaks -> the old _PER_WORKER_GB=3.7 (a steady-state figure) over-provisioned 15
# workers -> 15x~12 GB OOM-killed the run. Size to the build PEAK instead.
i_par = find("_PER_WORKER_GB = 3.7")
set_src(i_par, """# @crunch/keep:on
# crunch runs each infer worker in its OWN process; every worker builds its own onnxruntime
# session from the RF ONNX blob. MEASURED (2026-08-25, full-data 73-feat 800-tree RF): the
# session BUILD transiently spikes to ~11-12.5 GB PEAK RSS/worker (settling to ~3 GB steady).
# crunch starts all workers at once, so their build peaks COINCIDE -> total peak = workers x
# ~12 GB. The earlier 3.7 (steady) figure over-provisioned 15 workers and OOM-killed infer.
# Size the worker count to the BUILD PEAK to be safe (64 GB -> 4 workers).
_PER_WORKER_GB = 12.5
_cpu = max(1, (os.cpu_count() or 4) - 1)
try:
    _ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    _mem_cap = max(1, int(_ram_gb * 0.90 / _PER_WORKER_GB))
except (ValueError, OSError):
    _mem_cap = _cpu
INFER_PARALLELISM = max(1, min(_cpu, _mem_cap))
print(f"INFER_PARALLELISM={INFER_PARALLELISM} (cpu_cap={_cpu}, mem_cap={_mem_cap} @ {_PER_WORKER_GB} GB/worker)")
# INFER_PARALLELISM = 1  # set to 1 while debugging
""")

# ---------------------------------------------------------------- submit message
i_sub = find("crunch_tools.submit(")
set_src(i_sub, '''Message = "2-member ranker+rf percentile blend (.45/.55) + perm_entropy4+max_split_lr+mosum (isolation +0.0022, t=5.88, 3/3)"
crunch_tools.submit(message=Message)
''')

json.dump(nb, open(DST, "w"), indent=1)
print(f"wrote {DST} with {len(cells)} cells")
