"""Parity test: the parallelized train() feature build must be BYTE-IDENTICAL to
the original serial loop. Uses the notebook's OWN feature functions (exec'd from
the cells) driven by synthetic datasets -- parity only needs the two paths to see
identical inputs, not real labels.

Run: .venv/bin/python -u _parity_parallel.py
"""
import ast
import json

import numpy as np
from threadpoolctl import threadpool_limits
import joblib

NB = "streaming_detector_stack4.ipynb"


def _clean(src):
    return "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith(("%", "!")))


# --- exec the notebook's feature-definition + config cells into G ---
nb = json.load(open(NB))
train_idx = next(i for i, c in enumerate(nb["cells"])
                 if c["cell_type"] == "code" and "def train(" in "".join(c["source"]))

# also ast-parse the (edited) train cell as a sanity check
train_src = _clean("".join(nb["cells"][train_idx]["source"]))
ast.parse(train_src)
print(f"train cell parses OK (cell index {train_idx})")

G = {}
for i, c in enumerate(nb["cells"][:train_idx]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if ("import crunch" in src or "crunch.load_notebook" in src
            or "crunch_tools" in src or "%pip" in src or "load_data" in src):
        continue
    exec(_clean(src), G)

hist_context = G["hist_context"]
fit_calib_ref = G["fit_calib_ref"]
compute_features_ms = G["compute_features_ms"]
WINDOW = G["WINDOW"]
MIN_N = G["MIN_N"]
HIST_CAP = G["HIST_CAP"]
SEED = G["SEED"]
STEPS_PER_SERIES = G["STEPS_PER_SERIES"]
DENSE_K = G["DENSE_K"]
MAX_TRAIN_SERIES = G["MAX_TRAIN_SERIES"]
print(f"config: SEED={SEED} WINDOW={WINDOW} MIN_N={MIN_N} HIST_CAP={HIST_CAP} "
      f"STEPS_PER_SERIES={STEPS_PER_SERIES} DENSE_K={DENSE_K} MAX_TRAIN_SERIES={MAX_TRAIN_SERIES}")


# --- worker (module-level so loky can pickle it), identical to the notebook's ---
def _series_features(x_h, x_on, tau, pt_idx, rk_idx):
    with threadpool_limits(limits=1):
        ctx = hist_context(x_h, HIST_CAP)
        ctx["calib"] = fit_calib_ref(x_h, ctx)
        pt_feats, pt_labs = [], []
        for t in pt_idx:
            lo = max(0, t - WINDOW + 1)
            pt_feats.append(compute_features_ms(x_on[lo:t + 1], ctx))
            pt_labs.append(1 if (tau is not None and t >= tau) else 0)
        rk_feats, rk_labs, rk_ts = [], [], []
        for t in rk_idx:
            lo = max(0, t - WINDOW + 1)
            rk_feats.append(compute_features_ms(x_on[lo:t + 1], ctx))
            rk_labs.append(1 if (tau is not None and t >= tau) else 0)
            rk_ts.append(int(t))
    return pt_feats, pt_labs, rk_feats, rk_labs, rk_ts


def serial_build(datasets):
    rng = np.random.default_rng(SEED)
    hgb_rows, hgb_lab, rk_rows, rk_labels, rk_t = [], [], [], [], []
    n_used = 0
    for dataset_id, x_hist, x_online, tau in datasets:
        if n_used >= MAX_TRAIN_SERIES:
            break
        x_h = np.asarray(x_hist, dtype=np.float64)
        x_on = np.asarray(list(x_online), dtype=np.float64)
        L = len(x_on)
        if L <= MIN_N or len(x_h) < 5:
            continue
        ctx = hist_context(x_h, HIST_CAP)
        ctx["calib"] = fit_calib_ref(x_h, ctx)
        cand = np.arange(MIN_N, L)
        kh = min(STEPS_PER_SERIES, len(cand))
        for t in rng.choice(cand, size=kh, replace=False):
            lo = max(0, t - WINDOW + 1)
            hgb_rows.append(compute_features_ms(x_on[lo:t + 1], ctx))
            hgb_lab.append(1 if (tau is not None and t >= tau) else 0)
        kr = min(DENSE_K, len(cand))
        for t in rng.choice(cand, size=kr, replace=False):
            lo = max(0, t - WINDOW + 1)
            rk_rows.append(compute_features_ms(x_on[lo:t + 1], ctx))
            rk_labels.append(1 if (tau is not None and t >= tau) else 0)
            rk_t.append(int(t))
        n_used += 1
    return (np.asarray(hgb_rows, np.float64), np.asarray(hgb_lab, np.int64),
            np.asarray(rk_rows, np.float64), np.asarray(rk_labels, np.int64),
            np.asarray(rk_t, np.int64), n_used)


def parallel_build(datasets, n_jobs):
    rng = np.random.default_rng(SEED)
    tasks = []
    n_used = 0
    for dataset_id, x_hist, x_online, tau in datasets:
        if n_used >= MAX_TRAIN_SERIES:
            break
        x_h = np.asarray(x_hist, dtype=np.float64)
        x_on = np.asarray(list(x_online), dtype=np.float64)
        L = len(x_on)
        if L <= MIN_N or len(x_h) < 5:
            continue
        cand = np.arange(MIN_N, L)
        kh = min(STEPS_PER_SERIES, len(cand))
        pt_idx = rng.choice(cand, size=kh, replace=False)
        kr = min(DENSE_K, len(cand))
        rk_idx = rng.choice(cand, size=kr, replace=False)
        tasks.append((x_h, x_on, tau, pt_idx, rk_idx))
        n_used += 1
    results = joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        joblib.delayed(_series_features)(x_h, x_on, tau, pt_idx, rk_idx)
        for (x_h, x_on, tau, pt_idx, rk_idx) in tasks
    )
    hgb_rows, hgb_lab, rk_rows, rk_labels, rk_t = [], [], [], [], []
    for pt_feats, pt_labs, rk_feats, rk_labs, rk_ts in results:
        hgb_rows.extend(pt_feats)
        hgb_lab.extend(pt_labs)
        rk_rows.extend(rk_feats)
        rk_labels.extend(rk_labs)
        rk_t.extend(rk_ts)
    return (np.asarray(hgb_rows, np.float64), np.asarray(hgb_lab, np.int64),
            np.asarray(rk_rows, np.float64), np.asarray(rk_labels, np.int64),
            np.asarray(rk_t, np.int64), n_used)


def _synth_datasets():
    gen = np.random.default_rng(999)
    ds = []
    # edge cases that MUST be skipped by the guards (before any feature build)
    ds.append((0, gen.standard_normal(3).tolist(), gen.standard_normal(300).tolist(), 50))   # len(x_h)<5
    ds.append((1, gen.standard_normal(400).tolist(), gen.standard_normal(MIN_N).tolist(), None))  # L<=MIN_N
    # normal accepted series with realistic history so calib (sliding win 250) is exercised
    for sid in range(2, 90):
        Lh = int(gen.integers(260, 1200))
        Lo = int(gen.integers(MIN_N + 1, 900))
        tau = None if gen.random() < 0.4 else int(gen.integers(0, Lo))
        ds.append((sid, gen.standard_normal(Lh).tolist(), gen.standard_normal(Lo).tolist(), tau))
    return ds


def _cmp(name, a, b):
    ok = (a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes())
    extra = ""
    if not ok and a.shape == b.shape and a.size:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        extra = f"  max|diff|={np.nanmax(d):.3e} nmismatch={(a!=b).sum()}"
    print(f"  {name:10s} serial{a.shape}/{a.dtype} vs parallel{b.shape}/{b.dtype} "
          f"-> {'IDENTICAL' if ok else 'DIFFERENT'}{extra}")
    return ok


if __name__ == "__main__":
    ds = _synth_datasets()
    print(f"\nsynthetic datasets: {len(ds)} series (2 edge-skip + rest accepted)")
    s = serial_build(ds)
    for nj in (4, 15):
        p = parallel_build(ds, nj)
        print(f"\nn_jobs={nj}: serial n_used={s[-1]} parallel n_used={p[-1]}")
        names = ["X_pt", "y_pt", "X_rk", "y_rk", "t_rk"]
        allok = all(_cmp(nm, s[k], p[k]) for k, nm in enumerate(names))
        assert s[-1] == p[-1], "n_used mismatch"
        print(f"  => {'ALL BYTE-IDENTICAL' if allok else '*** MISMATCH ***'}")
        assert allok, "parity FAILED"
    print("\nPARITY OK: parallel feature build is byte-identical to the serial champion.")
