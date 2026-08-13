"""Fast supervised streaming break detector — pure-numpy features (no scipy.stats).

Measures per-phase timing and infer throughput so we can confirm it fits the
platform time budget before porting to the notebook.
"""
import math
import os
import time
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

WINDOW = 75
MIN_N = 15
HIST_CAP = 200
PREDICT_EVERY = 1
SEED = 42
MAX_ITER = 200

N_TRAIN_SERIES = 5000
N_VAL_SERIES = 400
STEPS_PER_SERIES = 10

FEATURE_NAMES = [
    "mean_z", "log_var_ratio", "ks", "skew_diff", "kurt_diff",
    "ac1_diff", "ac2_diff", "iqr_ratio", "tail2", "tail3",
    "zero_cross", "slope_norm", "spec_centroid_diff", "rng_ratio",
]


def _autocorr(a, lag):
    if len(a) <= lag:
        return 0.0
    x = a[:-lag]
    y = a[lag:]
    xm = x.mean()
    ym = y.mean()
    denom = math.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _spectral_centroid(a):
    a = a - a.mean()
    sp = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a))
    s = sp.sum()
    if s < 1e-12:
        return 0.0
    return float((freqs * sp).sum() / s)


def _skew_kurt(w, mean_w, var_w):
    d = w - mean_w
    m2 = var_w
    if m2 < 1e-12:
        return 0.0, 0.0
    m3 = float(np.mean(d ** 3))
    m4 = float(np.mean(d ** 4))
    sk = m3 / (m2 ** 1.5)
    ku = m4 / (m2 ** 2) - 3.0
    return sk, ku


def hist_context(x_h, hist_cap):
    n = len(x_h)
    mu = float(x_h.mean()) if n else 0.0
    var = float(x_h.var(ddof=1)) if n > 1 else 1.0
    var = max(var, 1e-12)
    sd = math.sqrt(var)
    pop_var = float(x_h.var()) if n else 1.0
    sk, ku = _skew_kurt(x_h, mu, pop_var) if n > 2 else (0.0, 0.0)
    ac1 = _autocorr(x_h, 1)
    ac2 = _autocorr(x_h, 2)
    q25, q75 = np.quantile(x_h, [0.25, 0.75]) if n else (0.0, 0.0)
    iqr = float(q75 - q25)
    iqr = iqr if iqr > 1e-12 else 1e-12
    rng = float(x_h.max() - x_h.min()) if n else 1.0
    rng = rng if rng > 1e-12 else 1e-12
    sc = _spectral_centroid(x_h) if n > 4 else 0.0
    if n > hist_cap:
        stride = n // hist_cap
        sample = x_h[::stride][:hist_cap]
    else:
        sample = x_h
    sample_ordered = np.asarray(sample, dtype=np.float64)
    sample_sorted = np.sort(sample_ordered)
    return {
        "mu": mu, "sd": sd, "var": var, "skew": sk, "kurt": ku,
        "ac1": ac1, "ac2": ac2, "iqr": iqr, "rng": rng, "sc": sc,
        "sample_sorted": sample_sorted, "sample_ordered": sample_ordered,
    }


def _fast_ks(w_sorted, h_sorted):
    h = len(h_sorted)
    n = len(w_sorted)
    allv = np.concatenate([h_sorted, w_sorted])
    cdf_h = np.searchsorted(h_sorted, allv, side="right") / h
    cdf_w = np.searchsorted(w_sorted, allv, side="right") / n
    return float(np.max(np.abs(cdf_h - cdf_w)))


def compute_features(w, ctx):
    n = len(w)
    mean_w = float(w.mean())
    var_w = max(float(w.var()), 1e-12)
    se = ctx["sd"] / math.sqrt(n)
    mean_z = (mean_w - ctx["mu"]) / max(se, 1e-8)
    log_var_ratio = math.log(var_w / ctx["var"])
    w_sorted = np.sort(w)
    ks = _fast_ks(w_sorted, ctx["sample_sorted"])
    sk, ku = _skew_kurt(w, mean_w, var_w)
    skew_diff = sk - ctx["skew"]
    kurt_diff = ku - ctx["kurt"]
    ac1_diff = _autocorr(w, 1) - ctx["ac1"]
    ac2_diff = _autocorr(w, 2) - ctx["ac2"]
    # IQR from the already-sorted window (cheap).
    q25 = w_sorted[int(0.25 * (n - 1))]
    q75 = w_sorted[int(0.75 * (n - 1))]
    iqr_w = max(float(q75 - q25), 1e-12)
    iqr_ratio = iqr_w / ctx["iqr"]
    dev = np.abs(w - ctx["mu"]) / ctx["sd"]
    tail2 = float((dev > 2.0).mean())
    tail3 = float((dev > 3.0).mean())
    centered = w - ctx["mu"]
    zero_cross = float((np.diff(np.sign(centered)) != 0).mean()) if n > 1 else 0.0
    i = np.arange(n, dtype=np.float64)
    sx = i.sum()
    sxx = (i * i).sum()
    sy = w.sum()
    sxy = (i * w).sum()
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if abs(denom) > 1e-12 else 0.0
    slope_norm = slope * n / ctx["sd"]
    spec_centroid_diff = _spectral_centroid(w) - ctx["sc"]
    rng_ratio = float(w_sorted[-1] - w_sorted[0]) / ctx["rng"]
    return [
        mean_z, log_var_ratio, ks, skew_diff, kurt_diff,
        ac1_diff, ac2_diff, iqr_ratio, tail2, tail3,
        zero_cross, slope_norm, spec_centroid_diff, rng_ratio,
    ]


def split_series(g):
    x_h = g.loc[g["period"] == 1, "value"].to_numpy(dtype=np.float64)
    onl = g.loc[g["period"] == 2, "value"]
    return x_h, onl


def build_training(X, y_index, ids, rng):
    rows = []
    labels = []
    for sid in ids:
        g = X.loc[sid]
        x_h, onl = split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        L = len(x_on)
        if L <= MIN_N:
            continue
        ctx = hist_context(x_h, HIST_CAP)
        tau = int(y_index.loc[sid, "tau_index"])
        cand = np.arange(MIN_N, L)
        k = min(STEPS_PER_SERIES, len(cand))
        picks = rng.choice(cand, size=k, replace=False)
        for t in picks:
            lo = max(0, t - WINDOW + 1)
            w = x_on[lo:t + 1]
            rows.append(compute_features(w, ctx))
            labels.append(1 if (tau >= 0 and t >= tau) else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


_T = {"compute": 0.0, "predict": 0.0}


def stream_predict(model, x_h, x_on):
    ctx = hist_context(x_h, HIST_CAP)
    buf = deque()
    out = np.zeros(len(x_on))
    score = 0.0
    n_calls = 0
    for idx, point in enumerate(x_on):
        if len(buf) == WINDOW:
            buf.popleft()
        buf.append(float(point))
        n = len(buf)
        if n < MIN_N:
            out[idx] = 0.0
            continue
        if idx % PREDICT_EVERY == 0:
            t0 = time.perf_counter()
            w = np.asarray(buf)
            feats = np.asarray(compute_features(w, ctx), dtype=np.float64).reshape(1, -1)
            t1 = time.perf_counter()
            score = float(model.predict_proba(feats)[0, 1])
            t2 = time.perf_counter()
            _T["compute"] += t1 - t0
            _T["predict"] += t2 - t1
            n_calls += 1
        out[idx] = score
    return out, n_calls


def ts_auc_from(preds, y):
    merged = preds.merge(y, how="left", left_index=True, right_index=True)
    merged["time_online"] = merged.groupby("id").cumcount()
    wsum = tot = 0.0
    for _, group in merged.groupby("time_online"):
        labels = group["target"].values
        scores = group["prediction"].values
        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        wsum += float(n_pos * n_neg) * float(roc_auc_score(labels, scores))
        tot += float(n_pos * n_neg)
    return wsum / tot if tot > 0 else 0.5


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng.shuffle(all_ids)
    train_ids = np.sort(all_ids[:N_TRAIN_SERIES])
    val_ids = np.sort(all_ids[N_TRAIN_SERIES:N_TRAIN_SERIES + N_VAL_SERIES])
    keep = set(train_ids.tolist()) | set(val_ids.tolist())

    print("Loading train data...")
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(DATA, "y_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    print(f"  loaded in {time.time()-t0:.1f}s")

    tb = time.time()
    Xtr, ytr = build_training(X, yi, train_ids, rng)
    print(f"Built training set {Xtr.shape}, positives {ytr.mean():.3f} in {time.time()-tb:.1f}s")

    tm = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=SEED,
    )
    model.fit(Xtr, ytr)
    print(f"Trained HGB ({model.n_iter_} trees) in {time.time()-tm:.1f}s")

    ti = time.time()
    cols = []
    total_points = 0
    total_calls = 0
    for sid in val_ids:
        g = X.loc[sid]
        x_h, onl = split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
        preds, ncalls = stream_predict(model, x_h, x_on)
        total_points += len(x_on)
        total_calls += ncalls
        cols.append(pd.Series(preds, index=midx))
    infer_time = time.time() - ti
    preds = pd.concat(cols).to_frame("prediction")
    auc = ts_auc_from(preds, y)

    print(f"\nInfer: {total_points} points, {total_calls} feature calls in {infer_time:.1f}s")
    print(f"  -> {1e3*infer_time/max(total_calls,1):.3f} ms/feature-call, "
          f"{1e6*infer_time/max(total_points,1):.2f} us/point")
    print(f"  split: compute {_T['compute']:.1f}s ({1e3*_T['compute']/max(total_calls,1):.3f} ms/call), "
          f"predict {_T['predict']:.1f}s ({1e3*_T['predict']/max(total_calls,1):.3f} ms/call), "
          f"loop/other {infer_time-_T['compute']-_T['predict']:.1f}s")
    print(f"VAL TS-AUC (fast, {len(val_ids)} series): {auc:.4f}")

    # Extrapolate to full test (~10000 series, similar online lengths, parallelism 4).
    pts_per_series = total_points / len(val_ids)
    est_full_points = 10000 * pts_per_series
    est_seconds_1core = infer_time / total_points * est_full_points
    print(f"Est. full-test infer (~10000 series): "
          f"{est_seconds_1core:.0f}s single-core, ~{est_seconds_1core/4:.0f}s at parallelism 4")
    print(f"Total experiment time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
