"""Experiment: supervised streaming break detector (sklearn HistGradientBoosting).

Shared feature extractor is used for both training-set construction and the
streaming validation, guaranteeing train/infer parity.
"""
import math
import os
import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew, kurtosis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

WINDOW = 75
MIN_N = 15
HIST_CAP = 400
SEED = 42

N_TRAIN_SERIES = 5000
N_VAL_SERIES = 800
STEPS_PER_SERIES = 10      # sampled online steps per train series
PREDICT_EVERY = 5          # inference throttle (hold score between)

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


def hist_context(x_h):
    n = len(x_h)
    mu = float(x_h.mean()) if n else 0.0
    var = float(x_h.var(ddof=1)) if n > 1 else 1.0
    var = max(var, 1e-12)
    sd = math.sqrt(var)
    sk = float(skew(x_h)) if n > 2 else 0.0
    ku = float(kurtosis(x_h)) if n > 3 else 0.0
    ac1 = _autocorr(x_h, 1)
    ac2 = _autocorr(x_h, 2)
    q75, q25 = np.percentile(x_h, [75, 25]) if n else (0.0, 0.0)
    iqr = float(q75 - q25)
    iqr = iqr if iqr > 1e-12 else 1e-12
    rng = float(x_h.max() - x_h.min()) if n else 1.0
    rng = rng if rng > 1e-12 else 1e-12
    sc = _spectral_centroid(x_h) if n > 4 else 0.0
    if n > HIST_CAP:
        stride = n // HIST_CAP
        sample = x_h[::stride][:HIST_CAP]
    else:
        sample = x_h
    return {
        "mu": mu, "sd": sd, "var": var, "skew": sk, "kurt": ku,
        "ac1": ac1, "ac2": ac2, "iqr": iqr, "rng": rng, "sc": sc,
        "sample": sample,
    }


def compute_features(w, ctx):
    """w: 1-D np.array trailing window. ctx: hist_context dict. -> feature list."""
    n = len(w)
    mean_w = float(w.mean())
    var_w = max(float(w.var()), 1e-12)
    se = ctx["sd"] / math.sqrt(n)
    mean_z = (mean_w - ctx["mu"]) / max(se, 1e-8)
    log_var_ratio = math.log(var_w / ctx["var"])
    ks = float(ks_2samp(ctx["sample"], w).statistic)
    skew_diff = float(skew(w)) - ctx["skew"]
    kurt_diff = float(kurtosis(w)) - ctx["kurt"]
    ac1_diff = _autocorr(w, 1) - ctx["ac1"]
    ac2_diff = _autocorr(w, 2) - ctx["ac2"]
    q75, q25 = np.percentile(w, [75, 25])
    iqr_w = max(float(q75 - q25), 1e-12)
    iqr_ratio = iqr_w / ctx["iqr"]
    dev = np.abs(w - ctx["mu"]) / ctx["sd"]
    tail2 = float((dev > 2.0).mean())
    tail3 = float((dev > 3.0).mean())
    centered = w - ctx["mu"]
    zero_cross = float((np.diff(np.sign(centered)) != 0).mean()) if n > 1 else 0.0
    idx = np.arange(n)
    slope = float(np.polyfit(idx, w, 1)[0])
    slope_norm = slope * n / ctx["sd"]
    spec_centroid_diff = _spectral_centroid(w) - ctx["sc"]
    rng_ratio = (float(w.max() - w.min()) if n else 0.0) / ctx["rng"]
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
        ctx = hist_context(x_h)
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


def stream_predict(model, x_h, x_on):
    ctx = hist_context(x_h)
    buf = deque()
    out = np.zeros(len(x_on))
    score = 0.0
    for idx, point in enumerate(x_on):
        if len(buf) == WINDOW:
            buf.popleft()
        buf.append(float(point))
        n = len(buf)
        if n < MIN_N:
            out[idx] = 0.0
            continue
        if idx % PREDICT_EVERY == 0:
            w = np.asarray(buf)
            feats = np.asarray(compute_features(w, ctx), dtype=np.float64).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
        out[idx] = score
    return out


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

    print("Building training set...")
    Xtr, ytr = build_training(X, yi, train_ids, rng)
    print(f"  train matrix {Xtr.shape}, positives {ytr.mean():.3f}")

    print("Training HGB...")
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=SEED,
    )
    model.fit(Xtr, ytr)

    print("Streaming validation...")
    cols = []
    for sid in val_ids:
        g = X.loc[sid]
        x_h, onl = split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
        cols.append(pd.Series(stream_predict(model, x_h, x_on), index=midx))
    preds = pd.concat(cols).to_frame("prediction")
    auc = ts_auc_from(preds, y)
    print(f"\nVAL TS-AUC (HGB, {len(val_ids)} series): {auc:.4f}")
    print(f"Total time {time.time()-t0:.1f}s")

    # Feature importances via permutation-free proxy: use model's training score.
    import joblib
    joblib.dump({"model": model, "config": {
        "WINDOW": WINDOW, "MIN_N": MIN_N, "HIST_CAP": HIST_CAP,
        "PREDICT_EVERY": PREDICT_EVERY,
    }}, os.path.join(HERE, "_hgb_model.joblib"))
    print("Saved _hgb_model.joblib")


if __name__ == "__main__":
    main()
