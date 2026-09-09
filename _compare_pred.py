"""Compare two prediction parquets and report each one's local TS-AUC (same
metric as the leaderboard, on the local reduced test split). Usage:
    .venv/bin/python _compare_pred.py prediction_ser.parquet prediction_par.parquet
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

Y_TEST = "data/y_test.reduced.parquet"


def ts_auc(pred_path):
    prediction = pd.read_parquet(pred_path)
    y_test = pd.read_parquet(Y_TEST)
    merged = prediction.merge(y_test, how="left", left_index=True, right_index=True)
    merged["time_online"] = merged.groupby("id").cumcount()
    wsum = 0.0
    tot = 0.0
    for t, group in merged.groupby("time_online"):
        labels = group["target"].values
        scores = group["prediction"].values
        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        auc_t = float(roc_auc_score(labels, scores))
        w = float(n_pos * n_neg)
        wsum += w * auc_t
        tot += w
    return wsum / tot if tot > 0 else 0.5


def main(a, b):
    pa = pd.read_parquet(a)
    pb = pd.read_parquet(b)
    print(f"A={a}  shape={pa.shape}")
    print(f"B={b}  shape={pb.shape}")

    same_index = pa.index.equals(pb.index)
    print(f"index identical: {same_index}")
    va = pa["prediction"].to_numpy(np.float64)
    vb = pb["prediction"].to_numpy(np.float64)
    if va.shape == vb.shape:
        bit_identical = va.tobytes() == vb.tobytes()
        maxdiff = float(np.nanmax(np.abs(va - vb))) if va.size else 0.0
        nmis = int((va != vb).sum())
        print(f"predictions BIT-IDENTICAL: {bit_identical}  (max|diff|={maxdiff:.3e}, n_mismatch={nmis}/{va.size})")
    else:
        print(f"*** shape mismatch: {va.shape} vs {vb.shape}")

    auc_a = ts_auc(a)
    auc_b = ts_auc(b)
    print(f"\nlocal TS-AUC  A={auc_a:.6f}   B={auc_b:.6f}   dAUC={auc_b - auc_a:+.6f}")
    print("=> " + ("MATCH (score-neutral)" if abs(auc_b - auc_a) < 1e-9 else "DIFFERENT"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
