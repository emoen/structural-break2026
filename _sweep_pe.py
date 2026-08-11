"""Sweep PREDICT_EVERY to quantify the staleness vs. inference-cost trade-off.

Trains once, then evaluates TS-AUC and infer time for each PREDICT_EVERY, so we can
choose the value that maximizes score within the inference budget. E-step 1.
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import _exp_fast as ef

PE_GRID = [15, 5, 1]


def main():
    t0 = time.time()
    rng = np.random.default_rng(ef.SEED)
    yi = pd.read_parquet(os.path.join(ef.DATA, "y_train_index.parquet"))
    all_ids = yi.index.to_numpy().copy()
    rng.shuffle(all_ids)
    train_ids = np.sort(all_ids[: ef.N_TRAIN_SERIES])
    val_ids = np.sort(all_ids[ef.N_TRAIN_SERIES: ef.N_TRAIN_SERIES + ef.N_VAL_SERIES])
    keep = set(train_ids.tolist()) | set(val_ids.tolist())

    print("Loading train data...")
    X = pd.read_parquet(os.path.join(ef.DATA, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(ef.DATA, "y_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = y[y.index.get_level_values("id").isin(set(val_ids.tolist()))]
    print(f"  loaded in {time.time()-t0:.1f}s")

    Xtr, ytr = ef.build_training(X, yi, train_ids, rng)
    model = HistGradientBoostingClassifier(
        max_iter=ef.MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=ef.SEED,
    )
    model.fit(Xtr, ytr)
    print(f"Trained HGB ({model.n_iter_} trees) on {Xtr.shape[0]} rows\n")

    # Pre-split val series once (independent of PREDICT_EVERY).
    val = []
    for sid in val_ids:
        g = X.loc[sid]
        x_h, onl = ef.split_series(g)
        x_on = onl.to_numpy(dtype=np.float64)
        midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
        val.append((sid, x_h, x_on, midx))

    print(f"{'PE':>4} {'TS-AUC':>8} {'infer_s':>8} {'us/point':>9} {'calls':>8} {'est_full/12core_s':>18}")
    for pe in PE_GRID:
        ef.PREDICT_EVERY = pe
        ef._T["compute"] = 0.0
        ef._T["predict"] = 0.0
        ti = time.time()
        cols = []
        total_points = total_calls = 0
        for sid, x_h, x_on, midx in val:
            preds, ncalls = ef.stream_predict(model, x_h, x_on)
            total_points += len(x_on)
            total_calls += ncalls
            cols.append(pd.Series(preds, index=midx))
        infer_time = time.time() - ti
        preds = pd.concat(cols).to_frame("prediction")
        auc = ef.ts_auc_from(preds, y)
        pts_per_series = total_points / len(val_ids)
        est_full_1core = infer_time / total_points * (10000 * pts_per_series)
        print(f"{pe:>4} {auc:>8.4f} {infer_time:>8.1f} "
              f"{1e6*infer_time/max(total_points,1):>9.2f} {total_calls:>8} "
              f"{est_full_1core/12:>18.0f}")

    print(f"\nTotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
