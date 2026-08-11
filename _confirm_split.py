"""Confirm base vs base+split at PREDICT_EVERY=1 (submission setting)."""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import _exp_fast as ef
import _sweep_feats as sf

sf.PE = 1
N_TRAIN, N_VAL = 6000, 400


def main():
    t0 = time.time()
    rng0 = np.random.default_rng(ef.SEED)
    yi = pd.read_parquet(os.path.join(ef.DATA, "y_train_index.parquet"))
    ids = yi.index.to_numpy().copy()
    rng0.shuffle(ids)
    tr = np.sort(ids[:N_TRAIN])
    vl = np.sort(ids[N_TRAIN:N_TRAIN + N_VAL])
    keep = set(tr.tolist()) | set(vl.tolist())
    X = pd.read_parquet(os.path.join(ef.DATA, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(ef.DATA, "y_train.parquet"))
    X = X[X.index.get_level_values("id").isin(keep)]
    y = y[y.index.get_level_values("id").isin(set(vl.tolist()))]
    val = []
    for sid in vl:
        g = X.loc[sid]
        xh, onl = ef.split_series(g)
        xo = onl.to_numpy(np.float64)
        midx = pd.MultiIndex.from_product([[sid], onl.index], names=["id", "time"])
        val.append((sid, xh, xo, midx))

    for name, us, up in [("base", False, False), ("base+split", False, True)]:
        fn = sf.make_feature_fn(us, up)
        rng = np.random.default_rng(ef.SEED)
        Xtr, ytr = sf.build(X, yi, tr, fn, rng)
        m = HistGradientBoostingClassifier(
            max_iter=ef.MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=ef.SEED,
        )
        m.fit(Xtr, ytr)
        ti = time.time()
        auc = sf.evaluate(m, val, y, fn)
        print(f"{name:>12} nfeat={Xtr.shape[1]} TS-AUC={auc:.4f} infer={time.time()-ti:.0f}s")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
