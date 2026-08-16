"""Validate a synthetic dataset folder against the real competition format + harness."""
import os
import sys

import numpy as np
import pandas as pd

from _exp_fast import split_series

D = sys.argv[1] if len(sys.argv) > 1 else "synthetic_smoke"
X = pd.read_parquet(os.path.join(D, "X_train.parquet"))
y = pd.read_parquet(os.path.join(D, "y_train.parquet"))
yi = pd.read_parquet(os.path.join(D, "y_train_index.parquet"))

print("X ", X.index.names, list(X.columns), dict(X.dtypes))
print("y ", y.index.names, list(y.columns), dict(y.dtypes))
print("yi", yi.index.names, list(yi.columns), dict(yi.dtypes))
print("series:", yi.shape[0], "| break rate:", round(float((yi["tau_index"] >= 0).mean()), 3))
print("period values:", sorted(X["period"].unique().tolist()),
      "| value dtype:", X["value"].dtype)

# check a break series: split_series must match H/O, target aligns with tau_index
sid = int(yi[yi["tau_index"] >= 0].index[0])
xh, on = split_series(X.loc[sid])
tau_index = int(yi.loc[sid, "tau_index"])
tgt = y.loc[sid]["target"].to_numpy()
assert len(tgt) == len(on), "target length != online length"
assert int(tgt.sum()) == len(on) - tau_index, "target sum != post-break count"
assert int((tgt > 0).argmax()) == tau_index, "first positive != tau_index"
assert int(on.index.min()) == len(xh), "online start time != historical length"
print(f"break series {sid}: H={len(xh)} O={len(on)} tau_index={tau_index} -> OK")

# check a no-break series (if any): all targets 0
nb = yi[yi["tau_index"] < 0]
if len(nb):
    sid0 = int(nb.index[0])
    assert int(y.loc[sid0]["target"].sum()) == 0, "no-break series has positive targets"
    print(f"no-break series {sid0}: all targets 0 -> OK")
print("VALID: matches competition format + split_series semantics")
