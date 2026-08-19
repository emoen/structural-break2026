"""Functional end-to-end smoke for streaming_detector_depearly.ipynb.

There is no X_train locally, so this is NOT a leaderboard estimate -- it only proves the
deployed 61-col train()/infer() pipeline runs without shape/runtime errors on real series
and yields a sane TS-AUC. Trains and infers on the 100 reduced X_test series (optimistic).
Feature correctness is already proven exactly by _parity_depearly.py.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import roc_auc_score

nb = nbformat.read("streaming_detector_depearly.ipynb", as_version=4)
g = {"__name__": "__nb__"}
for c in nb.cells:
    if c.cell_type != "code":
        continue
    s = c.source
    if ("import crunch" in s or "crunch_tools" in s or "crunch." in s
            or s.lstrip().startswith("%") or "%pip" in s):
        continue
    exec(compile(s, "<nb>", "exec"), g)

train, infer = g["train"], g["infer"]

X = pd.read_parquet("dataset/X_test.reduced.parquet")
yi = pd.read_parquet("dataset/y_test_index.reduced.parquet")
yt = pd.read_parquet("dataset/y_test.reduced.parquet")

datasets_train = []   # (id, x_hist, x_online, tau_online_or_None)
datasets_infer = []   # (x_hist, x_online)
sids = list(yi.index)
for sid in sids:
    s = X.loc[sid]
    x_hist = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
    x_online = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
    ti = int(yi.loc[sid, "tau_index"])
    tau = None if ti < 0 else ti
    datasets_train.append((sid, x_hist, x_online, tau))
    datasets_infer.append((x_hist, x_online))

with tempfile.TemporaryDirectory() as md:
    train(datasets_train, md)
    gen = infer(datasets_infer, md)
    next(gen)  # readiness
    preds = []
    for (_xh, x_online), sid in zip(datasets_infer, sids):
        for _ in range(len(x_online)):
            preds.append(next(gen))

# align predictions to y_test (per-series online order == y_test time order)
rows = []
for (_xh, x_online), sid in zip(datasets_infer, sids):
    pass
flat_pred = np.asarray(preds, dtype=np.float64)
y_all = yt["target"].to_numpy(np.int64)
assert len(flat_pred) == len(y_all), (len(flat_pred), len(y_all))

# per-time_online weighted rank-AUC (leaderboard metric)
merged = yt.copy()
merged["prediction"] = flat_pred
merged = merged.reset_index()
merged["time_online"] = merged.groupby("id").cumcount()
num = den = 0.0
for t, grp in merged.groupby("time_online"):
    lab = grp["target"].to_numpy()
    npos, nneg = int(lab.sum()), int((1 - lab).sum())
    if npos == 0 or nneg == 0:
        continue
    w = float(npos * nneg)
    num += w * roc_auc_score(lab, grp["prediction"].to_numpy())
    den += w
print(f"\nrows={len(flat_pred)}  feat_dim=61  series={len(sids)}")
print(f"functional smoke TS-AUC (train==test, optimistic) = {num/den:.4f}")
print("SMOKE OK: 61-col train()/infer() ran end-to-end with no errors")
