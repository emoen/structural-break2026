"""Project the full-train ONNX RF size WITHOUT fitting a full RF (cores are busy).
Anchor = the smoke measurement (11784 rows -> 15.0 MB ONNX). ONNX TreeEnsemble
size is ~linear in total node count, and RF leaf count ~ n_rows/min_samples_leaf,
so size scales ~linearly with the pointwise training-row count. Here we compute
the EXACT full-train row count that train() would build, then project.
"""
import numpy as np, pandas as pd

MIN_N = 15
STEPS_PER_SERIES = 30

# per-series online length (period==2) and hist length (period==1), one pass
df = pd.read_parquet("data/X_train.parquet", columns=["period"])
# index is the series id (MultiIndex or 'id' level); normalize to a flat id array
if df.index.nlevels > 1:
    ids = df.index.get_level_values(0).to_numpy()
else:
    ids = df.index.to_numpy()
period = df["period"].to_numpy()
del df

order = np.argsort(ids, kind="stable")
ids_s = ids[order]; per_s = period[order]
# boundaries of each series
bnd = np.flatnonzero(np.diff(ids_s)) + 1
starts = np.concatenate([[0], bnd])
ends = np.concatenate([bnd, [len(ids_s)]])

rows = 0
usable = 0
online_lens = []
for a, b in zip(starts, ends):
    seg = per_s[a:b]
    L = int((seg == 2).sum())      # online length
    hlen = int((seg == 1).sum())   # hist length
    if L <= MIN_N or hlen < 5:
        continue
    n_cand = L - MIN_N             # arange(MIN_N, L)
    rows += min(STEPS_PER_SERIES, n_cand)
    usable += 1
    online_lens.append(L)

anchor_rows, anchor_mb = 11784, 15.0
proj_mb = anchor_mb * rows / anchor_rows
print(f"usable series      : {usable}")
print(f"full pointwise rows: {rows}  ({rows/anchor_rows:.1f}x the smoke anchor)")
print(f"online len p50/p90 : {np.percentile(online_lens,50):.0f}/{np.percentile(online_lens,90):.0f}")
print(f"projected ONNX RF  : ~{proj_mb:.0f} MB (linear-in-rows from {anchor_mb} MB @ {anchor_rows} rows)")
print(f"  min_samples_leaf=20, 800 trees -> ~{800*2*rows/20/1e6:.1f}M nodes")
