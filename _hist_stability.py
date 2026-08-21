"""Phase-0 premise check: do the HISTORIES (period==1) actually contain internal structural breaks?
If histories are already one stable regime (by construction), Bai-Perron/BCP trimming is moot.
Coarse single-split sup-F (Chow) scan for a MEAN break and a VARIANCE break on each history;
report the fraction flagged at several thresholds and WHERE the split sits (recent?).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from _night_banks import _chow_supf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
vals = X["value"].to_numpy(np.float64); per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True); ends = np.append(start[1:], len(ids))
HI = {}
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; HI[int(u)] = vals[s:e][p == 1]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}

hist = [HI[s] for s in TAU if len(HI[s]) >= 120]
n = len(hist)
T = np.array([_chow_supf(h) for h in hist])       # (t_mean, |logVR|, pos)
tmean, lvr, pos = T[:, 0], T[:, 1], T[:, 2]
print(f"histories scanned: {n}  (len>=120)")
print(f"sup-F mean-break t:   median={np.median(tmean):.2f}  p90={np.percentile(tmean,90):.2f}  p99={np.percentile(tmean,99):.2f}")
print(f"max |log var-ratio|:  median={np.median(lvr):.2f}  p90={np.percentile(lvr,90):.2f}  p99={np.percentile(lvr,99):.2f}")
for thr in (3, 4, 5, 6):
    fr = float((tmean > thr).mean())
    # of those flagged, how many sit in the RECENT third of history?
    recent = float((pos[tmean > thr] > 0.66).mean()) if (tmean > thr).any() else 0.0
    print(f"  mean-break t>{thr}: {fr*100:5.1f}% of histories flagged   (of those, {recent*100:.0f}% split in recent 1/3)")
for thr in (0.7, 1.0, 1.5):
    fr = float((lvr > thr).mean())
    print(f"  var-break |logVR|>{thr}: {fr*100:5.1f}% of histories flagged")
