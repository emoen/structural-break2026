"""Probe the ONLINE-segment length distribution (period==2) per series.
Grounds the 'we massively under-use the data' claim: how many candidate
online windows actually exist per series vs the 10 we currently sample.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
per = X.index.get_level_values("id")
period = X["period"].to_numpy()
ids = per.to_numpy()

# online length per series = count of period==2 rows
df = pd.DataFrame({"id": ids, "period": period})
onl = df[df["period"] == 2].groupby("id").size()
hist = df[df["period"] == 1].groupby("id").size()
total = df.groupby("id").size()

def stats(name, s):
    q = np.percentile(s.values, [0, 10, 25, 50, 75, 90, 100])
    print(f"{name:8s} n={len(s):6d}  min={q[0]:8.0f} p10={q[1]:8.0f} p25={q[2]:8.0f} "
          f"med={q[3]:8.0f} p75={q[4]:8.0f} p90={q[5]:8.0f} max={q[6]:8.0f} sum={s.sum():,.0f}")

stats("TOTAL", total)
stats("HIST", hist)
stats("ONLINE", onl)

MIN_N = 15
cand = (onl - MIN_N).clip(lower=0)
print()
print(f"Candidate online windows per series (L-MIN_N, MIN_N={MIN_N}):")
stats("CAND", cand)
print()
print(f"Currently sampled: 10 per series => {10*len(onl):,} training rows total")
print(f"Available (all candidate windows):   {int(cand.sum()):,} windows total")
print(f"Under-utilisation factor: {cand.sum()/(10*len(onl)):.1f}x")
