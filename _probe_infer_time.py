"""Timing probe: how long does ONE streaming inference step cost, split into feature
computation vs each member's single-row predict? Explains the exit-124 timeout and sizes
the fix. Trains a quick model on a small subset (RF depth slightly underestimated vs full).
"""
import os, time, tempfile, json as _json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd, joblib

nb = _json.load(open("streaming_detector_stack4.ipynb"))
g = {"__name__": "__nb__"}
SKIP = ("crunch setup-notebook", "%pip", "crunch_tools", "import crunch", "Local TS-AUC")
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if any(k in s for k in SKIP):
        continue
    lines = [l for l in s.splitlines() if not l.strip().startswith(("%", "!"))]
    exec("\n".join(lines), g)

train = g["train"]; cfm = g["compute_features_ms"]; hist_context = g["hist_context"]
WINDOW, MIN_N, HIST_CAP = int(g["WINDOW"]), int(g["MIN_N"]), int(g["HIST_CAP"])

Xtr = pd.read_parquet("data/X_train.parquet")
yitr = pd.read_parquet("data/y_train_index.parquet")
NS = 600
ids = list(yitr.index[:NS])
ds = []
for sid in ids:
    s = Xtr.loc[sid]
    xh = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
    xo = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
    ti = int(yitr.loc[sid, "tau_index"])
    ds.append((sid, xh, xo, None if ti < 0 else ti))

md = tempfile.mkdtemp()
t0 = time.time(); train(ds, md); print(f"[quick-train {NS} series] {time.time()-t0:.0f}s")
saved = joblib.load(os.path.join(md, "model.joblib"))
models, refs = saved["models"], saved["refs"]
g["MS_SCALES"] = tuple(saved["config"]["MS_SCALES"])

# a realistic full-window buffer + ctx from one test series
Xte = pd.read_parquet("dataset/X_test.reduced.parquet")
yite = pd.read_parquet("dataset/y_test_index.reduced.parquet")
sid0 = list(yite.index)[0]
s0 = Xte.loc[sid0]
xh0 = s0.loc[s0["period"] == 1, "value"].to_numpy(np.float64)
xo0 = s0.loc[s0["period"] == 2, "value"].to_numpy(np.float64)
ctx = hist_context(xh0, HIST_CAP)
buf = xo0[:WINDOW] if len(xo0) >= WINDOW else xo0
feats = np.asarray(cfm(np.asarray(buf), ctx), dtype=np.float64).reshape(1, -1)

N = 400
def timeit(fn):
    fn(); t = time.time()
    for _ in range(N):
        fn()
    return (time.time() - t) / N * 1000.0  # ms

ms_feat = timeit(lambda: cfm(np.asarray(buf), ctx))
ms = {}
for name, kind in saved["members"]:
    m = models[name]
    if kind == "ranker":
        ms[name] = timeit(lambda m=m: m.predict(feats))
    else:
        ms[name] = timeit(lambda m=m: m.predict_proba(feats))

print("\n-- per streaming step (ms) --")
print(f"  feature compute (shared, paid once/point): {ms_feat:6.3f}")
for name in ("hgb", "ranker", "rf", "xgb", "catboost"):
    print(f"  predict {name:<9}: {ms[name]:6.3f}")

def total(members):
    return ms_feat + sum(ms[m] for m in members)

depearly = total(["hgb", "ranker"])
stack4 = total(["ranker", "rf", "xgb", "catboost"])
stack5 = total(["ranker", "rf", "xgb", "catboost", "hgb"])
print("\n-- per-point total (ms) --")
print(f"  depearly (2 mbr): {depearly:6.3f}")
print(f"  stack4   (4 mbr): {stack4:6.3f}   x{stack4/depearly:.2f} vs depearly")
print(f"  stack5   (5 mbr): {stack5:6.3f}   x{stack5/depearly:.2f} vs depearly")

REDUCED_ROWS = 50983
print("\n-- projected wall time for a full-test inference --")
for mult, lbl in [(1, "reduced 50,983 rows"), (10, "~10x hidden test"), (20, "~20x hidden test")]:
    rows = REDUCED_ROWS * mult
    print(f"  {lbl:<22}: depearly {depearly*rows/1000:7.0f}s  stack4 {stack4*rows/1000:7.0f}s  stack5 {stack5*rows/1000:7.0f}s")
import shutil; shutil.rmtree(md, ignore_errors=True)
