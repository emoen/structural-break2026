"""End-to-end smoke + determinism check for streaming_detector_rankerrf.ipynb:
exec the notebook code cells, run train() (ranker + rf->ONNX) then infer() twice,
verify feature dim = 73 (61 champion + 12 ship), validity, determinism, latency.
"""
import os, time, tempfile, json as _json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd, joblib

nb = _json.load(open("streaming_detector_rankerrf.ipynb"))
g = {"__name__": "__nb__"}
SKIP = ("crunch setup-notebook", "%pip", "crunch_tools", "import crunch", "Local TS-AUC")
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if any(k in s for k in SKIP):
        continue
    exec("\n".join(l for l in s.splitlines() if not l.strip().startswith(("%", "!"))), g)

train = g["train"]; infer = g["infer"]
g["MAX_TRAIN_SERIES"] = 400

Xtr = pd.read_parquet("data/X_train.parquet"); yitr = pd.read_parquet("data/y_train_index.parquet")
ds = []
for sid in list(yitr.index[:400]):
    s = Xtr.loc[sid]
    xh = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
    xo = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
    ti = int(yitr.loc[sid, "tau_index"])
    ds.append((sid, xh, xo, None if ti < 0 else ti))

md = tempfile.mkdtemp(); t0 = time.time(); train(ds, md)
print(f"[train] {time.time()-t0:.0f}s")
saved = joblib.load(os.path.join(md, "model.joblib"))
nfeat = int(saved["models"]["ranker"].n_features_in_)
print(f"[saved] members={saved['members']} blend={saved['blend']}")
print(f"[saved] n_features={nfeat} (expected 73 = 61 champ + 12 ship)")
print(f"[saved] has rf_onnx={saved.get('rf_onnx') is not None} refs={list(saved['refs'])}")
assert nfeat == 73, f"FEATURE DIM MISMATCH: {nfeat} != 73"

Xte = pd.read_parquet("dataset/X_test.reduced.parquet")
yite = pd.read_parquet("dataset/y_test_index.reduced.parquet")
infer_ds, n_online = [], 0
for sid in list(yite.index):
    s = Xte.loc[sid]
    xh = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
    xo = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
    infer_ds.append((xh, xo)); n_online += len(xo)

def run_infer():
    gen = infer(infer_ds, md)
    next(gen)
    return np.asarray([next(gen) for _ in range(n_online)], dtype=np.float64)

t0 = time.time(); scores = run_infer(); elapsed = time.time() - t0
n_nan = int(np.isnan(scores).sum())
print(f"\n[infer] {len(scores)} scores (expected {n_online}); nan={n_nan}; "
      f"min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")
print(f"[infer] {elapsed:.1f}s total -> {elapsed/n_online*1000:.2f} ms/point")
assert n_nan == 0 and len(scores) == n_online, "SMOKE FAIL (validity)"

scores2 = run_infer()
maxdiff = float(np.max(np.abs(scores - scores2)))
print(f"[determinism] max|pass1-pass2| = {maxdiff:.2e}")
assert maxdiff == 0.0, "DETERMINISM FAIL"

print("\nSMOKE OK")
