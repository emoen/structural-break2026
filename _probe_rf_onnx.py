"""Prototype: accelerate the 800-tree RF single-row streaming predict via ONNX.
Trains a quick RF (subset), converts with skl2onnx, times single-row predict_proba
sklearn-vs-ONNX (single-thread, as in crunch infer), and checks probability parity.
"""
import os, time, tempfile, json as _json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

nb = _json.load(open("streaming_detector_stack4.ipynb"))
g = {"__name__": "__nb__"}
SKIP = ("crunch setup-notebook", "%pip", "crunch_tools", "import crunch", "Local TS-AUC")
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if any(k in s for k in SKIP):
        continue
    exec("\n".join(l for l in s.splitlines() if not l.strip().startswith(("%", "!"))), g)

train = g["train"]; cfm = g["compute_features_ms"]; hist_context = g["hist_context"]
WINDOW, MIN_N, HIST_CAP = int(g["WINDOW"]), int(g["MIN_N"]), int(g["HIST_CAP"])

Xtr = pd.read_parquet("data/X_train.parquet"); yitr = pd.read_parquet("data/y_train_index.parquet")
NS = 600
ds = []
for sid in list(yitr.index[:NS]):
    s = Xtr.loc[sid]
    xh = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
    xo = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
    ti = int(yitr.loc[sid, "tau_index"])
    ds.append((sid, xh, xo, None if ti < 0 else ti))

md = tempfile.mkdtemp(); t0 = time.time(); train(ds, md)
import joblib
saved = joblib.load(os.path.join(md, "model.joblib"))
g["MS_SCALES"] = tuple(saved["config"]["MS_SCALES"])
rf = saved["models"]["rf"]
print(f"[quick-train] {time.time()-t0:.0f}s; rf n_estimators={rf.n_estimators}")

# ---- convert RF -> ONNX (raw probability array, no zipmap) ----
onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, rf.n_features_in_]))],
                      options={type(rf): {"zipmap": False}}, target_opset=17)
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
sess = ort.InferenceSession(onx.SerializeToString(), so, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
prob_name = out_names[1]  # [label, probabilities]
onnx_bytes = onx.SerializeToString()
print(f"[onnx] outputs={out_names}  model_size={len(onnx_bytes)/1e6:.1f} MB")

# realistic single feature row
Xte = pd.read_parquet("dataset/X_test.reduced.parquet"); yite = pd.read_parquet("dataset/y_test_index.reduced.parquet")
s0 = Xte.loc[list(yite.index)[0]]
xo0 = s0.loc[s0["period"] == 2, "value"].to_numpy(np.float64)
ctx = hist_context(s0.loc[s0["period"] == 1, "value"].to_numpy(np.float64), HIST_CAP)
row64 = np.asarray(cfm(np.asarray(xo0[:WINDOW]), ctx), dtype=np.float64).reshape(1, -1)
row32 = row64.astype(np.float32)

N = 400
def timeit(fn):
    fn(); t = time.time()
    for _ in range(N): fn()
    return (time.time() - t) / N * 1000.0

ms_skl = timeit(lambda: rf.predict_proba(row64)[:, 1])
ms_onx = timeit(lambda: sess.run([prob_name], {"input": row32})[0][:, 1])

# parity over many rows
feats = []
for t in range(MIN_N, min(len(xo0), MIN_N + 500)):
    lo = max(0, t - WINDOW + 1)
    feats.append(cfm(np.asarray(xo0[lo:t + 1]), ctx))
F64 = np.asarray(feats, dtype=np.float64); F32 = F64.astype(np.float32)
p_skl = rf.predict_proba(F64)[:, 1]
p_onx = sess.run([prob_name], {"input": F32})[0][:, 1]
maxd = float(np.max(np.abs(p_skl - p_onx)))
rank_corr = float(np.corrcoef(p_skl.argsort().argsort(), p_onx.argsort().argsort())[0, 1])

print("\n-- single-row RF predict (single-thread) --")
print(f"  sklearn : {ms_skl:7.3f} ms/row")
print(f"  onnx    : {ms_onx:7.3f} ms/row   speedup x{ms_skl/ms_onx:.1f}")
print(f"\n-- parity over {len(p_skl)} rows --")
print(f"  max |prob diff|      : {maxd:.2e}")
print(f"  rank correlation     : {rank_corr:.6f}")
print(f"\n-- projected stack4 per-point total (feat1.0 + rk0.59 + xgb0.25 + cat0.23 + rf) --")
base = 1.04 + 0.59 + 0.25 + 0.23
print(f"  with sklearn rf : {base+ms_skl:6.2f} ms/point")
print(f"  with onnx rf    : {base+ms_onx:6.2f} ms/point")
import shutil; shutil.rmtree(md, ignore_errors=True)
