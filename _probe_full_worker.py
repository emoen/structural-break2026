"""Measure ABSOLUTE per-worker RSS for the full-size (~393 MB) RF ONNX session,
with the raw blob freed (as infer() now does). Feeds the safe INFER_PARALLELISM
choice for the 64 GB CPU machine. Random data matched to the real node count
(~265k rows -> ~393 MB, same slope as _probe_ort_mem.py).
"""
import os, gc, numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

os.environ.setdefault("OMP_NUM_THREADS", "1")
NF = 61
RNG = np.random.default_rng(0)


def rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    return 0.0


print("training full-size RF (800 trees, 265k rows) ...", flush=True)
X = RNG.standard_normal((265000, NF)).astype(np.float32)
y = (RNG.random(265000) < 0.27).astype(int)
rf = RandomForestClassifier(n_estimators=800, max_features=0.5,
                            min_samples_leaf=20, n_jobs=15, random_state=42)
rf.fit(X, y)
del X, y; gc.collect()
onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, NF]))],
                      options={type(rf): {"zipmap": False}}, target_opset=17)
blob = onx.SerializeToString()
del rf, onx; gc.collect()
mb = len(blob) / 1e6
print(f"ONNX size = {mb:.1f} MB", flush=True)

pid = os.fork()
if pid == 0:  # child = one infer worker
    import json
    base = rss_gb()
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(blob, sess_options=so, providers=["CPUExecutionProvider"])
    oname = sess.get_outputs()[1].name
    blob = None; gc.collect()  # mirror infer(): free the blob after session build
    row = RNG.standard_normal((1, NF)).astype(np.float32)
    for _ in range(500):
        sess.run([oname], {"input": row})
    worker = rss_gb()
    print(json.dumps({"base_gb": round(base, 3), "worker_rss_gb": round(worker, 3),
                      "session_only_gb": round(worker - base, 3)}))
    os._exit(0)
os.waitpid(pid, 0)

print("\n=== 64 GB CPU machine planning (16 vCPU) ===")
# use a realistic per-worker = session RSS + ~0.4 GB (other members + data slice + interpreter)
PER = None
