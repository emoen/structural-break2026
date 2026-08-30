"""Does shrinking the RF cut the onnxruntime per-worker BUILD peak, or is the ~11 GB
peak a fixed ORT overhead? Reuse the cached full-data feature matrix; fit several RF
shapes, export ONNX, and measure a fresh child's PEAK (ru_maxrss) + STEADY RSS.
"""
import os, tempfile, subprocess, gc
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

d = np.load("_rf_featmat.npz"); X, y = d["X"], d["y"]
print(f"[rows] {X.shape}")

from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

CHILD = r'''
import sys, resource, gc, numpy as np, onnxruntime as ort
path, nfeat = sys.argv[1], int(sys.argv[2])
blob = open(path, "rb").read()
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
sess = ort.InferenceSession(blob, sess_options=so, providers=["CPUExecutionProvider"])
del blob; gc.collect()
for _ in range(100):
    sess.run(None, {"input": np.random.randn(1, nfeat).astype(np.float32)})
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
rss = int(open("/proc/self/statm").read().split()[1]) * resource.getpagesize() / (1024**3)
print(f"{peak:.2f} {rss:.2f}")
'''
cf = os.path.join(tempfile.mkdtemp(), "c.py"); open(cf, "w").write(CHILD)

# (n_estimators, max_depth, min_samples_leaf, label)
CONFIGS = [
    (800, None, 20, "ship 800/none/20"),
    (800, 14,   20, "depth<=14"),
    (800, 10,   20, "depth<=10"),
    (800, None, 100, "leaf>=100"),
    (300, 12,   50, "small 300/12/50"),
]
print(f"{'config':>18} {'nodes/tree':>11} {'onnx_MB':>8} {'peak_GB':>8} {'steady_GB':>10} {'w@64':>5}")
for trees, depth, leaf, label in CONFIGS:
    rf = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                                max_features=0.5, n_jobs=max(1, (os.cpu_count() or 4) - 1),
                                random_state=42)
    rf.fit(X, y)
    nfeat = int(rf.n_features_in_)
    npt = int(np.mean([t.tree_.node_count for t in rf.estimators_]))
    onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, nfeat]))],
                          options={type(rf): {"zipmap": False}}, target_opset=17)
    blob = onx.SerializeToString(); mb = len(blob) / 1e6
    p = os.path.join(tempfile.mkdtemp(), "rf.onnx"); open(p, "wb").write(blob)
    del rf, onx, blob; gc.collect()
    out = subprocess.check_output([".venv/bin/python", cf, p, str(nfeat)]).decode().split()
    peak, steady = float(out[0]), float(out[1])
    w = max(1, int(64 * 0.90 / peak))
    print(f"{label:>18} {npt:>11} {mb:>8.1f} {peak:>8.2f} {steady:>10.2f} {w:>5}", flush=True)
