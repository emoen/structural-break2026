"""Grounded per-worker memory measurement for the rankerrf RF ONNX.
Caches the full-data feature matrix once, then for each n_estimators fits rf, exports ONNX,
and in a fresh child measures PEAK (ru_maxrss) and STEADY (post-inference) RSS, with the ORT
CPU mem-arena on and off. Picks _PER_WORKER_GB from the worst-case peak.
"""
import os, json as _json, tempfile, time, subprocess
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd

CACHE = "_rf_featmat.npz"
if os.path.exists(CACHE):
    d = np.load(CACHE); X, y = d["X"], d["y"]
    print(f"[rows] cached {X.shape}")
else:
    nb = _json.load(open("streaming_detector_rankerrf.ipynb"))
    g = {"__name__": "__nb__"}
    SKIP = ("crunch setup-notebook", "%pip", "crunch_tools", "import crunch", "Local TS-AUC",
            "def train", "def infer", "INFER_PARALLELISM")
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if any(k in s for k in SKIP):
            continue
        exec("\n".join(l for l in s.splitlines() if not l.strip().startswith(("%", "!"))), g)
    hist_context = g["hist_context"]; compute_features_ms = g["compute_features_ms"]
    WINDOW = g["WINDOW"]; HIST_CAP = g["HIST_CAP"]; MIN_N = g["MIN_N"]
    STEPS_PER_SERIES = g["STEPS_PER_SERIES"]; SEED = g["SEED"]
    Xtr = pd.read_parquet("data/X_train.parquet"); yitr = pd.read_parquet("data/y_train_index.parquet")
    rng = np.random.default_rng(SEED); rows, lab = [], []; t0 = time.time()
    for sid in yitr.index:
        s = Xtr.loc[sid]
        xh = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
        xo = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
        ti = int(yitr.loc[sid, "tau_index"]); tau = None if ti < 0 else ti
        L = len(xo)
        if L <= MIN_N or len(xh) < 5:
            continue
        ctx = hist_context(xh, HIST_CAP); cand = np.arange(MIN_N, L)
        for t in rng.choice(cand, size=min(STEPS_PER_SERIES, len(cand)), replace=False):
            lo = max(0, t - WINDOW + 1)
            rows.append(compute_features_ms(xo[lo:t + 1], ctx))
            lab.append(1 if (tau is not None and t >= tau) else 0)
    X = np.nan_to_num(np.asarray(rows, np.float64)); y = np.asarray(lab, np.int64)
    np.savez(CACHE, X=X, y=y)
    print(f"[rows] {X.shape} in {time.time()-t0:.0f}s (cached)")

from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

CHILD = r'''
import sys, resource, gc, numpy as np, onnxruntime as ort
path, arena, nfeat = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3])
blob = open(path, "rb").read()
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
so.enable_cpu_mem_arena = arena
sess = ort.InferenceSession(blob, sess_options=so, providers=["CPUExecutionProvider"])
del blob; gc.collect()
for _ in range(200):
    sess.run(None, {"input": np.random.randn(1, nfeat).astype(np.float32)})
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
rss = int(open("/proc/self/statm").read().split()[1]) * (resource.getpagesize()) / (1024**3)
print(f"{peak:.2f} {rss:.2f}")
'''
cf = os.path.join(tempfile.mkdtemp(), "child.py"); open(cf, "w").write(CHILD)

import gc
print(f"{'trees':>6} {'onnx_MB':>8} {'arena':>6} {'peak_GB':>8} {'steady_GB':>10} {'workers@64':>11}")
for trees in (500, 800):
    rf = RandomForestClassifier(n_estimators=trees, max_features=0.5, min_samples_leaf=20,
                                n_jobs=max(1, (os.cpu_count() or 4) - 1), random_state=42)
    rf.fit(X, y)
    nfeat = int(rf.n_features_in_)
    onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, nfeat]))],
                          options={type(rf): {"zipmap": False}}, target_opset=17)
    blob = onx.SerializeToString(); mb = len(blob) / 1e6
    p = os.path.join(tempfile.mkdtemp(), "rf.onnx"); open(p, "wb").write(blob)
    del rf, onx, blob; gc.collect()  # free parent before spawning the heavy child
    for arena in ("1", "0"):
        out = subprocess.check_output([".venv/bin/python", cf, p, arena, str(nfeat)]).decode().split()
        peak, steady = float(out[0]), float(out[1])
        w = max(1, int(64 * 0.90 / peak))
        print(f"{trees:>6} {mb:>8.1f} {('on' if arena=='1' else 'off'):>6} {peak:>8.2f} {steady:>10.2f} {w:>11}", flush=True)
