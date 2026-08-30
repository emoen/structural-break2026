"""Probe onnxruntime per-session RSS vs ONNX model size, arena ON vs OFF.
Uses random data with the REAL RF hyperparams (node count depends on n_samples /
min_samples_leaf, not feature semantics) to synthesize models of a few sizes,
then extrapolates to the observed 393.1 MB cloud model to pick a safe
INFER_PARALLELISM.
"""
import os, resource, numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

os.environ.setdefault("OMP_NUM_THREADS", "1")
NF = 61
RNG = np.random.default_rng(0)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB on Linux


def build(n_rows):
    X = RNG.standard_normal((n_rows, NF)).astype(np.float32)
    y = (RNG.random(n_rows) < 0.27).astype(int)
    rf = RandomForestClassifier(n_estimators=800, max_features=0.5,
                                min_samples_leaf=20, n_jobs=15, random_state=42)
    rf.fit(X, y)
    onx = convert_sklearn(rf, initial_types=[("input", FloatTensorType([None, NF]))],
                          options={type(rf): {"zipmap": False}}, target_opset=17)
    return onx.SerializeToString()


def sess_rss(blob, arena):
    """Create a session (in a child process) and return the RSS attributable to it."""
    pid = os.fork()
    if pid == 0:  # child
        base = rss_mb()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = arena
        so.enable_mem_pattern = arena
        s = ort.InferenceSession(blob, sess_options=so, providers=["CPUExecutionProvider"])
        oname = s.get_outputs()[1].name
        row = RNG.standard_normal((1, NF)).astype(np.float32)
        for _ in range(200):
            s.run([oname], {"input": row})
        delta = rss_mb() - base
        os._exit(min(255, int(delta / 8)))  # encode MB/8 in exit code (cap 255)
    _, status = os.waitpid(pid, 0)
    return (status >> 8) * 8  # decode MB


rows = [25000, 50000]
pts = []
for n in rows:
    blob = build(n)
    mb = len(blob) / 1e6
    on = sess_rss(blob, True)
    off = sess_rss(blob, False)
    pts.append((mb, on, off))
    print(f"rows={n:6d}  onnx={mb:6.1f}MB  session RSS: arena_ON~{on}MB  arena_OFF~{off}MB")

# linear extrapolation RSS = b*size (through points) to the 393.1MB cloud model
TARGET = 393.1
(s1, on1, off1), (s2, on2, off2) = pts
def extrap(y1, y2):
    b = (y2 - y1) / (s2 - s1)
    a = y1 - b * s1
    return a + b * TARGET
e_on, e_off = extrap(on1, on2), extrap(off1, off2)
print(f"\n>>> extrapolated per-session RSS at {TARGET}MB: arena_ON~{e_on:.0f}MB  arena_OFF~{e_off:.0f}MB")
for ram in (16, 32, 64):
    budget = ram * 1024 * 0.80  # leave 20% headroom
    print(f"    RAM {ram}GB (budget {budget:.0f}MB): safe workers arena_ON~{budget/e_on:.1f}  arena_OFF~{budget/e_off:.1f}")
