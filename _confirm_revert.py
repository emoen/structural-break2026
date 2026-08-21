"""Confirm the REVERTED depearly champion (clean binary-classifier head) runs end-to-end
and produces a sane local TS-AUC. Execs the notebook's definition cells (skipping crunch/
magic/test/submit), trains on a real train split, runs infer() per val series (parallel),
and computes the leaderboard TS-AUC on the val split.
"""
import os, sys, time, tempfile, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import nbformat
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN = int(os.environ.get("N_TRAIN", "6000"))
N_VAL = int(os.environ.get("N_VAL", "1500"))
N_JOBS = int(os.environ.get("N_JOBS", "15"))

# --- exec notebook definition cells (mirror _parity_depearly) ---
nb = nbformat.read(os.path.join(HERE, "streaming_detector_depearly.ipynb"), as_version=4)
g = {"__name__": "__nbmod__"}
for c in nb.cells:
    if c.cell_type != "code":
        continue
    s = c.source
    if ("import crunch" in s or "crunch_tools" in s or "crunch." in s
            or s.lstrip().startswith("%") or "%pip" in s or "!crunch" in s):
        continue
    if "def train(" in s or "def infer(" in s or "compute_features_ms" in s or True:
        # keep all non-crunch code cells (defs + config); harmless top-level prints ok
        try:
            exec(compile(s, "<nb>", "exec"), g)
        except Exception as e:
            print(f"[warn] skipped a cell on exec: {type(e).__name__}: {e}")

train_fn = g["train"]; infer_fn = g["infer"]
print(f"config WINDOW={g['WINDOW']} STEPS_PER_SERIES={g['STEPS_PER_SERIES']} "
      f"DENSE_K={g['DENSE_K']} BLEND_W={g['BLEND_W']} MAX_ITER={g['MAX_ITER']}")

# --- load real data, split by series ---
X = pd.read_parquet(os.path.join(HERE, "data", "X_train.parquet"))
yi = pd.read_parquet(os.path.join(HERE, "data", "y_train_index.parquet"))
vals = X["value"].to_numpy(np.float64); per = X["period"].to_numpy(np.int64)
ids = X.index.get_level_values("id").to_numpy()
uniq, start = np.unique(ids, return_index=True); ends = np.append(start[1:], len(ids))
HI, ON = {}, {}
for u, s, e in zip(uniq, start, ends):
    p = per[s:e]; v = vals[s:e]; HI[int(u)] = v[p == 1]; ON[int(u)] = v[p == 2]
TAU = {int(k): int(v) for k, v in yi["tau_index"].items()}

rng = np.random.default_rng(0)
series = [s for s in ON if len(ON[s]) > g["MIN_N"] and len(HI[s]) >= 5]
rng.shuffle(series)
tr = series[:N_TRAIN]
va = series[N_TRAIN:N_TRAIN + N_VAL]
print(f"train series={len(tr)}  val series={len(va)}")

# --- train ---
tmpdir = tempfile.mkdtemp(prefix="revert_confirm_")
train_ds = [(s, HI[s].tolist(), ON[s].tolist(), (TAU[s] if TAU[s] >= 0 else None)) for s in tr]
t0 = time.time()
train_fn(train_ds, tmpdir)
print(f"train done in {time.time()-t0:.0f}s -> {tmpdir}")

# --- infer per val series (parallel); each worker reloads the model once ---
def score_series(s):
    x_h = HI[s]; x_on = ON[s]
    gen = infer_fn([(x_h.tolist(), x_on.tolist())], tmpdir)
    next(gen)  # readiness yield
    scores = np.fromiter((float(v) for v in gen), dtype=np.float64)
    return s, scores

t0 = time.time()
results = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(score_series)(s) for s in va)
print(f"infer done in {time.time()-t0:.0f}s")

# --- TS-AUC (leaderboard metric) on val ---
rows = []
for s, sc in results:
    tau = TAU[s]; L = len(sc)
    for t in range(L):
        lab = 1 if (tau >= 0 and t >= tau) else 0
        rows.append((t, lab, sc[t]))
df = pd.DataFrame(rows, columns=["t", "label", "score"])
wsum = 0.0; tot = 0.0
for t, grp in df.groupby("t"):
    lab = grp["label"].values
    npos = int(lab.sum()); nneg = int((1 - lab).sum())
    if npos == 0 or nneg == 0:
        continue
    a = roc_auc_score(lab, grp["score"].values)
    w = float(npos * nneg); wsum += w * a; tot += w
print(f"\nLocal VAL TS-AUC (reverted classifier head) = {wsum/tot:.4f}  "
      f"[{len(va)} val series, trained on {len(tr)}]")
