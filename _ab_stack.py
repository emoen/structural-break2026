"""Real held-out A/B: train the shared 5-member ensemble ONCE on data/X_train.parquet,
score dataset/X_test.reduced under the depearly / stack4-winner / stack5-centroid weight
vectors. stack4 and stack5 differ ONLY in blend weights, so one training gives an exact
apples-to-apples comparison. Leaderboard metric = per-time_online weighted rank-AUC.
"""
import os
import json
import time
import tempfile
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

os.makedirs("experiments/json", exist_ok=True)
LOG = "experiments/_ab_stack.log"


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


# ---- load stack5 notebook defs (train + feature fns), skipping crunch/pip/eval ----
import json as _json
nb = _json.load(open("streaming_detector_stack5.ipynb"))
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

train = g["train"]
compute_features_ms = g["compute_features_ms"]
hist_context = g["hist_context"]
WINDOW, MIN_N, HIST_CAP = int(g["WINDOW"]), int(g["MIN_N"]), int(g["HIST_CAP"])


def build_datasets(X, yi, want_tau):
    ds = []
    for sid in yi.index:
        s = X.loc[sid]
        x_h = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
        x_o = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
        if want_tau:
            ti = int(yi.loc[sid, "tau_index"])
            ds.append((sid, x_h, x_o, None if ti < 0 else ti))
        else:
            ds.append((sid, x_h, x_o))
    return ds


def member_pcts(saved, X, yi):
    """Per-member percentile predictions on the test online points (warmup rows -> 0)."""
    models, refs, members, cfg = saved["models"], saved["refs"], saved["members"], saved["config"]
    g["MS_SCALES"] = tuple(cfg["MS_SCALES"])
    names = [n for n, _ in members]
    out = {n: [] for n in names}
    y_rows = []
    for sid in yi.index:
        s = X.loc[sid]
        x_h = s.loc[s["period"] == 1, "value"].to_numpy(np.float64)
        x_o = s.loc[s["period"] == 2, "value"].to_numpy(np.float64)
        ctx = hist_context(x_h, HIST_CAP)
        L = len(x_o)
        feats, active_idx = [], []
        for t in range(L):
            n = min(t + 1, WINDOW)
            if n < MIN_N:
                continue
            lo = max(0, t - WINDOW + 1)
            feats.append(compute_features_ms(x_o[lo:t + 1], ctx))
            active_idx.append(t)
        F = np.asarray(feats, dtype=np.float64) if feats else np.zeros((0, 61))
        for name, kind in members:
            col = np.zeros(L)
            if len(F):
                if kind == "ranker":
                    raw = models[name].predict(F)
                else:
                    raw = models[name].predict_proba(F)[:, 1]
                ref = np.asarray(refs[name], dtype=np.float64)
                pct = np.searchsorted(ref, raw, side="right") / len(ref)
                col[np.asarray(active_idx)] = pct
            out[name].append(col)
        y_rows.append(sid)
    for n in names:
        out[n] = np.concatenate(out[n])
    return out, names


def ts_auc(pred, yt):
    m = yt.copy()
    m["prediction"] = pred
    m = m.reset_index()
    m["time_online"] = m.groupby("id").cumcount()
    num = den = 0.0
    for _, grp in m.groupby("time_online"):
        lab = grp["target"].to_numpy()
        npos, nneg = int(lab.sum()), int((1 - lab).sum())
        if npos == 0 or nneg == 0:
            continue
        w = float(npos * nneg)
        num += w * roc_auc_score(lab, grp["prediction"].to_numpy())
        den += w
    return num / den


def blend(pcts, weights):
    L = len(next(iter(pcts.values())))
    s = np.zeros(L)
    wsum = sum(weights.values())
    for n, w in weights.items():
        if w > 0:
            s += (w / wsum) * pcts[n]
    return s


def main():
    open(LOG, "w").close()
    log(f"=== AB STACK  {datetime.now():%F %T} ===")
    log("loading data ...")
    Xtr = pd.read_parquet("data/X_train.parquet")
    yitr = pd.read_parquet("data/y_train_index.parquet")
    Xte = pd.read_parquet("dataset/X_test.reduced.parquet")
    yite = pd.read_parquet("dataset/y_test_index.reduced.parquet")
    yte = pd.read_parquet("dataset/y_test.reduced.parquet")
    log(f"train series={len(yitr)}  test series={len(yite)}  test rows={len(yte)}")

    ds_train = build_datasets(Xtr, yitr, want_tau=True)
    with tempfile.TemporaryDirectory() as md:
        t0 = time.time()
        train(ds_train, md)
        log(f"trained 5 members in {time.time()-t0:.0f}s")
        saved = joblib.load(os.path.join(md, "model.joblib"))

    t0 = time.time()
    pcts, names = member_pcts(saved, Xte, yite)
    log(f"member percentile inference in {time.time()-t0:.0f}s; rows={len(pcts[names[0]])}")

    WSETS = {
        "depearly (hgb.40/ranker.60)": {"hgb": 0.40, "ranker": 0.60},
        "stack4  (rk.45/rf.40/xgb.10/cat.05)": {"ranker": 0.45, "rf": 0.40, "xgb": 0.10, "catboost": 0.05},
        "stack5  (rk.45/rf.37/xgb.11/cat.06/hgb.01)": {"ranker": 0.45, "rf": 0.37, "xgb": 0.11, "catboost": 0.06, "hgb": 0.01},
    }
    log("\n-- single-member TS-AUC (X_test.reduced) --")
    for n in names:
        log(f"   {n:<10} {ts_auc(pcts[n], yte):.4f}")

    log("\n-- blends TS-AUC (X_test.reduced, real held-out) --")
    res = {}
    base = None
    for name, w in WSETS.items():
        a = ts_auc(blend(pcts, w), yte)
        res[name] = a
        if base is None:
            base = a
        log(f"   {name:<44} {a:.4f}   d_vs_depearly {a-base:+.4f}")
    d = res["stack5  (rk.45/rf.37/xgb.11/cat.06/hgb.01)"] - res["stack4  (rk.45/rf.40/xgb.10/cat.05)"]
    log(f"\n   stack5 - stack4 = {d:+.4f}")
    json.dump({k: float(v) for k, v in res.items()}, open("experiments/json/_ab_stack.json", "w"), indent=2)
    log(f"\n=== DONE {datetime.now():%F %T} ===")


if __name__ == "__main__":
    main()
