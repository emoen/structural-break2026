"""BIG overnight EDA — mine the real train data for leads across all 3 axes.

Axes it feeds:
  A) calibration-mining : which base signals shift most at the boundary / which
     features would benefit from per-series studentization.
  1) synthetic-generator: online-length gap (real short-online vs synth min ~400)
     and process-manifold coverage (AR1 / var-ratio / tails).
  3) early-detection    : where the metric weight (n_pos*n_neg) concentrates
     across online-relative step, and how fast a naive detector could win there.

Read-only. Writes findings to _eda_night.txt (no .md). Deterministic.
"""
import json
import time
import numpy as np
import pandas as pd

T0 = time.time()
OUT = open("_eda_night.txt", "w")


def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    OUT.write(line + "\n")
    OUT.flush()


def q(x, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x, float)
    return {p: float(np.percentile(x, p)) for p in ps}


emit("=== BIG NIGHT EDA ", pd.Timestamp.now(), "===")
X = pd.read_parquet("data/X_train.parquet")
y = pd.read_parquet("data/y_train.parquet")
emit("X", X.shape, "y", y.shape)

# online index = the times that appear in y (per series). history = the rest.
y_ids = y.index.get_level_values(0).values
y_t = y.index.get_level_values(1).values
y_tg = y["target"].values
# first online time per id
ord_ = np.lexsort((y_t, y_ids))
y_ids, y_t, y_tg = y_ids[ord_], y_t[ord_], y_tg[ord_]
uids, first_pos = np.unique(y_ids, return_index=True)
online_start = dict(zip(uids, y_t[first_pos]))

val = X["value"].values
xi_id = X.index.get_level_values(0).values
xi_t = X.index.get_level_values(1).values
# X is already sorted by (id,time)
xord = np.lexsort((xi_t, xi_id))
val = val[xord]; xi_id = xi_id[xord]; xi_t = xi_t[xord]
xu, xstart = np.unique(xi_id, return_index=True)
xend = np.r_[xstart[1:], len(val)]
xslice = {sid: (s, e) for sid, s, e in zip(xu, xstart, xend)}

# y per-id slices
yu, ystart = np.unique(y_ids, return_index=True)
yend = np.r_[ystart[1:], len(y_ids)]
yslice = {sid: (s, e) for sid, s, e in zip(yu, ystart, yend)}

N = len(xu)
emit("n_series", N)

# ---- per-series records -------------------------------------------------
rec = {k: np.zeros(N) for k in
       ("hist_len", "on_len", "broke", "onset_rel", "pos_frac",
        "dmean", "dstd", "dmean_z", "ar1_h", "var_h", "var_ratio",
        "kurt_h", "abs_dmean", "abs_dstd")}
BW = 100  # boundary compare window


def ar1(a):
    a = a - a.mean()
    if len(a) < 20 or a.std() < 1e-12:
        return 0.0
    return float(np.dot(a[:-1], a[1:]) / (np.dot(a[:-1], a[:-1]) + 1e-12))


for i, sid in enumerate(xu):
    s, e = xslice[sid]
    v = val[s:e]
    t = xi_t[s:e]
    os_ = online_start.get(sid, None)
    if os_ is None:
        continue
    k = int(np.searchsorted(t, os_))     # history length
    xh = v[:k]; xo = v[k:]
    rec["hist_len"][i] = len(xh)
    rec["on_len"][i] = len(xo)
    ys, ye = yslice[sid]
    tg = y_tg[ys:ye]
    broke = tg.max() > 0
    rec["broke"][i] = broke
    rec["onset_rel"][i] = int(np.argmax(tg)) if broke else -1
    rec["pos_frac"][i] = float(tg.mean())
    hh = xh[-BW:] if len(xh) >= BW else xh
    oo = xo[:BW] if len(xo) >= BW else xo
    if len(hh) >= 10 and len(oo) >= 10:
        mh, mo = hh.mean(), oo.mean()
        sh, so = hh.std() + 1e-12, oo.std() + 1e-12
        rec["dmean"][i] = mo - mh
        rec["dstd"][i] = so - sh
        rec["dmean_z"][i] = (mo - mh) / sh
        rec["abs_dmean"][i] = abs(mo - mh) / sh
        rec["abs_dstd"][i] = abs(np.log(so / sh))
        rec["var_ratio"][i] = float(np.log((so ** 2) / (sh ** 2)))
    rec["ar1_h"][i] = ar1(xh)
    rec["var_h"][i] = float(np.var(xh))
    if len(xh) > 20:
        z = (xh - xh.mean()) / (xh.std() + 1e-12)
        rec["kurt_h"][i] = float((z ** 4).mean() - 3.0)
    if i % 2000 == 0:
        emit(f"  ...{i}/{N} ({time.time()-T0:.0f}s)")

brk = rec["broke"].astype(bool)
emit("\n### break rate:", round(brk.mean(), 4), " (n_break", int(brk.sum()), ")")

# ---- AXIS 1: online-length distribution (short-online gap) ---------------
emit("\n===== AXIS 1: ONLINE-LENGTH DISTRIBUTION (synth gap) =====")
ol = rec["on_len"]
emit("online_len pcts:", json.dumps({k: round(v, 1) for k, v in q(ol).items()}))
for thr in (20, 30, 50, 100, 200, 400):
    emit(f"  frac online_len < {thr}: {float((ol<thr).mean()):.4f}")
emit("history_len pcts:", json.dumps({k: round(v, 1) for k, v in q(rec['hist_len']).items()}))
emit("NOTE: synth generator min online ~400; real fraction below 400 is the gap.")

# ---- AXIS 3: per-online-step metric weight concentration -----------------
emit("\n===== AXIS 3: METRIC WEIGHT ACROSS ONLINE-RELATIVE STEP =====")
maxstep = int(min(1200, ol.max()))
# build per-step pos/neg counts using onset_rel + on_len
alive = np.zeros(maxstep); npos = np.zeros(maxstep); nneg = np.zeros(maxstep)
for i in range(N):
    L = int(rec["on_len"][i])
    if L <= 0:
        continue
    onset = rec["onset_rel"][i]
    up = min(L, maxstep)
    alive[:up] += 1
    if brk[i] and onset >= 0:
        for kk in range(up):
            if kk >= onset:
                npos[kk] += 1
            else:
                nneg[kk] += 1
    else:
        nneg[:up] += 1
weight = npos * nneg
wc = np.cumsum(weight) / (weight.sum() + 1e-12)
emit("total metric weight sum:", f"{weight.sum():.3e}")
for frac in (0.25, 0.5, 0.75, 0.9):
    step = int(np.searchsorted(wc, frac))
    emit(f"  {int(frac*100)}% of metric weight by online-step {step} "
         f"(alive={int(alive[min(step,maxstep-1)])}, "
         f"npos={int(npos[min(step,maxstep-1)])}, nneg={int(nneg[min(step,maxstep-1)])})")
emit("  weight by step buckets:")
for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, maxstep)]:
    hi = min(hi, maxstep)
    emit(f"    step[{lo:4d},{hi:4d}) weight_frac={weight[lo:hi].sum()/(weight.sum()+1e-12):.4f} "
         f"alive~{int(alive[min((lo+hi)//2,maxstep-1)])}")

# ---- break onset speed --------------------------------------------------
emit("\n===== BREAK ONSET (how fast breaks appear in online) =====")
onset = rec["onset_rel"][brk & (rec["onset_rel"] >= 0)]
emit("onset_rel (online idx of first labeled break) pcts:",
     json.dumps({k: round(v, 1) for k, v in q(onset).items()}))
for thr in (1, 5, 10, 25, 50):
    emit(f"  frac breaks onset within first {thr} online steps: {float((onset<thr).mean()):.4f}")
emit("pos_frac (labeled-1 share among broken series) pcts:",
     json.dumps({k: round(v, 3) for k, v in q(rec['pos_frac'][brk]).items()}))

# ---- AXIS A: boundary detectability by signal ---------------------------
emit("\n===== AXIS A: BOUNDARY SHIFT DETECTABILITY (broke vs not) =====")


def rank_auc(score, label):
    s = np.asarray(score, float); l = np.asarray(label, bool)
    if l.sum() == 0 or (~l).sum() == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    npos_ = l.sum()
    return float((ranks[l].sum() - npos_ * (npos_ + 1) / 2) / (npos_ * (~l).sum()))


for name in ("abs_dmean", "abs_dstd", "var_ratio", "dmean_z"):
    a = np.abs(rec[name]) if name in ("var_ratio", "dmean_z") else rec[name]
    good = np.isfinite(a)
    emit(f"  |{name}| single-signal series-level AUC(broke): "
         f"{rank_auc(a[good], brk[good]):.4f}")
emit("  (series-level = does the boundary shift predict *whether* it broke; "
     "cross-check vs champion's per-step task)")

# ---- process manifold (for synth realism) -------------------------------
emit("\n===== AXIS 1: PROCESS MANIFOLD (synth realism targets) =====")
for name in ("ar1_h", "kurt_h", "var_h"):
    a = rec[name]; a = a[np.isfinite(a)]
    emit(f"  {name} pcts:", json.dumps({k: round(v, 3) for k, v in q(a).items()}))
emit("  frac |ar1_h|>0.5:", round(float((np.abs(rec['ar1_h'])>0.5).mean()), 4),
     " frac kurt_h>3 (heavy tail):", round(float((rec['kurt_h']>3).mean()), 4))

# ---- correlations: which series props associate with breaking ----------
emit("\n===== CROSS-CUTS: broke-rate by online-length & by |mean-shift| =====")
for name, arr in [("on_len", ol), ("abs_dmean", rec["abs_dmean"]),
                  ("var_ratio", np.abs(rec["var_ratio"])), ("hist_len", rec["hist_len"])]:
    a = arr.copy(); good = np.isfinite(a)
    a = a[good]; b = brk[good]
    qs = np.quantile(a, [0, .2, .4, .6, .8, 1.0])
    line = []
    for j in range(5):
        m = (a >= qs[j]) & (a <= qs[j + 1] if j == 4 else a < qs[j + 1])
        line.append(f"[{qs[j]:.2f},{qs[j+1]:.2f}):{b[m].mean():.3f}")
    emit(f"  broke-rate by {name} quintile:", " ".join(line))

emit(f"\n=== DONE {time.time()-T0:.0f}s ===")
OUT.close()
