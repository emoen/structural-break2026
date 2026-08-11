"""Generate architecture figures from a real prediction run.

Outputs (into docs/):
  - prediction_vs_actual.png : one series, score vs actual step label
  - prediction.gif           : animated online reveal with live score
  - ts_auc_per_step.png      : per-step cross-sectional AUC + weights
Also prints a small step-by-step TS-AUC worked example.
"""
import os
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import _exp_fast as ef

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)

WINDOW, MIN_N, HIST_CAP, PREDICT_EVERY = ef.WINDOW, ef.MIN_N, ef.HIST_CAP, ef.PREDICT_EVERY
SEED = 42
N_TRAIN = 5000
N_VAL = 400


def stream(model, x_h, x_on, predict_every=PREDICT_EVERY):
    ctx = ef.hist_context(x_h, HIST_CAP)
    buf = deque()
    out = np.zeros(len(x_on))
    score = 0.0
    for idx, p in enumerate(x_on):
        if len(buf) == WINDOW:
            buf.popleft()
        buf.append(float(p))
        if len(buf) < MIN_N:
            out[idx] = 0.0
            continue
        if idx % predict_every == 0:
            feats = np.asarray(ef.compute_features(np.asarray(buf), ctx),
                               dtype=np.float64).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
        out[idx] = score
    return out


print("Loading data...")
yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
rng = np.random.default_rng(SEED)
all_ids = yi.index.to_numpy().copy()
rng.shuffle(all_ids)
train_ids = np.sort(all_ids[:N_TRAIN])
val_ids = np.sort(all_ids[N_TRAIN:N_TRAIN + N_VAL])
keep = set(train_ids.tolist()) | set(val_ids.tolist())

X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
X = X[X.index.get_level_values("id").isin(keep)]

print("Training model...")
Xtr, ytr = ef.build_training(X, yi, train_ids, np.random.default_rng(SEED))
model = HistGradientBoostingClassifier(
    max_iter=ef.MAX_ITER, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=True, random_state=SEED)
model.fit(Xtr, ytr)
print(f"  trained {model.n_iter_} trees on {Xtr.shape[0]} rows")


def tau_of(sid):
    t = int(yi.loc[sid, "tau_index"])
    return t if t >= 0 else None


# ---- one streaming pass over all val series (reused everywhere) ----
print("Predicting on validation series...")
val_rec = []
for vid in val_ids:
    tau_v = tau_of(vid)
    x_hv, onlv = ef.split_series(X.loc[vid])
    x_onv = onlv.to_numpy(dtype=np.float64)
    Lv = len(x_onv)
    sc = stream(model, x_hv, x_onv)
    lab = np.array([1 if (tau_v is not None and i >= tau_v) else 0 for i in range(Lv)])
    val_rec.append({"id": vid, "tau": tau_v, "x_h": x_hv, "x_on": x_onv,
                    "sc": sc, "lab": lab})


# ---- pick example: break series the model detects most cleanly ----
def margin(r):
    tau = r["tau"]
    L = len(r["x_on"])
    if tau is None or not (120 <= L <= 300 and 30 <= tau <= L - 30):
        return -1.0
    return float(r["sc"][tau:].mean() - r["sc"][:tau].mean())


best = max(val_rec, key=margin)
sid, tau, x_h, x_on, scores = best["id"], best["tau"], best["x_h"], best["x_on"], best["sc"]
L = len(x_on)
actual = best["lab"].astype(float)
print(f"Example series {sid}: len={L}, tau={tau}, "
      f"score jump {best['sc'][:tau].mean():.2f} -> {best['sc'][tau:].mean():.2f}")

# =====================================================================
# 1) Static: prediction vs actual
# =====================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
mu_h = x_h.mean()
ax1.axhline(mu_h, color="gray", ls=":", lw=1, label="historical mean")
ax1.plot(np.arange(L), x_on, color="#1f77b4", lw=1, label="online value")
ax1.axvline(tau, color="red", ls="--", lw=1.5, label=f"break tau={tau}")
ax1.set_ylabel("value")
ax1.set_title(f"Series {sid}: online segment (break at tau={tau})")
ax1.legend(loc="upper left", fontsize=8)

ax2.plot(np.arange(L), scores, color="#2ca02c", lw=1.6, label="predicted score s_t")
ax2.plot(np.arange(L), actual, color="red", ls="--", lw=1.2, label="actual label")
ax2.axvline(tau, color="red", ls="--", lw=1.5)
ax2.set_ylim(-0.05, 1.05)
ax2.set_ylabel("score / label")
ax2.set_xlabel("online step t")
ax2.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "prediction_vs_actual.png"), dpi=110)
plt.close(fig)
print("wrote docs/prediction_vs_actual.png")

# =====================================================================
# 2) Animated GIF: online reveal with live score
# =====================================================================
step = max(1, L // 120)
frames = list(range(0, L, step))
if frames[-1] != L - 1:
    frames.append(L - 1)

figA, (axa, axb) = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})
figA.tight_layout(pad=2.0)


def init():
    for ax in (axa, axb):
        ax.clear()
    axa.set_xlim(0, L)
    axa.set_ylim(x_on.min() - 0.5, x_on.max() + 0.5)
    axb.set_xlim(0, L)
    axb.set_ylim(-0.05, 1.05)
    return []


def draw(t):
    axa.clear()
    axb.clear()
    axa.set_xlim(0, L); axa.set_ylim(x_on.min() - 0.5, x_on.max() + 0.5)
    axb.set_xlim(0, L); axb.set_ylim(-0.05, 1.05)
    axa.axhline(mu_h, color="gray", ls=":", lw=1)
    axa.plot(np.arange(t + 1), x_on[:t + 1], color="#1f77b4", lw=1)
    axa.plot(t, x_on[t], "o", color="#1f77b4", ms=5)
    axb.plot(np.arange(t + 1), scores[:t + 1], color="#2ca02c", lw=1.6, label="prediction")
    axb.plot(np.arange(t + 1), actual[:t + 1], color="red", ls="--", lw=1.2, label="actual")
    if t >= tau:
        axa.axvline(tau, color="red", ls="--", lw=1.2)
        axb.axvline(tau, color="red", ls="--", lw=1.2)
    axa.set_ylabel("value")
    axb.set_ylabel("score")
    axb.set_xlabel("online step t")
    axa.set_title(f"t={t}   s_t={scores[t]:.2f}   actual={int(actual[t])}")
    axb.legend(loc="upper left", fontsize=8)
    return []


anim = FuncAnimation(figA, draw, frames=frames, init_func=init, blit=False)
anim.save(os.path.join(DOCS, "prediction.gif"), writer=PillowWriter(fps=15))
plt.close(figA)
print(f"wrote docs/prediction.gif ({len(frames)} frames)")

# =====================================================================
# 3) Per-step TS-AUC diagram over many series
# =====================================================================
max_len = max(len(r["sc"]) for r in val_rec)
auc_t, w_t, steps = [], [], []
for t in range(max_len):
    s = [r["sc"][t] for r in val_rec if t < len(r["sc"])]
    y = [r["lab"][t] for r in val_rec if t < len(r["sc"])]
    y = np.array(y); s = np.array(s)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        continue
    steps.append(t); auc_t.append(roc_auc_score(y, s)); w_t.append(npos * nneg)

auc_t = np.array(auc_t); w_t = np.array(w_t); steps = np.array(steps)
ts_auc = float((w_t * auc_t).sum() / w_t.sum())

figB, axc = plt.subplots(figsize=(8, 4))
axc.axhline(0.5, color="gray", ls=":", lw=1, label="random (0.5)")
axc.plot(steps, auc_t, color="#1f77b4", lw=1.4, label="per-step AUC(t)")
axc.axhline(ts_auc, color="#2ca02c", ls="--", lw=1.5,
            label=f"TS-AUC = {ts_auc:.3f}")
axc.set_ylabel("AUC(t)")
axc.set_xlabel("online step t")
axc.set_ylim(0.3, 1.0)
axc.set_title(f"Cross-sectional AUC per step ({len(val_rec)} val series)")
axt = axc.twinx()
axt.fill_between(steps, 0, w_t, color="orange", alpha=0.15)
axt.set_ylabel("weight w(t) = n_pos·n_neg", color="orange")
axt.tick_params(axis="y", labelcolor="orange")
lines1, labels1 = axc.get_legend_handles_labels()
axc.legend(lines1, labels1, loc="lower right", fontsize=8)
figB.tight_layout()
figB.savefig(os.path.join(DOCS, "ts_auc_per_step.png"), dpi=110)
plt.close(figB)
print(f"wrote docs/ts_auc_per_step.png  (TS-AUC={ts_auc:.4f} over {len(val_rec)} series)")

# =====================================================================
# 4) Step-by-step TS-AUC worked example at one online step
# =====================================================================
TSTAR = 50
rng2 = np.random.default_rng(7)
alive = [(r["id"], float(r["sc"][TSTAR]), int(r["lab"][TSTAR]))
         for r in val_rec if len(r["sc"]) > TSTAR]
pos = [a for a in alive if a[2] == 1]
neg = [a for a in alive if a[2] == 0]
rng2.shuffle(pos); rng2.shuffle(neg)
sample = pos[:2] + neg[:2]
rng2.shuffle(sample)

npos_s = sum(1 for a in sample if a[2] == 1)
nneg_s = sum(1 for a in sample if a[2] == 0)
conc = 0.0
for p in [a for a in sample if a[2] == 1]:
    for n in [a for a in sample if a[2] == 0]:
        conc += 1.0 if p[1] > n[1] else (0.5 if p[1] == n[1] else 0.0)
auc_sample = conc / (npos_s * nneg_s)

print(f"\n--- worked TS-AUC example at t={TSTAR} (real scores) ---")
for sid_s, sc_s, lab_s in sample:
    print(f"  series {sid_s:5d}  score={sc_s:.2f}  label={lab_s}")
print(f"  npos={npos_s} nneg={nneg_s}  concordant_pairs={conc}  "
      f"AUC(t={TSTAR})={auc_sample:.3f}")
if TSTAR in steps:
    idx_star = list(steps).index(TSTAR)
    print(f"  full AUC({TSTAR}) over all alive val series: "
          f"{auc_t[idx_star]:.3f}, weight={int(w_t[idx_star])}")
