"""Synthetic structural-break generator (P4) — unlimited labeled series for NN pretraining.

Writes the EXACT competition format into a separate folder (default synthetic/):
  X_train.parquet        (id,time) -> value(float32), period(int64)   1=historical 2=online
  y_train.parquet        (id,time) -> target(int64)                   online only; 1 iff t>=tau
  y_train_index.parquet  (id)      -> tau_index(int64), tau(int64)    -1/-1 if no break

A synthetic series = a HISTORICAL segment (period 1) + an ONLINE segment (period 2). For a
break series the process changes at tau_index within the online part; historical + pre-break
share process A, post-break is process B. The whole series is standardised by the pre-break
(historical+pre) mean/std so historical ~ N(0,1) and the break shows as the intended departure
(exactly how the champion standardises windows vs history).

Break families (anything goes):
  gauss_mean / gauss_var   level or scale shift (Gaussian)
  ar_coef / ar_var         AR(1-2) dependence or innovation-variance change
  ma / arma                MA / ARMA coefficient change
  garch                    GARCH(1,1) volatility-regime change (omega/alpha/beta)
  tdist                    distribution-shape change (Gaussian <-> Student-t heavy tails)
  skew                     symmetric <-> skewed innovations
  trend                    linear trend onset after tau
  spectral                 AR(2) oscillation frequency shift (change with ~unchanged variance)
  real_splice              glue a real series' pre-segment to a DIFFERENT real series' segment
No-break series draw ONE stationary process throughout (target all 0).

Usage:
  .venv/bin/python -u _synth_gen.py --n 4000 --out synthetic --seed 42
  .venv/bin/python -u _synth_gen.py --n 200 --out synthetic_smoke --no-real-splice
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_X = os.path.join(HERE, "data", "X_train.parquet")

# families that need no real data; real_splice is added separately when enabled
PARAM_FAMILIES = [
    "gauss_mean", "gauss_var", "ar_coef", "ar_var", "ma", "arma",
    "garch", "tdist", "skew", "trend", "spectral",
]


# --- primitive processes (numpy only) --------------------------------------
def _ar(rng, n, phi, sd, burn=200):
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    x = np.zeros(n + burn)
    e = rng.normal(0.0, sd, n + burn)
    for t in range(p, n + burn):
        x[t] = phi @ x[t - p:t][::-1] + e[t]
    return x[burn:]


def _ma(rng, n, theta, sd):
    theta = np.asarray(theta, dtype=float)
    e = rng.normal(0.0, sd, n + len(theta))
    return np.convolve(e, np.concatenate([[1.0], theta]), mode="valid")[:n]


def _arma(rng, n, phi, theta, sd, burn=200):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    p, q = len(phi), len(theta)
    e = rng.normal(0.0, sd, n + burn)
    u = np.convolve(e, np.concatenate([[1.0], theta]), mode="full")[:n + burn]  # MA part
    x = np.zeros(n + burn)
    for t in range(max(p, q), n + burn):
        x[t] = phi @ x[t - p:t][::-1] + u[t]
    return x[burn:]


def _garch(rng, n, omega, alpha, beta, burn=300):
    N = n + burn
    z = rng.standard_normal(N)
    s2 = np.empty(N)
    x = np.zeros(N)
    s2[0] = omega / max(1.0 - alpha - beta, 1e-2)
    for t in range(1, N):
        s2[t] = omega + alpha * x[t - 1] ** 2 + beta * s2[t - 1]
        x[t] = np.sqrt(s2[t]) * z[t]
    return x[burn:]


def _tdraw(rng, n, df, sd):
    return sd * rng.standard_t(df, n) / np.sqrt(df / (df - 2.0))  # unit-variance scaled


def _skewdraw(rng, n, a, sd):
    # skew via exponential-minus-mean mixed with gaussian; normalise to ~unit variance
    e = rng.standard_normal(n) + a * (rng.standard_exponential(n) - 1.0)
    return sd * (e - e.mean()) / (e.std() + 1e-9)


def _stationary_ar2(rng):
    """Sample a stationary AR(2) coefficient pair."""
    while True:
        phi1 = rng.uniform(-1.2, 1.2)
        phi2 = rng.uniform(-0.9, 0.6)
        if phi2 + phi1 < 0.98 and phi2 - phi1 < 0.98 and abs(phi2) < 0.95:
            return [phi1, phi2]


# --- break vs no-break segment builders ------------------------------------
def gen_break(rng, fam, n_pre, n_post):
    """Return (pre[n_pre], post[n_post]) where post = pre-process mutated by `fam`."""
    if fam == "gauss_mean":
        sd = rng.uniform(0.8, 1.2)
        shift = rng.uniform(0.4, 1.6) * rng.choice([-1, 1])
        return _ar(rng, n_pre, [0.0], sd), _ar(rng, n_post, [0.0], sd) + shift
    if fam == "gauss_var":
        r = rng.uniform(1.6, 3.5) ** rng.choice([-1, 1])
        return _ar(rng, n_pre, [0.0], 1.0), _ar(rng, n_post, [0.0], 1.0 * r)
    if fam == "ar_coef":
        a = _stationary_ar2(rng)
        b = _stationary_ar2(rng)
        return _ar(rng, n_pre, a, 1.0), _ar(rng, n_post, b, 1.0)
    if fam == "ar_var":
        a = _stationary_ar2(rng)
        r = rng.uniform(1.5, 3.0) ** rng.choice([-1, 1])
        return _ar(rng, n_pre, a, 1.0), _ar(rng, n_post, a, 1.0 * r)
    if fam == "ma":
        q = rng.integers(1, 4)
        ta = rng.uniform(-0.8, 0.8, q)
        tb = rng.uniform(-0.8, 0.8, q)
        return _ma(rng, n_pre, ta, 1.0), _ma(rng, n_post, tb, 1.0)
    if fam == "arma":
        a = _stationary_ar2(rng)
        ta = rng.uniform(-0.7, 0.7, rng.integers(1, 3))
        b = _stationary_ar2(rng)
        tb = rng.uniform(-0.7, 0.7, rng.integers(1, 3))
        return _arma(rng, n_pre, a, ta, 1.0), _arma(rng, n_post, b, tb, 1.0)
    if fam == "garch":
        pa = (rng.uniform(0.02, 0.1), rng.uniform(0.02, 0.15), rng.uniform(0.7, 0.9))
        pb = (rng.uniform(0.05, 0.3), rng.uniform(0.1, 0.35), rng.uniform(0.5, 0.85))
        return _garch(rng, n_pre, *pa), _garch(rng, n_post, *pb)
    if fam == "tdist":
        if rng.random() < 0.5:
            return _ar(rng, n_pre, [0.0], 1.0), _tdraw(rng, n_post, rng.uniform(3, 6), 1.0)
        return _tdraw(rng, n_pre, rng.uniform(3, 6), 1.0), _ar(rng, n_post, [0.0], 1.0)
    if fam == "skew":
        if rng.random() < 0.5:
            return _ar(rng, n_pre, [0.0], 1.0), _skewdraw(rng, n_post, rng.uniform(0.6, 1.5), 1.0)
        return _skewdraw(rng, n_pre, rng.uniform(0.6, 1.5), 1.0), _ar(rng, n_post, [0.0], 1.0)
    if fam == "trend":
        sd = rng.uniform(0.8, 1.2)
        slope = rng.uniform(0.004, 0.02) * rng.choice([-1, 1])
        post = _ar(rng, n_post, [0.0], sd) + slope * np.arange(n_post)
        return _ar(rng, n_pre, [0.0], sd), post
    if fam == "spectral":
        a = _stationary_ar2(rng)
        b = [-a[0], a[1]]  # flip oscillation frequency, keep magnitude structure
        return _ar(rng, n_pre, a, 1.0), _ar(rng, n_post, b, 1.0)
    raise ValueError(fam)


def gen_nobreak(rng, fam, n):
    """Homogeneous stationary series of length n (no break)."""
    if fam in ("gauss_mean", "gauss_var", "trend", "tdist", "skew"):
        return _ar(rng, n, [0.0], rng.uniform(0.8, 1.2))
    if fam in ("ar_coef", "ar_var", "spectral"):
        return _ar(rng, n, _stationary_ar2(rng), 1.0)
    if fam == "ma":
        return _ma(rng, n, rng.uniform(-0.8, 0.8, rng.integers(1, 4)), 1.0)
    if fam == "arma":
        return _arma(rng, n, _stationary_ar2(rng), rng.uniform(-0.7, 0.7, rng.integers(1, 3)), 1.0)
    if fam == "garch":
        return _garch(rng, n, rng.uniform(0.02, 0.1), rng.uniform(0.02, 0.15), rng.uniform(0.7, 0.9))
    raise ValueError(fam)


# --- real-series splice ----------------------------------------------------
def load_real_arrays(max_series, seed):
    """Return a list of per-series value arrays (historical+online) from real data."""
    X = pd.read_parquet(REAL_X, columns=["value"])
    vals = X["value"].to_numpy(dtype=np.float64)
    ids = X.index.get_level_values("id").to_numpy()
    # parquet is sorted by (id, time): first occurrence of each id gives contiguous bounds
    uniq, start = np.unique(ids, return_index=True)
    ends = np.append(start[1:], len(ids))
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(uniq), size=min(max_series, len(uniq)), replace=False)
    return [vals[start[i]:ends[i]] for i in sel]


def gen_real_splice(rng, real, n_pre, n_post):
    a = real[rng.integers(len(real))]
    b = real[rng.integers(len(real))]
    if len(a) < n_pre + 5 or len(b) < n_post + 5:
        # fall back to a parametric family if a real slice is too short
        return gen_break(rng, "ar_coef", n_pre, n_post)
    i = rng.integers(0, len(a) - n_pre)
    j = rng.integers(0, len(b) - n_post)
    return a[i:i + n_pre], b[j:j + n_post]


# --- one series ------------------------------------------------------------
def generate_one(rng, real, break_rate):
    H = int(rng.integers(1000, 2800))
    O = int(rng.integers(400, 1600))
    fams = PARAM_FAMILIES + (["real_splice"] * 3 if real is not None else [])
    fam = rng.choice(fams)
    has_break = rng.random() < break_rate
    if has_break:
        tau_index = int(rng.integers(40, max(41, O - 40)))
        n_pre, n_post = H + tau_index, O - tau_index
        if fam == "real_splice":
            pre, post = gen_real_splice(rng, real, n_pre, n_post)
        else:
            pre, post = gen_break(rng, fam, n_pre, n_post)
        seg = np.concatenate([pre, post])
        ref = pre
    else:
        tau_index = -1
        if fam == "real_splice":
            base = real[rng.integers(len(real))]
            if len(base) < H + O + 1:
                seg = gen_nobreak(rng, "ar_coef", H + O)
            else:
                s = rng.integers(0, len(base) - (H + O))
                seg = base[s:s + H + O]
        else:
            seg = gen_nobreak(rng, fam, H + O)
        ref = seg[:H]
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    seg = (seg - mu) / (sd if sd > 1e-6 else 1.0)
    return H, O, tau_index, seg.astype(np.float32), str(fam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", type=str, default="synthetic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--break-rate", type=float, default=0.5)
    ap.add_argument("--real-series", type=int, default=1500)
    ap.add_argument("--no-real-splice", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    real = None if args.no_real_splice else load_real_arrays(args.real_series, args.seed)
    if real is not None:
        print(f"loaded {len(real)} real series for splicing", flush=True)

    idv, tv, vv, pv = [], [], [], []
    y_id, y_t, y_tg = [], [], []
    idx = []
    fam_counts = {}
    for sid in range(args.n):
        H, O, tau_index, seg, fam = generate_one(rng, real, args.break_rate)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        T = H + O
        t_arr = np.arange(T, dtype=np.int64)
        idv.append(np.full(T, sid, dtype=np.int64))
        tv.append(t_arr)
        vv.append(seg)
        pv.append(np.where(t_arr < H, 1, 2).astype(np.int64))
        on_t = np.arange(H, H + O, dtype=np.int64)
        tgt = (np.arange(O) >= tau_index).astype(np.int64) if tau_index >= 0 \
            else np.zeros(O, dtype=np.int64)
        y_id.append(np.full(O, sid, dtype=np.int64))
        y_t.append(on_t)
        y_tg.append(tgt)
        idx.append((sid, tau_index, (H + tau_index) if tau_index >= 0 else -1))
        if (sid + 1) % 1000 == 0:
            print(f"  {sid + 1}/{args.n} series ({(time.time() - t0):.0f}s)", flush=True)

    X = pd.DataFrame(
        {"value": np.concatenate(vv), "period": np.concatenate(pv)},
        index=pd.MultiIndex.from_arrays(
            [np.concatenate(idv), np.concatenate(tv)], names=["id", "time"]),
    )
    y = pd.DataFrame(
        {"target": np.concatenate(y_tg)},
        index=pd.MultiIndex.from_arrays(
            [np.concatenate(y_id), np.concatenate(y_t)], names=["id", "time"]),
    )
    yi = pd.DataFrame(
        idx, columns=["id", "tau_index", "tau"]).set_index("id")

    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)
    X.to_parquet(os.path.join(out, "X_train.parquet"))
    y.to_parquet(os.path.join(out, "y_train.parquet"))
    yi.to_parquet(os.path.join(out, "y_train_index.parquet"))

    br = float((yi["tau_index"] >= 0).mean())
    print(f"\nwrote {args.n} series to {out}/ | X rows {len(X)} | break rate {br:.3f}", flush=True)
    print("family mix:", dict(sorted(fam_counts.items())), flush=True)
    print(f"=== DONE in {(time.time() - t0) / 60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
