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
REAL_Y = os.path.join(HERE, "data", "y_train.parquet")

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
    if alpha + beta > 0.98:                 # enforce stationarity (else variance explodes -> inf)
        beta = 0.98 - alpha
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


def _perturb_ar2(rng, a, scale=0.25):
    """Stationary AR(2) that is a SMALL perturbation of `a` (subtle real-like coef change)."""
    for _ in range(50):
        b = [a[0] + rng.normal(0.0, scale), a[1] + rng.normal(0.0, scale)]
        if b[1] + b[0] < 0.98 and b[1] - b[0] < 0.98 and abs(b[1]) < 0.95:
            return b
    return a


# --- break vs no-break segment builders ------------------------------------
def gen_break(rng, fam, n_pre, n_post):
    """Return (pre[n_pre], post[n_post]) where post = pre-process mutated by `fam`."""
    if fam == "gauss_mean":
        sd = rng.uniform(0.8, 1.2)
        # real |mean_shift|: median ~0.05, p95 ~0.28 -> lognormal in std units
        shift = rng.choice([-1, 1]) * float(rng.lognormal(-3.0, 1.0))
        return _ar(rng, n_pre, [0.0], sd), _ar(rng, n_post, [0.0], sd) + shift * sd
    if fam == "gauss_var":
        sd = rng.uniform(0.8, 1.2)
        # real var_ratio: median ~1.0, p95 ~2.4 -> lognormal (sd multiplier = sqrt)
        r = float(np.exp(rng.normal(0.0, 0.20)))
        return _ar(rng, n_pre, [0.0], sd), _ar(rng, n_post, [0.0], sd * r)
    if fam == "ar_coef":
        a = _stationary_ar2(rng)
        b = _perturb_ar2(rng, a, 0.25)              # subtle coef change, not a fresh process
        return _ar(rng, n_pre, a, 1.0), _ar(rng, n_post, b, 1.0)
    if fam == "ar_var":
        a = _stationary_ar2(rng)
        r = float(np.exp(rng.normal(0.0, 0.20)))    # subtle innovation-variance change
        return _ar(rng, n_pre, a, 1.0), _ar(rng, n_post, a, r)
    if fam == "ma":
        q = rng.integers(1, 4)
        ta = rng.uniform(-0.8, 0.8, q)
        tb = ta + rng.normal(0.0, 0.25, q)          # subtle MA-coef change
        return _ma(rng, n_pre, ta, 1.0), _ma(rng, n_post, tb, 1.0)
    if fam == "arma":
        a = _stationary_ar2(rng)
        ta = rng.uniform(-0.7, 0.7, rng.integers(1, 3))
        b = _perturb_ar2(rng, a, 0.25)
        tb = ta + rng.normal(0.0, 0.25, len(ta))
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
        slope = rng.choice([-1, 1]) * rng.uniform(0.0002, 0.0015) * sd   # subtle drift onset
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


def load_real_lengths():
    """Extract per-series (H, O, onset) triples from the REAL labels for length-matching.

    H = historical length (first online time), O = online length, onset = tau_index within the
    online segment (first target==1), or -1 for no-break. Returns four arrays:
      real_H          all series' historical lengths
      nobrk_O         online lengths of no-break series
      brk_O, brk_ons  online length + onset of break series (kept as a JOINT pair, so a
                      sampled break reproduces a real (O, onset) exactly -> real short-online
                      series keep their genuinely-early onsets).
    """
    y = pd.read_parquet(REAL_Y, columns=["target"])
    tgt = y["target"].to_numpy(dtype=np.int8)
    tt = y.index.get_level_values("time").to_numpy(dtype=np.int64)
    ids = y.index.get_level_values("id").to_numpy()
    uniq, start = np.unique(ids, return_index=True)
    ends = np.append(start[1:], len(ids))
    H = tt[start].astype(np.int64)                 # first online time == historical length
    O = (ends - start).astype(np.int64)            # online point count
    onset = np.full(len(uniq), -1, dtype=np.int64)
    for k in range(len(uniq)):
        seg = tgt[start[k]:ends[k]]
        w = int(np.argmax(seg))
        if seg[w] == 1:
            onset[k] = w
    brk = onset >= 0
    return H, O[~brk], O[brk], onset[brk]


def gen_real_splice(rng, real, n_pre, n_post):
    a = real[rng.integers(len(real))]
    b = real[rng.integers(len(real))]
    if len(a) < n_pre + 5 or len(b) < n_post + 5:
        # fall back to a parametric family if a real slice is too short
        return gen_break(rng, "ar_coef", n_pre, n_post)
    i = rng.integers(0, len(a) - n_pre)
    j = rng.integers(0, len(b) - n_post)
    return a[i:i + n_pre], b[j:j + n_post]


def _heavy_tail_overlay(rng, seg):
    """Impose clustered stochastic volatility on a minority of series -> rare heavy tails.

    Real online segments have p95 kurtosis ~17 (vs ~2 for pure Gaussian families). A persistent
    log-vol envelope multiplies the whole series equally, so mean~0 and the pre/post variance
    ratio are preserved (standardisation by the pre-break ref renormalises the scale).
    """
    if rng.random() >= 0.20:                       # ~80% stay near-Gaussian
        return seg
    if rng.random() < 0.20:                        # ~4% overall: extreme fat tails (real p95 kurt ~17)
        s = rng.uniform(0.85, 1.30)
    else:
        s = rng.uniform(0.35, 0.85)                # log-vol amplitude -> tail heaviness
    rho = rng.uniform(0.90, 0.98)                  # volatility clustering
    n = len(seg)
    h = np.empty(n)
    h[0] = rng.normal(0.0, s)
    innov = rng.normal(0.0, s * np.sqrt(1.0 - rho * rho), n)
    for t in range(1, n):
        h[t] = rho * h[t - 1] + innov[t]
    return seg * np.exp(h - 0.5 * s * s)           # ~unit-mean-square envelope


# --- one series ------------------------------------------------------------
def generate_one(rng, real, break_rate, lengths=None):
    if lengths is None:
        H = int(rng.integers(1100, 4900))      # real H: p5~1188 p50~2970 p95~4808
        O = int(rng.integers(50, 1000))        # real O: p5~59 p50~506 p95~951 (short-online tail)
        has_break = rng.random() < break_rate
        tau_index = int(rng.integers(5, max(6, O - 5))) if has_break else -1
    else:
        # bootstrap real (H, O, onset) so synthetic length+onset dists match real exactly
        real_H, nobrk_O, brk_O, brk_ons = lengths
        H = int(real_H[rng.integers(len(real_H))])
        has_break = rng.random() < break_rate
        if has_break:
            j = rng.integers(len(brk_O))
            O, tau_index = int(brk_O[j]), int(brk_ons[j])
        else:
            O, tau_index = int(nobrk_O[rng.integers(len(nobrk_O))]), -1
    fams = PARAM_FAMILIES + (["real_splice"] * 3 if real is not None else [])
    fam = rng.choice(fams)
    if has_break:
        tau_index = int(np.clip(tau_index, 1, max(1, O - 1)))
        n_pre, n_post = H + tau_index, O - tau_index
        if fam == "real_splice":
            pre, post = gen_real_splice(rng, real, n_pre, n_post)
        else:
            pre, post = gen_break(rng, fam, n_pre, n_post)
        seg = np.concatenate([pre, post])
        ref_len = len(pre)
    else:
        tau_index = -1
        ref_len = H
        if fam == "real_splice":
            base = real[rng.integers(len(real))]
            if len(base) < H + O + 1:
                seg = gen_nobreak(rng, "ar_coef", H + O)
            else:
                s = rng.integers(0, len(base) - (H + O))
                seg = base[s:s + H + O]
        else:
            seg = gen_nobreak(rng, fam, H + O)
    seg = _heavy_tail_overlay(rng, np.asarray(seg, dtype=np.float64))
    ref = seg[:ref_len]
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
    ap.add_argument("--match-lengths", action="store_true",
                    help="bootstrap real (H,O,onset) triples so synthetic length+onset dists match "
                         "real exactly (covers the O<50 short-online tail the uniform range misses)")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    real = None if args.no_real_splice else load_real_arrays(args.real_series, args.seed)
    if real is not None:
        print(f"loaded {len(real)} real series for splicing", flush=True)
    lengths = None
    if args.match_lengths:
        lengths = load_real_lengths()
        rH, nO, bO, bOns = lengths
        print(f"length-match ON: {len(rH)} real series | O p5/p50/p95 = "
              f"{np.percentile(np.concatenate([nO, bO]), [5, 50, 95]).astype(int).tolist()} | "
              f"break onset p5/p50/p95 = {np.percentile(bOns, [5, 50, 95]).astype(int).tolist()}",
              flush=True)

    idv, tv, vv, pv = [], [], [], []
    y_id, y_t, y_tg = [], [], []
    idx = []
    fam_counts = {}
    for sid in range(args.n):
        H, O, tau_index, seg, fam = generate_one(rng, real, args.break_rate, lengths)
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
