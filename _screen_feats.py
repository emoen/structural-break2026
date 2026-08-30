"""NEW feature families for the ranker+rf re-screen (2026-08-23).

Ported from the ADIA-Lab 2025 4th-place writeup (Julian Mukaj), adapted to our REAL-TIME
unknown-boundary setting: everything is computed from the trailing window `w` and the history
context `ctx` (period-0/1 history) only — no known boundary, no future.

Each fn has the standard battery contract  (w, ctx) -> list[float]  (see _feat_lib), so it
plugs straight into the multiscale extra-column build (applied at each SCALE subwindow).

Families:
  cov    coefficient of variation  (the 4th-place "magic feature": unitless std/mean, AUC~0.53
         alone last year, gated 3 data-generation regimes). window CoV, log-ratio vs history CoV,
         history CoV, robust (mad/median) CoV.
  lz     Lempel-Ziv (LZ76) complexity of the median-binarized series: compressibility / structure
         of the window vs history (normalized-compression-distance flavour).
  mosum  moving-sum-of-squares local variance level: max local variance burst vs mean (volatility
         clustering / scale burst inside the window).
  ssa    singular-spectrum top-mode share of the trailing Hankel matrix vs history: subspace /
         dominant-mode reallocation across the split.
"""
import math

import numpy as np

EPS = 1e-12


# --- coefficient of variation ("magic feature") ----------------------------
def f_cov(w, ctx):
    w = np.asarray(w, dtype=np.float64)
    m = float(w.mean())
    s = float(w.std())
    cov_win = s / max(abs(m), EPS)
    cov_hist = ctx["sd"] / max(abs(ctx["mu"]), EPS)
    med = float(np.median(w))
    mad = float(np.median(np.abs(w - med))) * 1.4826
    cov_rob = mad / max(abs(med), EPS)
    return [cov_win, math.log((cov_win + EPS) / (cov_hist + EPS)), cov_hist, cov_rob]


# --- "magic" std^2 / mean (ADIA25 Hong&Gu: mul_cv_whole_std_whole = cv*std) ---
def f_magic(w, ctx):
    w = np.asarray(w, dtype=np.float64)
    m = float(w.mean())
    v = float(w.var())
    ms = m if abs(m) > EPS else (EPS if m >= 0 else -EPS)
    magic_win = v / ms                       # std^2 / mean, sign-preserving
    mh = ctx["mu"] if abs(ctx["mu"]) > EPS else EPS
    magic_hist = ctx["var"] / mh
    return [magic_win, magic_hist, magic_win - magic_hist,
            math.copysign(math.log1p(abs(magic_win)), magic_win)]


# --- Lempel-Ziv complexity --------------------------------------------------
def _lz76(b):
    """Normalized LZ76 complexity of a 0/1 sequence (Kaspar-Schuster)."""
    n = len(b)
    if n < 4:
        return 0.0
    i = 0
    c = 1
    l = 1
    k = 1
    k_max = 1
    while True:
        if b[i + k - 1] == b[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    # normalize to [0,1]-ish: c / (n / log2 n)  (binary alphabet)
    return float(c * math.log2(n) / n)


def f_lz(w, ctx):
    w = np.asarray(w, dtype=np.float64)
    thr = float(np.median(w))
    bw = (w > thr).astype(np.int8)
    lz_win = _lz76(bw)
    h = ctx.get("sample_ordered")
    if h is not None and len(h) >= 8:
        thr_h = float(np.median(h))
        lz_hist = _lz76((np.asarray(h) > thr_h).astype(np.int8))
    else:
        lz_hist = lz_win
    return [lz_win, lz_win - lz_hist]


# --- MOSUM local variance ---------------------------------------------------
def f_mosum(w, ctx, k=10):
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / max(ctx["sd"], 1e-8)
    sq = z * z
    n = len(sq)
    if n < 2 * k:
        return [0.0, 1.0]
    c = np.concatenate([[0.0], np.cumsum(sq)])
    blocks = (c[k:] - c[:-k]) / k                    # moving mean of squares
    mx = float(blocks.max())
    mn = float(blocks.mean())
    return [math.log(mx + EPS), mx / max(mn, EPS)]


# --- SSA top singular-mode share -------------------------------------------
def f_ssa(w, ctx, L=10):
    z = (np.asarray(w, dtype=np.float64) - ctx["mu"]) / max(ctx["sd"], 1e-8)
    n = len(z)
    if n < 2 * L:
        return [0.0, 0.0]
    H = np.lib.stride_tricks.sliding_window_view(z, L)   # (n-L+1, L)
    try:
        sv = np.linalg.svd(H, compute_uv=False)
    except np.linalg.LinAlgError:
        return [0.0, 0.0]
    tot = float((sv * sv).sum())
    share = float(sv[0] * sv[0] / max(tot, EPS))
    # crude history reference: top-mode share of an equal-length slice of history
    h = ctx.get("sample_ordered")
    share_h = share
    if h is not None and len(h) >= 2 * L:
        hz = (np.asarray(h, dtype=np.float64) - ctx["mu"]) / max(ctx["sd"], 1e-8)
        Hh = np.lib.stride_tricks.sliding_window_view(hz, L)
        try:
            svh = np.linalg.svd(Hh, compute_uv=False)
            share_h = float(svh[0] * svh[0] / max(float((svh * svh).sum()), EPS))
        except np.linalg.LinAlgError:
            pass
    return [share, share - share_h]


REG = {"cov": f_cov, "magic": f_magic, "lz": f_lz, "mosum": f_mosum, "ssa": f_ssa}
