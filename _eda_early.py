"""Early-detection cliff — champion detection AUC vs how many POST-break points are in the window.

The Part-A gap map scores CLEAN windows (break fully in the past) at ~0.95 AUC, yet real streaming
TS-AUC is ~0.62. The difference is the boundary regime: at each online step the window contains only
`post_len` points after tau. This sweeps post_len and reports pooled champion AUC (break vs no-break),
which is where the per-time-online metric actually lives.

Usage:  .venv/bin/python -u _eda_early.py
"""
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _ensemble_tune import make_hgb
from _exp_fast import HIST_CAP, hist_context
from _night_banks import champ_base_row, dep_feats, fit_champ_refs, fit_dep_refs
from _synth_gen import PARAM_FAMILIES, gen_break, gen_nobreak

warnings.filterwarnings("ignore")
WINDOW, HIST_N, K, SEED = 250, 1200, 150, 0
_DEP_KEEP = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
POST_LENS = [10, 20, 40, 80, 160, 250]


def champ61(w, ctx, cref, dref):
    base = champ_base_row(w, ctx, cref)
    df = dep_feats(w, ctx, dref)
    return np.concatenate([base, np.asarray([df[i] for i in _DEP_KEEP], dtype=np.float64)])


def series_with_post(rng, fam, post_len, broken):
    """Window = last 250 points containing `post_len` post-break points (if broken)."""
    pre_in_win = WINDOW - post_len
    if broken:
        pre, post = gen_break(rng, fam, HIST_N + pre_in_win, max(post_len, 1))
        seg = np.concatenate([pre, post])
        ref = pre
    else:
        seg = gen_nobreak(rng, fam, HIST_N + WINDOW)
        ref = seg[:HIST_N + pre_in_win]
    mu, sd = float(ref.mean()), float(ref.std())
    seg = (seg - mu) / (sd if sd > 1e-6 else 1.0)
    return seg[:HIST_N], seg[-WINDOW:]


def auc_at(post_len):
    rng = np.random.default_rng(SEED + post_len)
    X, y = [], []
    for fam in PARAM_FAMILIES:
        for broken in (True, False):
            for _ in range(K):
                x_h, w = series_with_post(rng, fam, post_len, broken)
                ctx = hist_context(x_h, HIST_CAP)
                X.append(champ61(w, ctx, fit_champ_refs(x_h), fit_dep_refs(x_h)))
                y.append(1 if broken else 0)
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = make_hgb()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    print(f"=== early-detection cliff | pooled all families | {K}/class/family ===\n", flush=True)
    print("  post_len = # post-break points in the 250-window (the boundary regime)\n", flush=True)
    for p in POST_LENS:
        a = auc_at(p)
        bar = "#" * int((a - 0.5) * 100)
        print(f"  post_len={p:>4}   champ_auc={a:.3f}  {bar}", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
