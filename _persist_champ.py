"""Port the EWMA temporal-smoothing lever to the CURRENT 61-col champion (b_no_accf) under the
real 0.6/0.4 HGB+ranker blend + TS-AUC. Mirrors _residual_overnight._eval EXACTLY, but intercepts
the per-step blend trajectory and applies a per-series CAUSAL EWMA (adjust=False = the streaming-
deployable recursive form) before scoring TS-AUC. Multi-seed; paired-t vs raw (no smoothing).

Base16 screen (_persist.py) found ewma10 +0.0040 t4.98 (10-fold). Question: does it survive on the
sharper 61-col champion, across seeds? Ship gate mirror: pos_all & mean_d>=0.0015 & min_t>=2.0.
"""
import os, time
from datetime import datetime
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from _night_stack import SEARCH_SEEDS, NCHAMP_BASE, prepare_seed, paired
from _night_banks import NDEP
from _residual_overnight import _score, pct, TUNED
from _ensemble_tune import make_hgb
from _lambdarank import fit_ranker, sort_into_groups
from _exp_fast import ts_auc_from

C = NCHAMP_BASE
DEP_KEEP = [C + i for i in (0, 1, 2, 3, 4, 5, 10, 11, 12, 13)]
BASE_COLS = np.concatenate([np.arange(C), np.asarray(DEP_KEEP, dtype=int)])   # 61 = b_no_accf

# variant -> ("ewma", alpha) full smoothing | ("half", alpha) 0.5 raw + 0.5 ewma
VARIANTS = {"raw": None,
            "ewma05": ("ewma", 0.05), "ewma10": ("ewma", 0.10),
            "ewma15": ("ewma", 0.15), "ewma20": ("ewma", 0.20), "ewma30": ("ewma", 0.30),
            "half_ewma10": ("half", 0.10), "half_ewma20": ("half", 0.20)}
ORDER = ["raw", "ewma05", "ewma10", "ewma15", "ewma20", "ewma30", "half_ewma10", "half_ewma20"]


def _smooth(blend, cfg):
    if cfg is None:
        return blend
    kind, a = cfg
    sm = blend.groupby(level=0, group_keys=False).apply(lambda x: x.ewm(alpha=a, adjust=False).mean())
    return sm if kind == "ewma" else 0.5 * blend + 0.5 * sm


def eval_ewma(cols, S, seed):
    Xtr, ytr, Rr, Rl, Rt = S["Xtr"], S["ytr"], S["Rr"], S["Rl"], S["Rt"]
    caches, y_val = S["caches"], S["y_val"]
    hgb = make_hgb(**TUNED, random_state=seed)
    with threadpool_limits(limits=1):
        hgb.fit(Xtr[:, cols], ytr)
    ref_h = np.sort(hgb.predict_proba(Xtr[:, cols])[:, 1])
    hs = [_score(hgb, c, cols) for c in caches]
    rrows, rlab, rgrp = sort_into_groups(Rr[:, cols], Rl, Rt)
    ranker = fit_ranker("rank_xendcg", rrows, rlab, rgrp)
    ref_r = np.sort(ranker.r.predict(Rr[:, cols]))
    rs = [_score(ranker, c, cols) for c in caches]
    out = {k: [] for k in VARIANTS}
    for a_s, r_s in zip(hs, rs):
        blend = pd.Series(0.6 * pct(a_s.to_numpy(), ref_h) + 0.4 * pct(r_s.to_numpy(), ref_r),
                          index=a_s.index)
        for k in VARIANTS:
            b = _smooth(blend, VARIANTS[k])
            out[k].append(ts_auc_from(b.to_frame("prediction"), y_val))
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    print(f"=== EWMA on 61-col champion  {datetime.now():%F %T}  seeds={SEARCH_SEEDS} ===", flush=True)
    store = {k: [] for k in VARIANTS if k != "raw"}
    rawmeans = []
    for seed in SEARCH_SEEDS:
        t0 = time.time()
        S = prepare_seed(seed, "champ_dep")
        res = eval_ewma(BASE_COLS, S, seed)
        rawmeans.append(res["raw"].mean())
        line = [f"[seed {seed}] raw={res['raw'].mean():.4f} ({time.time()-t0:.0f}s)"]
        for k in ORDER:
            if k == "raw":
                continue
            md, _s, t, _ = paired(res[k], res["raw"])
            store[k].append((md, t))
            line.append(f"{k}={res[k].mean():.4f} d{md:+.4f} t{t:+.2f}")
        print("  " + "  ".join(line), flush=True)
        del S

    print(f"\n=== verdict (raw mean {np.mean(rawmeans):.4f}, {len(SEARCH_SEEDS)} seeds) ===", flush=True)
    for k in ORDER:
        if k == "raw":
            continue
        ds = np.array([d for d, _ in store[k]]); ts = np.array([t for _, t in store[k]])
        npos = int((ds > 0).sum())
        ok = (ds > 0).all() and ds.mean() >= 0.0015 and ts.min() >= 2.0
        print(f"{k:<12} mean_d={ds.mean():+.4f}  min_d={ds.min():+.4f}  npos={npos}/{len(ds)}  "
              f"min_t={ts.min():+.2f}  {'SHIP-CANDIDATE' if ok else 'below-bar'}", flush=True)


if __name__ == "__main__":
    main()
