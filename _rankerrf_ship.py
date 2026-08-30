"""Confirm the JOINT delta of the ship candidates on the ranker+rf base, from cache.

The re-screen scored each feature block alone. Before wiring a submission we need the
combined effect of the two 3/3-positive winners (perm_entropy4 + max_split_lr), and the
+mosum follow-up, on the same ranker+rf blend / seeds. Reuses _screen_cache/seed_*.pkl.
"""
import numpy as np

from _night_rankerrf import (prepare_seed, make_folds_meta, score_cols, cand_cols,
                             BASE_COLS, SEARCH_SEEDS)

COMBOS = {
    "base": (),
    "+perm4": ("perm_entropy4",),
    "+mosum": ("mosum",),
    "+perm4+mosum": ("perm_entropy4", "mosum"),
    "+perm4+maxlr+mosum": ("perm_entropy4", "max_split_lr", "mosum"),
}

rows = {k: [] for k in COMBOS}
base_by_seed = {}
for seed in SEARCH_SEEDS:
    sd = prepare_seed(seed)
    metas = make_folds_meta(sd)
    b = score_cols(sd, metas, BASE_COLS, seed)
    base_by_seed[seed] = b
    for name, keys in COMBOS.items():
        cols = BASE_COLS if not keys else cand_cols(keys)
        auc = score_cols(sd, metas, cols, seed)
        rows[name].append((seed, auc, auc - b))
        print(f"[seed {seed}] {name:<22} auc={auc:.5f} d={auc-b:+.5f}", flush=True)

print(f"\n{'combo':<22}{'score':>9}{'mean_d':>10}{'t':>8}{'npos':>7}")
for name in COMBOS:
    ds = np.array([d for _s, _a, d in rows[name]])
    aa = np.array([a for _s, a, _d in rows[name]])
    sd_ = ds.std(ddof=1)
    t = ds.mean() / (sd_ / np.sqrt(len(ds))) if sd_ > 0 else 0.0
    npos = int((ds > 0).sum())
    print(f"{name:<22}{aa.mean():>9.5f}{ds.mean():>+10.5f}{t:>8.2f}{npos:>4}/{len(ds)}")
