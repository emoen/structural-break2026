"""Multi-seed x multi-offset confirmation of the _grind_remainder T1-honest positive.
The single-seed screen was offset=200 only (deep-post-break, easy: champ AUC 0.887). Here we test
whether Bai-Perron / BayesCP augmentation still helps (a) across seeds and (b) at EARLY offsets
(5,25,100) that resemble the real ~0.62 streaming regime. Only a stable positive across seeds/offsets
justifies the expensive champion-inference ship-gate escalation.
"""
import numpy as np
from _grind_remainder import ON, HI, TAU, usable, build, auc, IDX, N_EACH

OFFSETS = [5, 25, 100, 200]
SEEDS = [0, 1, 2]
NAMES = ["baiperron", "bayescp", "ALL"]

agg = {off: {nm: [] for nm in NAMES} for off in OFFSETS}
base = {off: [] for off in OFFSETS}

for seed in SEEDS:
    rng = np.random.default_rng(1000 + seed)
    for off in OFFSETS:
        br = [s for s in usable if TAU[s] >= 120 and len(ON[s]) > TAU[s] + off]
        rng.shuffle(br); brc = br[:N_EACH]
        pairs = []
        for s in brc:
            pairs.append((s, TAU[s] + off, 1))
            pairs.append((s, int(rng.integers(60, TAU[s] - 20)), 0))
        Xc, Bl, y = build(pairs)
        b0 = auc(Xc, y)
        base[off].append(b0)
        line = f"seed{seed} off{off:>3} n{int(y.sum())} champ={b0:.4f}"
        for nm in NAMES:
            a = auc(np.hstack([Xc, Bl[:, IDX[nm]]]), y)
            agg[off][nm].append(a - b0)
            line += f"  {nm}{a-b0:+.4f}"
        print(line, flush=True)

print("\n=== mean deltas over seeds ===", flush=True)
for off in OFFSETS:
    row = f"off{off:>3} champ_mean={np.mean(base[off]):.4f}"
    for nm in NAMES:
        d = np.array(agg[off][nm])
        row += f"   {nm} mean{d.mean():+.4f} min{d.min():+.4f} npos{(d>0).sum()}/{len(d)}"
    print(row, flush=True)
