# E123 Fixed-Position Multichannel White-GLOC Curve

The compact grouped ranker consumes 48-bin amplitude, normalized-shape,
posterior-mass, and validity-mask channels on a fixed relative-position grid.
Baseline is exact E116 with its accepted allocation and fixed-CDF weights.

| config | absolute TS-AUC | mean_d | min_d | pos | t_agg | seed 42 d / t | seed 7 d / t | seed 123 d / t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| curve2_ranker @ 0.025 | 0.638107 | +0.000187 | +0.000134 | 3/3 | +1.14 | +0.000262 / +1.28 | +0.000134 / +0.61 | +0.000165 / +0.40 |
| curve2_ranker @ 0.050 | 0.638132 | +0.000212 | +0.000055 | 3/3 | +0.63 | +0.000421 / +0.98 | +0.000055 / +0.13 | +0.000160 / +0.19 |
| curve2_ranker @ 0.075 | 0.637985 | +0.000065 | -0.000206 | 1/3 | +0.13 | +0.000401 / +0.61 | -0.000206 / -0.31 | -0.000001 / -0.00 |
| curve2_ranker @ 0.100 | 0.637699 | -0.000221 | -0.000560 | 1/3 | -0.32 | +0.000221 / +0.25 | -0.000560 / -0.61 | -0.000323 / -0.19 |
