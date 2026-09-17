# E122 Full White-GLOC Likelihood-Curve Screen

Specialists consume the 48-bin causal, affine-invariant Gaussian gain curve only.
Baseline is exact E116 with its accepted allocation and fixed-CDF weights.

| config | absolute TS-AUC | mean_d | min_d | pos | t_agg | seed 42 d / t | seed 7 d / t | seed 123 d / t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| curve_ranker @ 0.050 | 0.638128 | +0.000208 | +0.000061 | 3/3 | +0.77 | +0.000061 / +0.15 | +0.000292 / +0.80 | +0.000270 / +0.43 |
| curve_point_lgbm @ 0.025 | 0.637750 | -0.000170 | -0.000339 | 0/3 | -2.14 | -0.000088 / -0.67 | -0.000082 / -0.63 | -0.000339 / -2.29 |
