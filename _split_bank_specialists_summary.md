# Split-Bank Specialists Against E116

Compact rankers use mutually exclusive raw, calibration, or whitening banks.
The equal arm averages calibration and whitening specialist percentiles before blending.

| config | absolute TS-AUC | mean_d | min_d | pos | t_agg | seed 42 d / t | seed 7 d / t | seed 123 d / t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw79 @ 0.075 | 0.638085 | +0.000165 | -0.000074 | 2/3 | +0.73 | +0.000199 / +0.46 | -0.000074 / -0.24 | +0.000369 / +0.82 |
| calibration79 @ 0.025 | 0.637677 | -0.000243 | -0.000250 | 0/3 | -2.67 | -0.000245 / -1.43 | -0.000235 / -1.28 | -0.000250 / -1.91 |
| whitening15 @ 0.050 | 0.638019 | +0.000099 | -0.000420 | 2/3 | +0.48 | -0.000420 / -1.59 | +0.000343 / +0.88 | +0.000376 / +0.97 |
| calibration_whitening_equal @ 0.025 | 0.637856 | -0.000064 | -0.000201 | 2/3 | -0.96 | -0.000201 / -2.36 | +0.000002 / +0.01 | +0.000007 / +0.09 |
