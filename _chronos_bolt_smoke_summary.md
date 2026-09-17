# Chronos-Bolt Tiny Frozen-Embedding Smoke

Model: `amazon/chronos-bolt-tiny`; representation: final encoder `[REG]` token (256 dimensions); seed 42; first 2 folds.

E116 absolute TS-AUC: **0.644043**. Standalone Chronos specialist: **0.532300**. Cross-fit residual TS-AUC: **0.509801**.

| config | absolute TS-AUC | mean_d | fold 0 d | fold 1 d |
| --- | ---: | ---: | ---: | ---: |
| Chronos ranker @ 0.025 | 0.644220 | +0.000178 | -0.000329 | +0.000684 |
| Chronos ranker @ 0.050 | 0.644196 | +0.000153 | -0.000909 | +0.001215 |
| Chronos ranker @ 0.075 | 0.643970 | -0.000073 | -0.001731 | +0.001585 |
| Chronos ranker @ 0.100 | 0.643450 | -0.000592 | -0.002916 | +0.001732 |

| prediction | Spearman with Chronos |
| --- | ---: |
| ranker | 0.129174 |
| rf | 0.141366 |
| xgb | 0.107258 |
| catboost | 0.100984 |
| E116 | 0.145466 |
