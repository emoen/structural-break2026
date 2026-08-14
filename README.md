# Structural Break — Real-Time (CrunchDAO)

Streaming structural-break detection. For each series we get a full `x_historical`
segment (no break) and an `x_online` segment revealed one point at a time. At every
online step `t` we emit a score `s_t ∈ [0,1]` = belief a break has already occurred.

**Metric — TS-AUC (Time-Stratified AUC).** At each step, AUC is computed *across
series* (label 1 if `t ≥ tau`), weighted by `w(t) = n_pos(t)·n_neg(t)`. Random = 0.5.
Because it is cross-sectional per step, scores that merely grow with `t` gain nothing —
only genuine per-step separation helps.

## Files

| File | Purpose |
| --- | --- |
| `streaming_detector.ipynb` | The submission notebook (supervised model). |
| `_exp_fast.py` | Standalone train + streaming-validation harness (measures TS-AUC and infer speed). |
| `_val_harness.py` | **Robust multi-fold** validation harness: per-candidate mean±std across K disjoint folds + paired deltas vs base (the noise-aware decision tool, see below). |
| `baseline.ipynb` | Original EWMA baseline from the competition. |
| `data/` | Local train/test parquet files. |

## Model (current submission)

Per online step, a trailing window (`WINDOW=75`) is compared to the historical segment
via 14 two-sample features, and a `HistGradientBoostingClassifier` maps them to a break
probability. Features are recomputed every `PREDICT_EVERY` steps and held between.

Features: `mean_z, log_var_ratio, ks, skew_diff, kurt_diff, ac1_diff, ac2_diff,
iqr_ratio, tail2, tail3, zero_cross, slope_norm, spec_centroid_diff, rng_ratio`.

Key config: `WINDOW=75, MIN_N=15, HIST_CAP=200, PREDICT_EVERY=1, MAX_ITER=200,
STEPS_PER_SERIES=10, MAX_TRAIN_SERIES=10000, SEED=42`.

## Validation methodology

Feature decisions are made with `_val_harness.py`, **not** a single val split. It trains
one model per candidate feature set on a fixed training pool (disjoint from every fold),
evaluates each on `N_FOLDS=5` disjoint `FOLD_SIZE=400`-series folds, and reports:

* **per-fold TS-AUC + mean + std** — the std is the *noise floor*;
* **paired delta vs base** on the same folds, with `t = mean_delta / (std_delta/√K)`.

**Decision rule:** only port a change to the notebook when its paired mean gain clearly
exceeds the noise floor (target `|t| ≥ 2`, gain > fold std). This exists because E8's
+0.006 single-split "win" was ~4× smaller than the fold-to-fold noise and *reversed* on
the leaderboard. Streaming is parallelised across series (joblib, BLAS capped to 1).

*First run (5×400 folds, 6000-series pool):* base `mean 0.5511, std 0.0125` (same model
swings 0.5402→0.5708 across folds); the E8 split features scored `+0.0009 (t=0.31)` —
decisively **noise**. The harness would have rejected split *before* submitting it.

## Experiment log

| # | Approach | Result | Notes |
| --- | --- | --- | --- |
| E0 | EWMA z-score baseline (submitted) | **LB 50.65%** (rank 388) | Only reacts to mean shifts; blind to variance/shape/dependence. |
| E1 | Unsupervised (KS + variance + mean, noisy-OR) | ~0.518 local | Marginal-distribution stats miss autocorrelation/stationarity breaks. |
| E2 | Unsupervised CUSUM / monotone-accumulating variants | ~0.50 local | *Worse.* Saturating/accumulating scores destroy cross-sectional separation. |
| E3 | Supervised HGB, 14 two-sample features (scipy) | **TS-AUC 0.5701** (800 val series) | Best score, but too slow → submission timed out (exit 124). |
| E4 | E3 rewritten: pure-numpy features + single-thread + fewer trees | **TS-AUC 0.5636**, ~10× faster infer | Speed fix, negligible score change. |
| E5 | Cap threads at call time (`threadpool_limits`) + raise parallelism | **LB 55.30%** (rank 296/1144) | Fixed the 5 h+ cloud hang; ran fast, determinism check passed. |
| E6 | `PREDICT_EVERY` 15 → 1 (predict every step, no staleness) | **LB 55.48%** (rank 289) | Free win, +0.18 pt. 31 min on 16 vCPU. Local 0.5636 → 0.5663. |
| E7 | Train on all 10000 series (`MAX_TRAIN_SERIES` 6000 → 10000), keep 10 steps/series | **LB 55.94%** (rank 268) | +0.46 pt. Trained 9934 series/98.8k rows; parallelism 15; 42 min. |
| E8 | +3 recent-vs-older split features (window halves) | **LB 55.70%** (rank 269) — *reverted* | Local +0.006 but LB **−0.24 pt** vs E7. 400-series val was misleading; rolled back to 14 feats. |
| E9 | Robust/rank-based stats (median/MAD, Mann–Whitney, Brown–Forsythe) via 5×400-fold harness | **all within noise** (best t=+0.19) — *not shipped* | Redundant with existing KS/`mean_z`/`log_var_ratio`. No submission; harness caught it locally. |
| E10 | HGB hyperparameter grid (108 configs, GroupKFold row-level proxy) | best +0.0013 proxy AUC — *not shipped* | Model isn't the bottleneck; gain under proxy noise. |
| E11 | Window geometry: `WINDOW` 75 → 130 | **LB 56.63%** (rank 235) | +0.69 pt, +33 ranks — biggest single jump. Infer cost ~identical (free). 10-fold Δ +0.0110, t=+3.96. |
| E12 | Overnight 40-candidate feature battery (5×400 folds @ W75) | 3 ship-grade: `runs` t=2.48, `combo_volatility` t=2.40, `spec_entropy` t=2.39 | Winners cluster by family (run/level-crossing, complexity/spectral). Re-confirm at W130 before shipping. |
| E13 | Re-confirm E12 winners at WINDOW=130 (10 folds); ship `spec_entropy` (15th feat) | **LB 56.84%** (rank 229) | +0.21 pt, +6 ranks. `runs` (t=0.81) and `combo_volatility` (t=0.41) **collapsed at W130** — window-specific artifacts. Only `spec_entropy` held (t=+4.67). |
| E14 | Ship `perm_entropy` (order-3, 16th feat) on promising single-feature signal | **LB 56.75%** (rank 229) — *reverted* | −0.09 pt vs E13; sub-threshold bet (t=+1.93) didn't transfer. Reverted to 15 feats. Spectral-shape family (`spec_flatness/rolloff/spread/peak`) all noise on top of `spec_entropy`. |
| A | Lever A: per-series score calibration (sub/z early-window baseline, causal expanding median) | **all negative** (best t=−2.49) — *not shipped* | Pooled-trained `predict_proba` is already cross-series-comparable; de-meaning removes signal. Under TS-AUC only better within-step *ordering* helps → pursue changepoint localisation (Lever B). |
| B | Lever B: within-online max-split changepoint features on base15 @W130, 10 folds (`max_split_t`/`lr`/`var`/`pos` + combo) | **all noise** (best `max_split_var` t=+0.95) — *not shipped* | Self-normalized within-window supremum stats (Chow/LR/var/position). Smoke t=+4.34 (`max_split_t`) was a 2-fold artifact → collapsed to t=−1.37 at 10 folds. The 130-window vs-history features already capture the mean/var shift; within-window split is redundant. |
| E16 | WINDOW re-sweep on base15, 10 folds: W=130 (base) vs 160/180/200/250 | **W=130 optimal** — *no change* | W=160 +0.0016 t=+0.54 (noise); 180 −0.0042 t=−1.81; 200 −0.0018; 250 −0.0011. Champion window confirmed at the plateau top; larger windows don't ship. |
| E17 | Multi-scale: base15 features computed on nested trailing sub-windows and concatenated, 10 folds paired vs base15@130 | **LB 57.50%** (rank 214) — *CHAMPION* | `ms_60_130_250` shipped (Δ+0.0075 t=+3.06). +0.66 pt, +15 ranks vs E13 — biggest single jump since WINDOW=130. Scales (60,130,250) → 45 feats @ buffer W=250. Also SHIP in CV: `ms_40_130_250` (t=+2.27), `ms_40_130` (t=+2.02). First representational lever to pay off — short (responsive) + champion (130) + long (slow) views of the same series-vs-history contrast. |
| E18 | Full-online-vs-history: add UNBOUNDED scale (base15 over entire online-so-far) on top of champion, 10 folds paired vs `ms_60_130_250` | **all noise** (best `ms_60_130_250_full` t=+0.51) — *not shipped* | Unbounded scale via BIG buffer + FULL sentinel. `ms_60_130_250_full` (60f) Δ+0.0035 t=+0.51 (higher raw mean 0.5829 but fold-inconsistent); `ms_60_130_full` (45f) t=−0.13; `ms_130_full` (30f) t=−0.72. The trailing 250 view already captures post-break evidence; the growing full-online contrast is dominated by stale pre-break points and adds no reliable signal. Full-online lever dead. |
| E19 | Scale-set sweep: alternative multi-scale sets vs champion `ms_60_130_250`, 10 folds (4th scale / different spacing / longer top: `ms_40_60_130_250`, `ms_60_90_130_250`, `ms_60_130_190_250`, `ms_60_100_160_250`, `ms_40_90_160_250`, `ms_60_130_250_350`) | **all noise** — *no change* | No alternative scale set beats (60,130,250). Adding a 4th scale, re-spacing, or extending the top scale all wash out. Also confirmed on LB: probe `ms_40_130_250` = 57.38%/216 (worse). Champion scale set confirmed; representation saturating. |
| E20 | Feature-space distances: anomaly of the 45-dim multiscale vector (Mahalanobis / QDA-LLR / std-Euclid / L1 / Linf, window-vs-history), 10 folds paired vs champion | **all noise** (best `maha` t=+0.73) — *not shipped* | `maha` Δ+0.0022 t=+0.73 (raw 0.5816 highest but fold-inconsistent); `seuclid` t=+0.26; `qda_llr` t=−0.70; `l1` t=−1.22; `linf` t=−1.89. Joint-covariance whitening adds nothing over axis-aligned GBT splits. Feature-space distance lever dead. |
| E21 | Two-sample distribution distances: window-vs-history Wasserstein-1 / energy / Cramér–von-Mises / MMD per scale, 10 folds paired vs champion | **best non-champion signal since E17, but below bar** (`ts_wass` t=+1.83) — *not shipped from CV* | `ts_wass` Δ+0.0026 t=+1.83; `ts_energy` t=+1.81 (≈collinear w/ wass — both CDF-gap integrals); `ts_cvm` t=+1.44; `ts_mmd` t=+0.87; stacking all (`ts_all`, 57f) dilutes to t=+1.07. Real distribution-shape signal exists but sits at t≈1.8, inside the CV↔LB gap. |
| E22a | Ship `ts_wass` (single strongest E21 distance) to LB | **LB 57.37% (rank 217)** — REVERSED, *reverted* | Wired `_w1` per scale → 48 feats, bit-exact vs harness. Lost −0.13 pt / −3 ranks vs champion despite CV t=1.83. **Now 0-for-2 on marginal features** (E14 perm_entropy t=1.93 also reversed). *** Hardened ship bar: do NOT spend an LB slot unless CV t≥~2.5 (ideally ≥3); the old t≥1.9 rule is retired. *** |
| E22b | `ts_cvm` (rank-based, orthogonal to wass) submitted as parallel LB candidate | **LB 57.40% (rank 218)** — REVERSED, *reverted* | CV t=1.44 (below even the retired bar) → lost −0.10 pt vs champion, as the hardened rule predicted. **Now 0-for-3 on marginal features** (perm_entropy t=1.93, wass t=1.83, cvm t=1.44 all reversed). Two-sample distances fully dead as ship candidates; champion stands. |
| E23 | Series-transforms: cumsum / diff / moving-std / abs-dev of the standardized series, each contrasted (mean_z, log_var_ratio, KS) window-vs-history per scale, 10 folds paired vs champion | **all noise** (best `tf_diff` t=+1.40) — *not shipped* | `tf_diff` Δ+0.0018 t=+1.40; `tf_cumsum` t=+0.53; `tf_mstd` t=−0.21; `tf_absdev` t=−0.32; `tf_all` (81f) dilutes to t=+0.53. These are exactly the 2nd-place solution's transforms — noise here because our problem is real-time/unknown-boundary. Series-transform lever dead. |
| E24 | Variance two-sample tests on top of champion: F-test / Brown–Forsythe Levene / Fligner–Killeen per scale, 10 folds paired vs champion | **all noise** (best `vt_fligner` t=+0.42) — *not shipped* | `vt_fligner` Δ+0.0013 t=+0.42; `vt_levene` t=+0.32; `vt_lev_fli` (51f) t=+0.13; `vt_ftest` t=+0.10 (baseline 0.5795). The 2-fold smoke `vt_fligner` t=15.33 was a pure small-fold artifact (→ 0.42 at 10 folds) — reconfirms smoke deltas are worthless. Robust variance tests redundant with the existing `log_var_ratio`. |
| E25 | Large feature bank + importance selection (Option A): champion base45 + boundary-agnostic families {Hurst, AR(4), Anderson–Darling, Levene, Fligner, Mood, sample-entropy} ×3 scales (75 feats), ExtraTrees Gini selection (keep base + new features ≥ weakest-base importance), 10 folds paired vs champion | **noise** (0.5759 vs 0.5795, Δ=−0.0036, **t=−1.04**, slightly worse; 71.7 min) | Replicated the top solutions' *structural* recipe (large bank + selection). Selection kept all 30 new features (AR(4)/sampen/variance-tests rank high alongside base), yet the 75-feat model did **not** beat the 45-feat champion — it was marginally worse. Strong evidence the trailing-window paradigm is **saturated**; no feature-bank path clears the bar. → bank the champion. |
| E26 | **Paradigm search** — change the training *target* (frozen champion feats): ramp regression `clip((t-τ)/H,0,1)` + deweight-near-τ sample weights vs champion binary classifier, 10 folds | **noise** (ramp20 t=+0.68, deweight20 t=+0.59, ramp40 t=−0.50; all vs 0.5795) | Target-shaping does not move the ranking metric. Dead lever. |
| E27 | **Paradigm search** — scale/diversify training *data* (breadth vs depth, E7): N_TRAIN ∈ {6k, 10k, MAX} ×10 steps + depth control 6k×20, fixed val folds, champion feats+model | **dead / exhausted** (breadth impossible; depth t=+0.82 noise) | Dataset = 10 000 series; 10-fold×400 val reserves 4 000 → **MAX_TRAIN = 6 000 = exactly what the champion already trains on**. The breadth configs capped back to 6 000 (Δ=0, t=+inf artifact); the smoke's 9 920 was an artifact of the tiny 2-fold val reservation. No more data to add without shrinking validation. Depth (2× steps/series) = noise (matches E7). |
| E28 | **Paradigm search** — stateful *unsupervised* online detectors: CUSUM / Page-Hinkley / BOCPD (Gaussian const-hazard), per-step score ranked by TS-AUC vs trained champion, 10 folds | **far below champion** (cusum 0.5227 t=−11.6, page-hinkley 0.5202 t=−10.2, bocpd 0.5427 t=−6.8 vs 0.5795) | Stateful online detectors as standalone rankers are much weaker than the trained model. BOCPD is the best of them but still ~0.54 alone — too weak to help as a feature given the saturation evidence. Dead lever. |
| E29 | **Paradigm search** — winners' toolchain: LightGBM + gain selection. A) LGBM vs HGB on champion feats (model swap); B) bank (champion + Hurst + AR(4)) → LGBM gain>0 select → LGBM, 10 folds | **noise / worse** (A model-swap t=−2.18 worse; B bank+select t=−1.03; vs 0.5795) | LightGBM did not beat HistGBM on our features, and the bank→gain-select path was noise. Confirms the 7pt gap is **problem structure** (2025 known-boundary offline vs 2026 real-time unknown-boundary), not toolchain. The smoke SHIP (t=+2.68) fully evaporated — textbook "never trust smoke". |
| E30 | **Ensemble + tuning** — HGB champion ⊕ LogisticRegression blend, tune both. New validation: fixed-train 80/20 stratified (8000 train / 2000 val, 10 folds × 200, per-fold break-rate ≈0.497). LR default + C∈{0.1,0.3,1,3}+balanced; HGB grid 7 configs; blend weight sweeps; best×best blend | **all noise / worse** (LR alone 0.5322 t=−4.91; best HGB `msl40` +0.0004 t=+0.33; every blend ≤0, best w_lr=0.2 t=−0.68; best×best t=−0.01; vs champion 0.5674) | LR adds **no orthogonal signal** — blending it in only degrades monotonically, so HGB already captures the linearly-separable structure. HGB tuning is **dead-flat** (7 configs within ±0.0004). Model/ensemble axis is exhausted; the ceiling is the feature signal, not the model. Dead lever. |
| E31 | **Sequence model + proper tuning** — (A) 1D-CNN over the raw history-standardized window (2×250, translation-invariant) vs champion, same folds/labels/TS-AUC; (B) randomized *joint* HGB search (24 configs) then 5-seed robustness re-confirmation | **both null** — CNN alone 0.5048 t=−6.62, every HGB+CNN blend negative; joint search "winner" `hgb#11` (lr0.047/it300/leaves31/msl40/l2=5/mf0.5) hit +0.0035 t=+3.51 on seed 42 but **NOT ROBUST** across independent seeds (delta +0.0016 mean, positive 4/5, t≥2.5 on 1/5 — selection bias from best-of-24) | The learned-representation lever fails: a CNN reading the raw window is *worse* than 45 engineered contrasts (the online window is short/noisy; hand features already extract the shift/variance signal). Proper joint tuning found no config that survives independent fold draws → the E30 flat-tuning verdict holds. Champion E17 stands. |
| E32 | **Multi-seed joint HGB tuning** (the principled fix for E31's selection bias) — sample 48 joint configs once, score each by *mean* paired-delta vs champion across 4 independent pool seeds {42,7,123,2024}; gate = positive on all seeds AND mean ≥ 0.0015; then confirm the winner on 4 *fresh* disjoint seeds {31337,2025,555,99} | **LB 57.72% (rank 210) — NEW CHAMPION** | Robust winner `{lr 0.0211, max_iter 400, max_leaf_nodes 63, min_samples_leaf 10, l2 1.0, max_features 0.5}`: search-seed mean_d +0.0038 (positive 4/4), fresh-seed mean +0.0019 (positive 4/4) → **positive on 8/8 independent splits**. +0.22 pt vs old champion 57.50%/214, and beats the seed-ensemble (57.63%). Direction = slow-lr + many-trees + feature-bagging + deeper trees = generalization-friendly boosting; transfers where marginal features didn't. Single model (infer ~2× champion). Shipped as `streaming_detector_tuned.ipynb`. First robust win of the tuning arc. |
| E32-seedens | Seed-ensemble of the champion (K=5 HGB, avg proba) submitted in parallel | **LB 57.63%** (was 57.50%) | +0.13 pt from pure variance reduction. Local harness was a "HOLD" (mean +0.0026, 2/3 seeds) yet still gave a small **real** LB gain → confirms the local harness is *directionally predictive* for small positive deltas. Superseded by E32 tuned. |
| E33 | **LambdaMART / rank objective** — LGBMRanker (group = online step `t`) with `lambdarank` + `rank_xendcg`, dense-60 vs sparse-10 sampling, LGBM pointwise control, vs *old* champion HGB, 4 pool seeds | **biggest local signal since E17** — `rankxend_dense` mean_d +0.0159 (positive 4/4), min +0.0087, t=+2.69; pointwise control +0.0057 (toolchain adds a little, but the rank *objective* adds ~+0.010 more) | Real train/metric-mismatch signal (TS-AUC is per-step ranking; we always trained pointwise). Caveats: baseline was old (not tuned) champion, and winner was best-of-4 → must re-baseline vs tuned + confirm on fresh seeds (→ E34). |
| E34 | **Confirm rank objective vs the *tuned* champion on fresh seeds** {31337,2025,555,99} | **FAILED gate** — `rankxend_dense` mean_d +0.0080, min −0.0058, t=+1.33, positive **not** all seeds (negative on 555); `lambdarank_dense` +0.0083, min −0.0007, t=+1.35 | E33's +0.0159 was inflated by best-of-4 selection **and** a weaker old baseline. True rank-objective effect vs the tuned champion ≈ +0.008 but **not robust** → standalone rank dead as a champion replacement. Only remaining use of the rank signal = a rank-*average* ensemble with the tuned HGB (→ E36). Lesson: always re-baseline levers vs the *current* champion on fresh seeds. |
| E35 | **Feature re-eval vs the tuned champion** (night run, 3.7 h) — re-screen the whole battery (53 candidates: 47 EXTRA singles + 6 combos) added at scales 60/130/250 on top of the 45-feat champion, tuned params, 4 seeds | **all noise** — zero CONFIRM; best `max_split_lr`/`spec_peak` +0.0024 mean_d, t≈0.9; only 5 candidates positive 4/4, all t<1.0 | Feature-**addition** lever exhausted under *both* the default and the tuned model — the 45-feat champion already extracts the available single-feature signal, and tuned params did not resurrect any rejected feature. Remaining levers are structural/model-level (ensembles), not feature-level. |
| E36 | **Rank + tuned ensemble** — rank-average the tuned-champion HGB proba with an LGBM `rank_xendcg` (dense-60) raw score; per fold blend `w·rank(hgb) + (1−w)·rank(ranker)`, sweep w, 4 pool seeds {42,7,123,2024} | **BLOCKBUSTER (local) — every weight 0.8→0.3 CONFIRM (positive on all 4 seeds)** | mean_d +0.0067 (w0.8) → +0.0139 (w0.3); mean_t +5.22→+3.43; min_d +0.0040→+0.0075; sweet spot ~w0.5 (mean_d +0.0128, t 4.22, min_d +0.0073). Biggest signal since E17. Works where standalone rank (E34) failed: the champion anchors, the decorrelated listwise ranker corrects within-step orderings → variance reduction turns a non-robust +0.008 model into a robust +0.007–0.014 blend. **Ship caveat:** the harness uses a *live* per-fold global rank (not available at inference); the deployable form replaces it with each model's *fixed empirical CDF from training scores* (`w·F̂_hgb + (1−w)·F̂_rank`). **Confirmed on fresh seeds {31337,2025,555,99}:** the fixed-CDF blend transfers ~losslessly (FIX_mean_d tracks the live rank to 3 dp) — w=0.6 mean_d +0.0080, min +0.0018, t +3.07, positive 4/4; w=0.5 mean_d +0.0086, min +0.0009, t +2.60, positive 4/4 (both `*** CONFIRMED (ship) ***`; w=0.4 positive only 3/4). → build the ensemble submission notebook at w≈0.6. **SHIPPED at w=0.6 → LB 58.78% (rank 176) — NEW CHAMPION, +1.06pt & +34 ranks vs tuned HGB (57.72%/210): biggest single jump of the arc.** Local fixed-CDF mean_d ~+0.008 → +1.06pt LB (harness directionally predictive; decorrelated members transfer strongly). lightgbm installed fine on cloud. |
| E37 | **ARCH / volatility-clustering feature** — `f_arch` = [autocorr(r², lag1), Σ_{k≤5} autocorr(r²,k)²] over the 3 scales (+6 cols), targeted paired test vs tuned champion, seeds {42,7,123,2024} (`FEAT_ONLY` filter added to `_feat_reeval.py`) | **NOISE — rejected** | mean_d −0.0002, min −0.0054, mean_t −0.04, positive 2/4; signs flip across seeds. Confirms E35 feature saturation — volatility clustering adds nothing on the 45-feat champion. Do not revisit ARCH-family. |
| E38 | **RandomForest member** — `RandomForestClassifier(n_estimators=400, max_features=sqrt, min_samples_leaf=5)` on the 45 champion feats; blended (proba-avg) with the tuned HGB, paired-t vs tuned-HGB-alone, 8 independent seeds | **CONFIRM on 8/8 seeds — decorrelated member** | pool {42,7,123,2024}: hgb_rf 0.5/0.5 mean_d +0.0064 t+2.92 4/4, hgb_rf_30 0.7/0.3 +0.0044 t+3.21 4/4. Fresh {31337,2025,555,99}: hgb_rf +0.0058 t+2.81 4/4, hgb_rf_30 +0.0040 t+3.25 4/4. RF *alone* ~beats tuned HGB (pos 8/8). Bagging genuinely decorrelates from boosting (contra E30 expectation). **Caveat:** tested vs HGB-alone with raw proba-avg; open question = does RF add on top of the *ranker* in the shipped CDF-blend (step 2, 3-member ensemble). |

**Takeaways:** instantaneous window-vs-history comparison beats any accumulating score
under TS-AUC; the local 100-series reduced set is noisy/optimistic (use ≥800 series to
judge); a supervised classifier over distribution-comparison features clearly beats
hand-tuned unsupervised rules. **On training data (E7):** breadth beats depth — training
on *more distinct series* helps, but taking *more sampled steps from the same series*
hurts (correlated, near-duplicate rows), and *oversampling steps near the break `τ`* hurts
(the most ambiguous, hardest-to-label points). **On inference (E6):** never let a score go
stale — TS-AUC ranks series against each other *at each step*, so predict every step.
**On validation (E8):** a single 400-series local split is too noisy to trust ~0.006
deltas — the split features looked like a win locally but *lost* on the full test; use a
larger/multi-fold val or treat the leaderboard as ground truth for small changes.
**On features (E9):** robust/rank-based restatements (median/MAD, Mann–Whitney,
Brown–Forsythe) add nothing over the existing KS + `mean_z` + `log_var_ratio` — they
measure the same location/scale/distribution shift, so they are redundant, not additive.

## Experiment details

### E0 — EWMA z-score baseline

The competition baseline. Summarize history by its mean `μ_H` and std `σ_H`, then track
an exponentially-weighted moving average of the online values and score by how far it has
drifted, in standard-deviation units:

$$z_t = \frac{\bar{x}_t - \mu_H}{\sigma_H / \sqrt{n_\text{eff}}}, \qquad s_t = \tanh(|z_t|)$$

Truly O(1) per step, so it never times out. But it **only measures the mean** — a break
that changes variance, distribution shape, or autocorrelation while keeping the mean
roughly constant is invisible to it. That is why it lands at essentially random
(**LB 50.65%**) and set the bar for everything below.

### E1 — Unsupervised two-sample (KS + variance + mean)

First improvement attempt: at each step compare a trailing online window against the
historical segment with several classical statistics — a Kolmogorov–Smirnov distance, a
variance ratio, and a mean shift — then combine them with a noisy-OR into one score.

This adds sensitivity to variance and distribution-shape changes the EWMA misses, so it
edges above random (**~0.518 local**). But it is still built on *marginal-distribution*
statistics: it cannot see breaks that live in the **temporal dependence** (e.g. a change
in autocorrelation with an unchanged histogram). Hand-tuning the OR weights gave only
marginal, noisy gains — the ceiling of rule-based combining was clearly low.

### E2 — CUSUM / monotone-accumulating variants

Intuition said "accumulate evidence over time" — CUSUM-style running sums and scores
forced to be monotonically non-decreasing. Under TS-AUC this **backfired** (**~0.50**).

The reason is the metric's cross-sectional nature: at each step AUC ranks series *against
each other*. A score that only grows with `t` orders series by how long they have been
running, not by whether they have actually broken — so it carries no discriminative
signal at any given step. This produced the single most important design rule:

```mermaid
flowchart LR
    A["Break happens at tau"] --> B["Window vs history<br/>diverges NOW"]
    B --> C["Score jumps at tau,<br/>relaxes if evidence fades"]
    C --> D["High per-step<br/>cross-sectional separation"]
    E["Accumulate / saturate<br/>over time"] --> F["Score encodes elapsed t,<br/>not the break"]
    F --> G["No separation ⇒ AUC ≈ 0.5"]
```

**Instantaneous** window-vs-history comparison wins; accumulation loses.

### E3 — Supervised HGB with two-sample features (scipy)

Replace hand-tuned rules with a learned mapping. For each training series we take the
historical context once, sample online positions, build a 14-dim feature vector from the
trailing window, and label it `1` if the position is at/after the break `tau`. A
`HistGradientBoostingClassifier` learns the mapping; `lightgbm` was unavailable, and HGB
avoids a platform dependency.

The 14 features are all **window-vs-history contrasts** covering the failure modes of
E0/E1: level (`mean_z`), scale (`log_var_ratio`, `iqr_ratio`, `rng_ratio`), shape
(`ks`, `skew_diff`, `kurt_diff`, `tail2`, `tail3`), dependence (`ac1_diff`, `ac2_diff`,
`zero_cross`), and trend/spectrum (`slope_norm`, `spec_centroid_diff`).

This was the **best score (TS-AUC 0.5701** on 800 val series) — but the scipy-based
features cost ~7 ms each, and with millions of calls the cloud `infer()` **timed out
(exit code 124)**. Correct model, unshippable speed.

### E4 — Pure-numpy rewrite (current submission)

E4 keeps E3's model and features but makes them fast enough to ship. The inference loop:

```mermaid
flowchart LR
    P["new online point"] --> W["update trailing<br/>window (O(1))"]
    W --> C{"step %<br/>PREDICT_EVERY<br/>== 0 ?"}
    C -- no --> H["hold last score"]
    C -- yes --> F["compute 14 features<br/>(pure numpy)"]
    F --> M["HGB predict_proba"]
    M --> S["new score"]
    H --> Y["yield s_t"]
    S --> Y
```

**Diagnosis (measured, not guessed):**

- scipy `compute_features` (`ks_2samp`, `skew`, `kurtosis`, `polyfit`) cost **~7 ms/call**.
- Split timing inside infer showed **`predict_proba` dominated**, not feature compute.
- Single-row `predict_proba` benchmark:
  - default multi-thread: **1.56 ms/call**
  - `OMP_NUM_THREADS=1`: **0.29 ms/call** (random-data model, few trees)
  - real 200-tree model, single-thread: **~2.5 ms/call**
  - batched (not usable in a streaming generator): **0.6 µs/row**
- Cause: OpenMP thread-pool spin-up per tiny prediction, worsened by parallel infer
  processes oversubscribing cores.

**Fixes applied:**

- Pure-numpy features (manual skew/kurtosis moments, KS via pre-sorted sample +
  `searchsorted`, direct slope formula): **7 ms → 0.36 ms/call**.
- `OMP_NUM_THREADS=1` (+ OpenBLAS/MKL) set before numpy import.
- `MAX_ITER` 300 → 200 trees; `PREDICT_EVERY` 5 → 15; `HIST_CAP` 400 → 200;
  `MAX_TRAIN_SERIES=6000` cap.

**Validation (400 held-out train series, `_exp_fast.py`):**

- TS-AUC **0.5636** (vs 0.5635 pre-optimization — no loss).
- Infer **40 s / 400 series** → est. **~250 s at parallelism 4** for full test
  (was ~2400 s+; exit 124 before).
- Notebook re-verified: parses, `train()`+`infer()` run on synthetic data, exactly one
  score in `[0,1]` per online point.

### E5 — Cloud thread oversubscription (current submission)

After E4 passed locally, a cloud run **ran 5+ hours without finishing** (16 vCPU / 64 GB).

**Cause:** E4 capped threads with `os.environ["OMP_NUM_THREADS"]="1"`, which only takes
effect if set *before* numpy/OpenMP load. On the cloud the crunch runner imports
numpy/sklearn before the notebook cell runs, and each `infer()` worker is a separate
process — so the env var was ignored. With `INFER_PARALLELISM=4` workers each spawning
~16 OpenMP threads on 16 cores (~64 threads contending), every single-row `predict_proba`
became 10–50× slower.

**Fix:**

- Wrap the infer loop in `with threadpool_limits(limits=1):` (from `threadpoolctl`,
  ships with sklearn) — caps BLAS/OpenMP threads **at call time**, regardless of when
  libraries loaded or how workers are spawned.
- Raise `INFER_PARALLELISM` to `cpu_count()-1` now that each worker is single-threaded,
  to use the available vCPUs (auto-scales to whatever machine you pick).

**Where the parallelism happens:** across *series* (many single-threaded worker
processes, one series each), **not** inside a math op (BLAS/OpenMP threads capped to 1).
Parallelizing both layers on the same cores just oversubscribes them — hence 1 thread
per op × N worker processes = all cores busy, zero contention.

Model and features are unchanged from E4, so the score is unaffected; only wall-clock
changes. Re-verified on synthetic data (train + infer, scores in `[0,1]`).

**Result (2026-08-10):** cloud run finished quickly (parallelism 12, no hang), the
determinism re-check passed, and it scored **LB 55.30% — rank 296 / 1144** (up from
E0's rank 388). Confirms the hang was pure thread oversubscription, not the model.

### E6 — Predict every step (remove staleness)

With speed solved (E4/E5), `PREDICT_EVERY=15` was pure downside: TS-AUC ranks series
against each other **at each step**, so a score held 14 steps stale is compared against
freshly-updated peers. Swept `PREDICT_EVERY ∈ {15, 5, 1}` (train once, 400 val series):

| `PREDICT_EVERY` | TS-AUC | est. full test (12-core) |
| --- | --- | --- |
| 15 | 0.5636 | ~30 s |
| 5 | 0.5654 | ~87 s |
| **1** | **0.5663** | **~433 s (~7 min)** |

Set `PREDICT_EVERY=1`: best score, staleness eliminated, and ~7 min at 12 cores is well
within the inference budget (faster still on 16/32-core machines via adaptive parallelism).
Model and features unchanged. Sweep harness: `_sweep_pe.py`.

**Result (2026-08-10):** **LB 55.48% — rank 289** (up from 296), 31 min on 16 vCPU.
Confirms the staleness fix helps on the real leaderboard and stays within budget.

### E7 — Train on all series (more data)

Swept training-set size / density on a fixed disjoint 400-series val set (`PREDICT_EVERY=1`,
harness `_sweep_data.py`):

| n_series | steps/series | near-τ | rows | TS-AUC |
| --- | --- | --- | --- | --- |
| 5000 | 10 | — | 49k | 0.5457 |
| **9000** | **10** | **—** | 89k | **0.5648** |
| 9000 | 20 | — | 177k | 0.5582 |
| 9000 | 40 | — | 350k | 0.5585 |
| 9000 | 20 | 0.5 | 177k | 0.5486 |

(Absolute values differ from E6's table — different held-out val set; compare only within
this table.) **More *series* helps a lot** (+0.019); **more *steps per series* hurts**
(correlated, redundant samples skew the fit); **near-τ densification hurts** (ambiguous
boundary points confuse the classifier). Change: `MAX_TRAIN_SERIES` 6000 → 10000, keep
`STEPS_PER_SERIES=10`. Train-time only; infer cost unchanged.

**Result (2026-08-10):** **LB 55.94% — rank 268** (up from 289). Cloud trained on 9934
series / 98.8k rows; adaptive parallelism selected 15 workers on the 16-vCPU box;
determinism check passed; 42 min. Local +0.019 → LB +0.46 pt.

### E8 — Recent-vs-older split features

Every base feature contrasts the current window against the *full history*. That baseline
is sluggish and can't localize a break that happens **inside** the online stream. Added 3
features that split the current window into an older half and a recent half and contrast
them directly (standardized mean shift, log-variance ratio, KS) — **history-independent**,
firing exactly when the window straddles the break.

Feature A/B on a fixed disjoint 400-series val set (harness `_sweep_feats.py`, PE=5):

| feature set | n_feat | TS-AUC | Δ |
| --- | --- | --- | --- |
| base | 14 | 0.5370 | — |
| base+short (last-25 scale) | 17 | 0.5410 | +0.0040 |
| **base+split (recent-vs-older)** | 17 | **0.5430** | **+0.0060** |
| base+short+split | 20 | 0.5422 | +0.0052 |

Split is the biggest single win and short *doesn't stack* on top of it (overlapping
"recent change" signal), so we kept split only. Confirmed at the submission setting
(PE=1, `_confirm_split.py`): base 0.5380 → base+split **0.5441** (+0.0061), infer +22%
(551 s → 672 s on 400 val series, 1 core) — still within budget. Feature count 14 → 17.

**Result (2026-08-10) — REGRESSION, REVERTED:** on the leaderboard this scored **55.70%
(rank 269)**, *down* 0.24 pt from E7's 55.94%. The +0.006 gain on a single 400-series val
set did **not** generalize — it reversed sign on the full 10000-series test. Reverted to the
14-feature set. **Lesson:** a single 400-series local split is too noisy to trust a ~0.006
delta; only act on feature changes with a large margin or multi-fold / larger-val evidence,
and treat the leaderboard as the source of truth for small deltas.

### E9 — Robust / rank-based statistics

Step 4 of the roadmap: replace the moment-based contrasts with **outlier-resistant**
restatements, in case heavy tails were distorting `mean_z` / `log_var_ratio`. Four
candidates, each = base14 + the new feature(s), judged on the multi-fold harness
(`_val_harness.py`, 5×400 disjoint folds, 6000-series pool), paired against base on the
same folds:

| candidate | added feature(s) | mean Δ vs base | t | verdict |
| --- | --- | --- | --- | --- |
| `rob_loc` | median-z, log-MAD-ratio | +0.0002 | +0.19 | noise |
| `mwu` | Mann–Whitney rank-location z | −0.0004 | −0.47 | noise |
| `bf` | Brown–Forsythe robust scale | +0.0006 | +0.18 | noise |
| `rob_all` | all four combined | −0.0032 | −0.88 | regression |

Every delta sits inside the noise floor (base fold std **0.0125**); the best `t` is 0.19,
nowhere near the `|t| ≥ 2` bar. Conclusion: the existing KS + `mean_z` + `log_var_ratio`
already capture location/scale/distribution shift, and the robust versions are redundant
rather than additive. **Not shipped** — the harness rejected all four locally, at zero
leaderboard cost. Notebook stays at the E7 14-feature config.

```bash
# quick single-split validation harness (train + streaming TS-AUC + speed)
.venv/bin/python -u _exp_fast.py

# robust multi-fold harness: per-candidate mean+-std across 5 folds + paired t vs base
.venv/bin/python -u _val_harness.py
```

### E11 — Window geometry (`WINDOW` 75 → 130)

Step 5 of the roadmap: the trailing-window length is a near-free knob (the window is
maintained in O(1), so inference cost is unchanged). A proxy sweep (row-level GroupKFold
AUC) rose monotonically 0.544 (w40) → 0.568 (w75) → 0.594 (w130) → plateau ~160–250, but
the proxy overstates large-window benefit; streaming TS-AUC confirmation is the arbiter.

- **5-fold streaming:** WINDOW=130 mean 0.5563, Δ **+0.0051**, t=**1.73** (all 5 folds
  positive) — promising but borderline.
- **10-fold streaming** (overnight, higher power): WINDOW=130 vs 75, Δ **+0.0110**, t=**+3.96**.

Clears the `|t| ≥ 2` bar decisively. **Shipped** (`WINDOW = 130`): **LB 56.63%** (rank 235),
up from E7's 55.94% (rank 268) — **+0.69 pt, +33 ranks**, at ~identical infer cost. The
biggest single jump so far, and a clean confirmation that the 10-fold harness (local +0.011)
translates to real LB gains. (Note: the first WINDOW=130 attempt reverted to 75 before
submission and reproduced E7's 55.94% exactly — a useful reproducibility check.)

### E12 — Overnight feature battery (40 candidates)

Unattended run (`_overnight.py` + `_feat_lib.py`, ~3.4 h): base14 vs base14 + each of 34
feature groups + 6 combos, at WINDOW=75 on the same 5×400 folds as `_val_harness` (base14
reference reproduces 0.5511 exactly), paired t per candidate, crash-safe incremental TSV.

Ship-grade (`|t| ≥ 2`, Δ > 0):

| candidate | added feature(s) | Δ vs base | t |
| --- | --- | --- | --- |
| `runs` | # runs about the median (normalized) | +0.0028 | +2.48 |
| `combo_volatility` | abs-change, realized-vol ratio, curvature, down/up vol | +0.0050 | +2.40 |
| `spec_entropy` | spectral entropy of the window | +0.0054 | +2.39 |

Promising (1.5 ≤ t < 2): `perm_entropy` (1.85), `median_cross` (1.84), `longest_run`
(1.69), `rv_ratio` (1.62). Everything else (distance/robust/cusum/wasserstein/…) was noise
or negative — reinforcing E9 (distance/robust restatements are redundant with KS/`mean_z`/
`log_var_ratio`). Winners **cluster by family** (run/level-crossing; complexity/spectral),
which argues real signal over the ~2 false positives expected from 40 tests.

Not yet shipped: the battery ran at WINDOW=75, so the top features must be **re-confirmed at
WINDOW=130** (10 folds) before adding, since features can interact with window length.

### E13 — Feature re-confirmation at WINDOW=130

The E12 winners were ranked at WINDOW=75; since features can interact with window length,
each was re-tested at the shipped WINDOW=130 over 10 folds (`_confirm_feats130.py`,
base14@130 = 0.5670 reference, paired t):

| candidate | added feature(s) | Δ vs base | t | verdict |
| --- | --- | --- | --- | --- |
| `stack_runs_spec` | runs + spec_entropy | +0.0071 | +3.49 | SHIP |
| `spec_entropy` | spectral entropy of the window | +0.0050 | +4.67 | **SHIP** |
| `perm_entropy` | permutation entropy | +0.0022 | +1.93 | promising |
| `runs` | # runs about the median | +0.0011 | +0.81 | noise |
| `combo_volatility` | abs-change / rv-ratio / curvature / down-up vol | +0.0008 | +0.41 | noise |
| `rv_ratio` | realized-vol ratio | −0.0001 | −0.12 | noise |

**Two of the three W75 "winners" collapsed at W130** — `runs`-alone (t=2.48→0.81) and
`combo_volatility` (t=2.40→0.41) were window-specific artifacts, vindicating the
re-confirmation step. Only **`spec_entropy` survived and strengthened** (t=2.39→4.67).
`stack_runs_spec` has a higher mean, but its extra +0.0021 comes entirely from `runs`
(noise on its own), so shipping it risks an E8-style reversal. **Shipped `spec_entropy`
alone** as the 15th feature (`compute_features` 14→15): **LB 56.84%** (rank 229), up from
E11's 56.63% (rank 235) — **+0.21 pt, +6 ranks**. `+runs` on top queued as a follow-up test.

```bash
# quick single-split validation harness (train + streaming TS-AUC + speed)
.venv/bin/python -u _exp_fast.py

# robust multi-fold harness: per-candidate mean+-std across 5 folds + paired t vs base
.venv/bin/python -u _val_harness.py

# overnight feature/window battery: writes _overnight_results.tsv + _overnight_summary.txt
.venv/bin/python -u _overnight.py
```

In the notebook: run all model cells, then `crunch_tools.test()`, then the TS-AUC cell,
then `crunch_tools.submit()`.

## History / provenance

Records before this file existed were reconstructed from the assistant's conversation
summary and the local session store (`/chronicle`), which retains past VS Code Copilot
sessions for this workspace (setup + baseline submission on 2026-08-07, improvement work
following). There is no automatic persistent memory across chats — going forward, each
experiment will be appended to the table above.
