# Backlog (Post-63.17 Freeze)

Created: 2026-09-11

## Deferred Items

1. Full-buffer global tau locator
- Status: whitened GLOC selected as local candidate (2026-09-13)
- Reason: higher-plumbing-risk change; not a deadline-safe tweak.
- Scope: causal full-prefix scan for tau-hat at each online step, then compact readout features.
- Screen `{42,7,123}` vs base79: isolated `gloc` mean `+0.00084`, min `+0.00064`, 3/3; `czrb_gloc` mean `+0.00260`, min `+0.00217`, 3/3.
- Correct incremental comparison vs `czrb_all79`: `czrb_gloc` mean `+0.00138` (seed deltas `+0.00186,+0.00049,+0.00179`), 3/3 but below the `+0.0015` promotion gate. It is also not a comparison against the current WBLOCK champion.
- The fresh `{31337,2025,555,99}` run was stopped before its first cache completed because it would have tested the known calibration+GLOC bundle against base79, not GLOC against the current calibration+WBLOCK champion.
- Frozen local candidate: `backups/_calib_rankerrf.LOCAL_GLOC_20260912_193234.py` plus matching seed logs/raw/summary files under the same stamp. The frozen 63.17 notebook and model artifact were not modified.
- Six-arm combined refinement screen: `GLOC6=1 WBLOCK=1 BASELINE=czrb_all79,wblock`. One cache contains all candidate columns and preserves identical sampled rows; model fits remain separate. Arms are raw, whitened likelihood, dual/agreement, asymmetric bounds, variance-selected, and multi-horizon GLOC.
- Full command after smoke: `nohup env PYTHONWARNINGS=ignore GLOC6=1 WBLOCK=1 BASELINE=czrb_all79,wblock N_JOBS=15 .venv/bin/python -u _calib_rankerrf.py > _calib_gloc6.out 2>&1 < /dev/null &`.
- Full screen launched 2026-09-12 19:38:46 as PID `31335`; PID file `_calib_gloc6.pid`, log `_calib_gloc6.out`, cache `_calib_cache_gloc6_wblock`. Startup passed affine check at `4.47e-07`; disk had 62 GB free.
- Final screen: `white_gloc` won with absolute mean TS-AUC `0.636735`, mean delta `+0.002559`, min delta `+0.001342`, 3/3 positive. Raw GLOC was runner-up at `+0.002268`; dual `+0.001808`; variance-selected `+0.001503`; horizon and asymmetric failed 3/3.
- Winner frozen under `LOCAL_WHITE_GLOC_20260913_061257` (harness, log, raw rows, summary). Completed 12 GB cache deleted; 62 GB free afterward.
- Post-hoc combination screen launched 2026-09-13 06:20:38 as PID `61743`: `white_var`, `white_raw`, and `white_raw_var`, each compared directly with white GLOC on `{42,7,123}`. Log `_calib_gloc_combo.out`, PID file `_calib_gloc_combo.pid`; no different-seed stage.
- Original combination process stopped after seed 123 completed `white_var` and `white_raw`, with no traceback. Partial log preserved as `_calib_gloc_combo_partial.out`. Targeted resume for only missing seed-123 `white_raw_var` launched 2026-09-13 10:17:54 as PID `17562`, log `_calib_gloc_combo_resume.out`, reusing the intact cache.
- Recovery correction: PID `17562` was mistakenly stopped and its cache deleted while switching to the user-requested fresh white-vs-white_raw comparison. That fresh run was immediately paused (startup log `_calib_white_raw_fresh_paused.out`), and the original missing seed-123 `white_raw_var` job was restored as PID `18608` at 10:30:16 (`_calib_gloc_combo_resume2.out`). It must rebuild seed 123 because the cache was deleted.
- Combination screen complete: versus white GLOC, `white_var` mean `-0.000988` (1/3), `white_raw` `+0.000187` (1/3), and `white_raw_var` `-0.001281` (1/3). White GLOC remains the local winner. Next, per explicit user request, compare white vs white_raw on disjoint seeds `{31337,2025,555,99}`.
- Complete combination evidence frozen under `GLOC_COMBOS_COMPLETE_20260913_131052`; completed cache removed. User-requested disjoint-seed white-vs-white_raw run launched 2026-09-13 13:10:53 as PID `25013`, log `_calib_white_raw_fresh.out`, PID file `_calib_white_raw_fresh.pid`.
- Explicit disjoint-seed result: `white_raw` vs white GLOC deltas `-0.000901,+0.000698,-0.000722,-0.001072`, mean `-0.000499`, min `-0.001072`, 1/4. Reject every tested combination; final local candidate is white GLOC alone.
- Tracking files: `_calib_gloc_seed42.out`, `_calib_gloc_resume.out`, `_calib_gloc_fresh.out`, `_calib_gloc_fresh.pid`, `_calib_rankerrf_raw.tsv`, `_calib_rankerrf_summary.txt`.

2. Independent feature banks
- Status: deferred
- Reason: larger architecture/dataflow change; better done after deadline.
- Scope: train separate members on orthogonal banks (raw base bank vs other bank), then blend with fixed CDF mapping first; postpone learned stacking.

3. Cross-series context at step t (within-step ranking helper)
- Status: deferred
- Reason: high determinism risk in current parallel infer setup.
- Scope: add causal cross-sectional context signals computed from series alive at the same absolute step t.
- Guardrails: must be order-invariant across workers/series and pass strict determinism checks before any submission use.

4. Post-white-GLOC ensemble refinement
- Status: completed 2026-09-14 at 06:51:49. The combined local candidate passed the screen gate; deployment into a backed-up working notebook, artifact regeneration, parity checks and any leaderboard run remain separate user-approved work. Harness: `_post_white_pipeline.py`; console log: `_post_white_pipeline.out`; detailed log: `_post_white_pipeline.log`; JSON: `_post_white_pipeline.json`; final Markdown: `_post_white_pipeline_summary.md`; checkpoints: `_post_white_pipeline_cache/`.
- Baseline: frozen 63.17 four-member fixed-CDF ensemble. Candidate feature set is calibration + WBLOCK + white GLOC (173 columns). Do not edit the champion before backup/freeze.
- Step 1 — true four-member attribution: fit each member on 166-column calibration+WBLOCK and 173-column calibration+WBLOCK+white-GLOC inputs using identical sampled rows. Score all 16 assignments of white GLOC across ranker, random forest, XGBoost, and CatBoost from cached member percentiles. This tests whether feeding GLOC to every member dilutes its value and directly explores independent feature banks.
- Step 2 — finalist randomness: retain only the baseline, best hyperparameter candidate per member, and a near-tied runner-up when justified. Pair baseline/candidate on identical rows for data seeds `{42,7,123}` and model seeds `{42,7,123,2024,31337}`. Separate data/fold/sampling variance from model-internal variance; do not open a new data-seed confirmation stage automatically.
- Step 3 — fixed-CDF weight retune: after feature allocation and member parameters are fixed, search a bounded simplex around `ranker=.45, random forest=.40, XGBoost=.10, CatBoost=.05` at 0.025 resolution using cached predictions. Judge the ensemble delta, not standalone member accuracy.
- Step 4 — optional seed averaging: only test 2–3 seeds for a member if Step 2 shows material model variance and averaging improves the whole ensemble. Account for random-forest ONNX size and inference cost.
- Step 5 — final candidate: combine the winning GLOC allocation, robust hyperparameter changes, and fixed-CDF weights; run one exact full-recipe paired comparison against the untouched 63.17 champion, freeze notebook/model/evidence, and leave leaderboard submission to the user.
- Fallback — locator specialist: if Steps 1–5 do not produce a robust candidate, train a compact member on the seven white-GLOC features plus a small calibrated break-strength subset and test fixed weights `0.025–0.10`. This is preferred over another broad feature grid or learned meta-stacker.
- Promotion rule: lead with absolute TS-AUC; require ensemble `mean_d >= +0.0015` and positive on all declared screen seeds, with aggregate and per-seed t scores reported.
- Scheduled implementation: run Steps 1, 2, 3 and 5 sequentially, omitting optional seed averaging (Step 4). Step 2 uses model seeds `{42,7,123,2024,31337}` and carries forward only parameter changes whose model-seed mean is positive for every data seed. Step 3 evaluates 303 bounded fixed-CDF weight vectors at 0.025 resolution. Step 6 runs only if Step 5 misses the promotion gate, using a 27-column specialist bank (white-GLOC7 + WBLOCK8 + robust-calibrated mean/variance/KS/IQR signals at three scales) at weights `0.025,0.05,0.075,0.10`.
- Result: Step 1 selected white GLOC for the ranker and random forest only (absolute TS-AUC `0.637092`, mean delta `+0.002871`, min `+0.001667`, 3/3). Step 2 accepted ranker `n_estimators=500, learning_rate=.03` and CatBoost `depth=8`; XGBoost `depth=8` was not robust across data seeds. Step 3 selected weights `ranker=.55, random forest=.30, XGBoost=.05, CatBoost=.10`. Step 5 combined absolute TS-AUC `0.637920`, mean delta `+0.003699`, min `+0.002889`, 3/3, aggregate t `+3.43`; promotion gate passed. Step 6 was skipped by design. No notebook edit or submission was made.

5. Methodology inventory and genuinely open probabilistic/pretrained-model work
- Status: research backlog after post-white-GLOC ensemble refinement; do not run concurrently with `_ensemble_hp_white_gloc.py` or another 15-job harness.
- Category 1 — classical statistical inference / hand-engineered detectors: **extensively tested; partly core to the champion.** Includes location, scale, distribution, dependence, entropy, CUSUM, sup-F/Chow, MOSUM, robust and two-sample tests. New restatements are low priority because most were redundant, while the useful multi-scale/residual/calibrated forms are already represented.
- Category 2 — online unsupervised sequential change-point detection: **extensively tested; generic methods weak, directed localization still productive.** CUSUM, Page-Hinkley, Gaussian constant-hazard Bayesian online change-point detection, sequential feature banks and local split scans failed to add robust lift. Full-prefix history-whitened GLOC is the important exception and is the current local feature candidate.
- Category 3 — supervised incremental feature extraction plus classifier/ranker: **core successful methodology.** Training rows are labeled `(series,time step)` pairs with `y=1[tau>=0 and t>=tau]`; the current ranker, random forest, XGBoost and CatBoost ensemble is this category. This remains the primary optimization path.
- Category 4 — probabilistic/generative and Bayesian state-space inference: **moderately explored, with room for narrower new work.** Tested forms include standalone Bayesian online change-point detection, a conjugate one-change Bayes-factor/MAP-location/posterior-entropy bank (`mean_d=-0.0013`, 0/3), hazard targets and Kalman/particle-style features. Do not repeat those formulations unchanged.
- Category 5 — supervised deep sequence/distribution learning: **extensively tested.** CNN, multiscale CNN, GRU, TCN, ModernTCN, Hyena, S4D-lite, set-transformer, input-lens, capacity and synthetic-pretraining probes were null or negative as ensemble additions. Synthetic pretraining is in this category, not the foundation-model category.
- Category 6 — pretrained time-series foundation models / transfer learning: **not materially tested.** No Chronos, TimesFM, Moirai, Lag-Llama, TimeGPT or comparable externally pretrained time-series model has been evaluated. This is the only one of the six broad categories that remains genuinely untested.

### Bayesian / probabilistic follow-ups

Run these in priority order and stop the branch when the probability model has no useful standalone or orthogonal signal. All implementations must be causal and affine-invariant, preserve the declared screen rows, and be evaluated as additions to the current four-member ensemble rather than against an old base.

1. **Bayesian full-prefix GLOC posterior (highest priority; prepared 2026-09-14, not launched).** `_bayes_white_gloc.py` is a guarded launcher: no arguments prints the plan and performs no computation; `--smoke` and `--full` are explicit opt-ins. `_calib_rankerrf.py` now supports `BAYES_GLOC=1` and appends a fixed eight-column posterior block to the current `base79 + robust calibration79 + WBLOCK8 + white-GLOC7` baseline. The posterior uses the same history-whitened Gaussian split likelihood as white GLOC, a fixed break prior of 0.5 spread uniformly over admissible locations, and a two-parameter Bayesian information criterion penalty. Readouts are break probability, conditional posterior mean and maximum-a-posteriori location, location standard deviation, 90% credible width, normalized entropy, top-two-mode gap, and posterior-averaged standardized effect. Compile, finite/range, shape, short-prefix and affine checks passed; no cache, model fit, screen or submission was started. Stage A is the declared ranker/random-forest screen on `{42,7,123}` with unchanged sampled rows. Only if it clears `mean_d >= +0.0015` and 3/3 positive should Stage B test member allocation in the full four-member E116 ensemble; do not integrate it into the notebook before that exact ensemble test passes.
2. **Heavy-tail and dependence-aware likelihoods.** Compare a small predeclared set of likelihoods on whitened innovations: Gaussian control, Student-t robust emissions, and an AR-coefficient/regime likelihood if it can reuse existing fitted references cheaply. Test posterior/model evidence as a seven-to-ten-column block; avoid a broad likelihood grid.
3. **Bayesian model averaging over change families.** Combine posterior evidence for mean, variance and dependence changes using fixed priors learned only from training data. Emit family posterior probabilities and posterior-averaged break probability. The value proposition is calibrated uncertainty and family competition, not another raw statistic bank.
4. **Probabilistic specialist member.** Only if one of Steps 1–3 has standalone TS-AUC materially above chance and an orthogonal residual signal, train a compact classifier/ranker on its posterior summaries plus a small calibrated feature subset and blend it at weights `0.025–0.10`.
5. **Kill conditions.** Stop if the posterior block misses `mean_d >= +0.0015` or is not positive on all `{42,7,123}` screen seeds, or if per-point inference cost threatens the cloud budget. Do not automatically open a different-seed confirmation stage.

### Foundation-model follow-ups

Treat this as a feasibility-first transfer-learning experiment, not a full fine-tuning campaign. Most available foundation models are forecasting models, while the objective here is causal within-step structural-break ranking.

1. **Model shortlist and compatibility gate.** Select at most two open, locally downloadable models with offline weights, variable-length causal inputs and a usable hidden representation. Reject any model that requires future context, internet access at inference, an unshippable license/artifact, or excessive CPU latency. Prefer a small encoder-style model; include one forecasting foundation model only if hidden states are accessible.
2. **Frozen-embedding screen.** History-normalize each causal trailing window/prefix, extract frozen embeddings offline on the exact sampled rows, reduce them to a compact fixed representation, and train a lightweight pointwise classifier or ranker. No end-to-end fine-tuning in the first test.
3. **Orthogonality diagnostic before blending.** Report standalone absolute TS-AUC, correlation with each champion member, residual/conditional signal after the four member scores, artifact size and milliseconds per online point. Continue only if the embedding contains measurable residual signal; standalone weakness alone is not a rejection if orthogonality is real.
4. **Ensemble test.** Add the embedding model as a fixed-CDF specialist at weights `0.025–0.10` and evaluate on `{42,7,123}` with identical rows and folds. Apply the usual ensemble gate (`mean_d >= +0.0015`, 3/3 positive).
5. **Fine-tuning only after a frozen win.** If frozen embeddings pass, test parameter-efficient fine-tuning or a small projection head. Full-model fine-tuning is out of scope unless it clearly fits memory, artifact and inference budgets.
6. **Deployment gate.** Bundle weights because the cloud has no internet; verify license/provenance, submission size, deterministic inference within `1e-8`, causal-prefix behavior and streaming latency before touching the champion. Keep the 63.17 notebook frozen and leave submission to the user.

6. Time-regime and trajectory diagnostics
- Status: queued 2026-09-13 as PID `72526`, waiting on tuning PID `41864` before it consumes compute. Log: `_eda_time_regimes.out`; outputs: `docs/time_regime_eda/`.
- Scope: evaluate the untouched 166-feature, four-member champion on real seed-42 held-out series at every online step. Report absolute TS-AUC by post-break age, early-alarm behavior before future breaks, detection delay, post-detection decay, and score jump/plateau statistics.
- Produce a deterministic eight-series gallery covering clean early detection, delayed detection, missed break, transient/decaying detection, pre-break false alarm, quiet no-break, no-break false positive, and the jumpiest remaining trajectory. Show the raw online series, predicted score, ideal step label, and true tau.
- Interpretation rule: visual smoothness is not an objective. Only pursue temporal summaries such as exponentially weighted evidence, persistence, peak, or slope when this EDA isolates a repeatable weak regime; evaluate any change by within-step cross-series TS-AUC because smoothing can delay detection or damage ordering.
- Follow-up rule: use the diagnosed regime to motivate one compact specialist or feature block, not a broad temporal grid. Preserve exact rows and compare against the current champion under the normal `{42,7,123}` screen and promotion gate.

## Executed Now

- Additive whitening: kept as active path in current stack4 winner via additive whitened readout block (not wholesale stream replacement).
