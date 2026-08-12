# structural-break2026 (CrunchDAO real-time)

## Conventions
- Document EVERY experiment in `README.md` (experiment-log table). Keep concise, no bloat.
- Append new runs to the table; record tests/observations that informed decisions.

## Environment
- Python venv: `.venv/bin/python` (system `python` not on PATH). Use absolute path.
- Available: scipy, sklearn, numpy, pandas, joblib, crunch. NOT: lightgbm, nbformat.
- Terminal sandbox sometimes strips `cd`; use absolute interpreter path.

## Key facts
- Metric = TS-AUC, cross-sectional per online step. Accumulating scores gain nothing.
- infer() is a generator: one yield per online point, x_online iterated once (no batching).

## E14 (2026-08-11): ship perm_entropy (order-3) as 16th feature — LB 56.75%, REVERTED.
- RESULT: LB 56.75% rank 229 = -0.09pt vs E13 (56.84%), same rank. Sub-threshold bet
  (t=+1.93 vs base14) did NOT transfer -> E8-style reversal. REVERTED notebook to 15 feats
  (E13 champion) via programmatic json script (_revert_cell.py, deleted). Champion notebook
  is now E13 (base14+spec_entropy, W130). Did NOT resubmit E13 (known 56.84%, +0.09 not worth
  a slot); keep 15-feat locally as base for Lever B.
- User rule: t>=1.9 is good enough to TEST on LB (features are hard to find; interactions
  discovered at ship, not pre-stacked). NO STACKING until single features exhausted.
- perm_entropy inline matched _feat_lib.f_perm_entropy exactly (diff 0.0) before revert.
- GOTCHA HIT: editing the .ipynb via replace_string_in_file CORRUPTED the JSON (return-list
  lines merged into inserted lines). FIX: wrote a python script that json.load's the nb,
  finds the cell, replaces cell["source"]=to_source_lines(text), json.dump(indent=1). For
  multi-line notebook cell edits prefer this programmatic approach over string-replace.
- E14 spectral sweep (_explore_spectral.py) ~9 min/eval (NOT 55s), ~2.6h total. Early:
  base15 mean 0.5720; spec_flatness noise (t=-0.73); spec_rolloff noise (t=+0.38).
- E14 KILLED early (user, 2026-08-11) after spectral-SHAPE family, to prioritise Lever A.
  FEATURE INFO (marginal over spec_entropy @W130, 10 folds, base15 mean 0.5720):
  spec_flatness d=-0.0013 t=-0.73; spec_rolloff d=+0.0005 t=+0.38; spec_spread d=-0.0021
  t=-1.04; spec_peak d=+0.0004 t=+0.21 — ALL NOISE. Spectral-SHAPE family is dead on top of
  spec_entropy. NOT tested (queued if we revisit features): spec_entropy_diff,
  spec_entropy_short, perm_entropy4, higuchi, katz, lzc, hjorth, spec_bands, combos.

## LEVER A RESULT (2026-08-11): NEGATIVE — per-series calibration does NOT help. ABANDON.
- Harness _lever_a.py: fit shipped base16 once, stream each fold for RAW per-(series,step)
  scores, re-score TS-AUC under per-series transforms (sub_b10/20/40 = subtract early-window
  median baseline; z_b20/40 = /MAD; exp_med = subtract causal expanding median).
- FULL 10-fold @W130: base16 raw mean=0.5742 beats EVERY variant; best (sub_b10)
  d=-0.0133 t=-2.49; sub_b20 t=-3.20; sub_b40 t=-3.77; z/exp down to t=-6.49. All
  significantly NEGATIVE. Smoke (2 folds) identical direction. (Also cross-checks
  perm_entropy: base16 0.5742 vs E14 base15 0.5720 => +0.0022 on this pool.)
- WHY (key insight): the model is trained on POOLED cross-series labeled data, so
  predict_proba is ALREADY a cross-series-comparable break probability. A series sitting at a
  high raw score IS genuinely more break-like — that level is SIGNAL, not nuisance.
  Subtracting each series' own baseline DESTROYS that shared calibration (and early baselines
  are contaminated when tau is small). Rank-invariance still holds (uniform per-step
  transforms are no-ops), but per-series de-meaning removes real signal.
- IMPLICATION: under TS-AUC the only way to gain is to improve the WITHIN-STEP ORDERING of
  series = make the RAW signal sharper/earlier. => go to Lever B (within-online changepoint
  localisation: max-split CUSUM/KS/LR over candidate split points in the online history).
  Cross-series state is NOT a useful lever here (post-hoc recalibration can't beat the pooled
  model's ordering). Do NOT revisit A.

## LEVER B (2026-08-11): within-online max-split changepoint features — NEGATIVE. NOT shipped.
- Full 10-fold @W130 on top of base15 (mean 0.5720): max_split_var d=+0.0014 t=+0.95;
  combo_maxsplit +0.0002 t=0.12; max_split_lr +0.0000 t=0.01; max_split_pos -0.0006 t=-0.64;
  max_split_t -0.0020 t=-1.37. ALL NOISE. (55 min run.)
- SMOKE t=+4.34 for max_split_t was a 2-FOLD ARTIFACT -> collapsed to t=-1.37 on 10 folds.
  LESSON: never trust 2-fold smoke deltas; smoke is a plumbing check only.
- WHY: the 130-window vs-history features (mean_z/log_var_ratio/KS/skew/kurt...) already
  capture the mean/var shift the break induces; a self-normalized within-window supremum
  statistic is redundant + noisier. Changepoint-as-feature is dead (cusum E12 W75 + max-split
  E15 W130 both null). Do NOT revisit changepoint features.
- Feats live in _feat_lib.py (max_split_t/lr/var/pos, _msplit_prefix) + harness _lever_b.py.

## LEVER B design details (features/harness, for reference; result NEGATIVE above)
- Idea: scan candidate split points INSIDE the trailing window (self-normalized within window,
  NOT vs history mean like cusum_max) -> max standardized contrast between the two sides.
  Fires when a break lies INSIDE the recent window -> complements vs-history features. Vectorized
  via prefix sums (O(n)/window), MIN_SEG=5 guard.
- New feats in _feat_lib.py: max_split_t (Chow/max-t mean contrast), max_split_lr (Gaussian
  max log-LR mean+var), max_split_var (max |log var ratio|), max_split_pos (k*/n position).
  Registered in EXTRA. _msplit_prefix() shared helper.
- Harness _lever_b.py (clone of _explore_spectral.py): each STANDALONE on top of base15
  (spec_entropy) @W130, 10 folds, + combo_maxsplit. Ship |t|>=2 & d>0. Outputs
  _lever_b_results.tsv/_lever_b_summary.txt/_lever_b.log.
- SMOKE (2 folds): base15 0.5485; max_split_t d=+0.0061 t=+4.34 SHIP(!); max_split_lr t=0.19
  noise. Prime candidate = max_split_t. Full run launched PID 150659 ~50 min.

## NEXT BIG LEVERS QUEUE (return here after feature sweep — user endorsed 2026-08-11)
- STRATEGIC READ (2026-08-11): 56->64 gap is ~8pt = NOT closeable by feature nibbles (~0.2pt
  each, families exhausted). Leverage is GEOMETRIC/DATA (WINDOW 75->130 gave +0.69pt biggest
  jump; more series E7). Likely leaders use different representation: multi-scale windows OR
  full-online-history-vs-history (growing two-sample test) instead of fixed 130 trailing win.
- E16 WINDOW re-sweep: DONE, NEGATIVE (W=130 optimal). 10 folds base15 paired vs W130
  (mean 0.5720): W160 +0.0016 t=+0.54 noise; W180 -0.0042 t=-1.81; W200 -0.0018 t=-0.58;
  W250 -0.0011 t=-0.40. 130 confirmed at plateau top; NO larger window ships. Do NOT revisit
  single-window length.
- E17 MULTI-SCALE: DONE. WINNER=ms_60_130_250 (Δ+0.0075 t=+3.06 SHIP), strongest of campaign.
  Full 10-fold vs base15@130 (ref 0.5720): ms_60_130_250 (45f,W250) 0.5795 t=+3.06 SHIP;
  ms_40_130_250 (45f,W250) 0.5796 t=+2.27 SHIP; ms_40_130 (30f,W130) 0.5739 t=+2.02 SHIP;
  ms_130_250 (30f,W250) 0.5761 t=+1.45 noise; ms_20_60_130 (45f,W130) 0.5714 t=-0.60 noise.
  WIRED into streaming_detector.ipynb: added compute_features_ms(w,ctx) wrapper (concat base15
  over nested trailing w[-s:] for s in MS_SCALES, guarded s>=n reuses full w); config WINDOW=250,
  MS_SCALES=(60,130,250); train/infer call compute_features_ms; MS_SCALES saved+read in cfg.
  Verified: 45 feats, all finite, short-window guard OK. Patch via _patch_ms.py (json load/dump,
  anchors end WITHOUT trailing \n on the `    ]` line -- gotcha). LB RESULT: 57.50% rank 214
  (was 56.84%/229) = +0.66pt +15 ranks, biggest jump since WINDOW=130. CHAMPION CONFIRMED.
- E18 FULL-ONLINE-VS-HISTORY: DONE, NEGATIVE (all noise). 10 folds paired vs champion ms_60_130_250
  (ref 0.5795): ms_60_130_250_full (60f) +0.0035 t=+0.51 (raw mean 0.5829 highest but fold-inconsistent);
  ms_60_130_full (45f) -0.0007 t=-0.13; ms_130_full (30f) -0.0048 t=-0.72. Unbounded online view adds NO
  reliable signal on top of trailing 60/130/250 -- growing full-online contrast dominated by stale
  pre-break points; 250 window already captures post-break. FULL-ONLINE LEVER DEAD, do not revisit.
- E19 SCALE-SET SWEEP: RUNNING (PID 177494, launched 20:10). _scalesweep.py -> _scalesweep_summary.txt.
  Multi-scale is the proven winning representation (E17 LB +0.66pt); only 5 sets tried. Baseline =
  champion ms_60_130_250 @250. Candidates (paired vs champion): ms_40_60_130_250, ms_60_90_130_250,
  ms_60_130_190_250, ms_60_100_160_250, ms_40_90_160_250 (all @250), ms_60_130_250_350 (@350).
  Tests: does a 4th trailing scale / different spacing / longer top beat 60/130/250? SMOKE OK.
- NOTEBOOK STATE (2026-08-11): champion=MS_SCALES(60,130,250) LB 57.50%/214, REVERTED back to champion.
  PROBE ms_40_130_250 LB = 57.38%/216 (sub #14) = WORSE -> short-60 beats short-40 on LB. Champion
  (60,130,250) CONFIRMED best. Do not use short-40.
- E19/E20 STATUS: E19 full DONE (all noise, see below). E20 maha t=2.78 SHIP was a 2-FOLD SMOKE
  ARTIFACT = NOISE, DO NOT TRUST; wait for real 10-fold _distances_summary.txt.
- *** WAITER SELF-MATCH BUG (2026-08-11, cost ~56min) ***: the E19->E20 auto-chain waiter used
  `while pgrep -f _scalesweep.py; do sleep 30; done`. `pgrep -f` matches the FULL command line of
  ALL procs INCLUDING THE WAITER ITSELF (its cmdline literally contains "_scalesweep.py") -> loop
  never exits -> E20 NEVER launched (no _distances.log). FIX = bracket trick: `pgrep -f "[_]distances\.py"`.
  The regex `[_]distances\.py` matches the real `python _distances.py` proc but NOT the waiter's own
  cmdline (which contains the literal text "[_]distances\.py", no bare "_distances.py"). ALWAYS use the
  bracket trick in chain waiters. Verified: only the real python PID matches.
- E20/E21 RELAUNCH (2026-08-11): killed broken self-matching waiters. Relaunched E20 _distances.py
  DIRECTLY (full 10-fold). Fixed E21 waiter (`pgrep -f "[_]distances\.py"`) chained _twosample.py after.
- E20 RESULT (2026-08-11, full 10-fold, 90.8min): ALL NOISE vs champion 0.5795. maha d=+0.0022 t=+0.73
  (raw 0.5816 highest but fold-inconsistent); seuclid t=+0.26; family t=-0.15; qda_llr t=-0.70;
  maha_perscale t=-0.88; l1 t=-1.22; linf t=-1.89. => FEATURE-SPACE distances (anomaly of 45-dim
  vector: Mahalanobis/QDA/seuclid/L1/Linf) add NOTHING on top of champion. Joint-covariance whitening
  does not beat axis-aligned GBT splits here. E20 DEAD, do not revisit feature-space distances.
- *** WAITER ZOMBIE BUG (2026-08-12, cost the E21 overnight run) ***: E21 never launched overnight.
  Root cause: an EARLIER broken E21 waiter (bare `pgrep -f _distances.py`, self-matching) was only
  PARTIALLY killed — `kill <outer_pid>` + `pkill -P` missed the DETACHED inner `nohup bash -c` loop
  (it reparented to init, not a child of the outer). That zombie kept looping AND its cmdline contains
  bare "_distances.py", so the CORRECT new waiter (`[_]distances\.py`) matched the ZOMBIE's cmdline and
  also looped forever. LESSONS: (1) to kill a detached nohup bash loop you must target ITS pid directly
  (find via `pgrep -af _twosample|_distances`), children/pkill -P won't reach a reparented process;
  (2) NEVER leave an old bare-pattern waiter around — its cmdline poisons bracket-trick pgrep of later
  waiters; (3) after killing, VERIFY with `pgrep -af <pattern>` returns nothing before relaunching;
  (4) SIMPLEST: prefer launching the next harness DIRECTLY when the machine is free rather than chaining.
- E21 RELAUNCH (2026-08-12 08:35): killed both zombies, launched _twosample.py DIRECTLY (PID 216895,
  full 10-fold). RESULT PENDING -> read _twosample_summary.txt.
- E21 BUILT = _twosample.py: champion base15-multiscale (45f, scales 60/130/250) + NONPARAMETRIC
  TWO-SAMPLE distances (trailing sub-window vs history sample ctx["sample_sorted"], std by ctx["sd"]),
  all pure-numpy. Families: wass (W1), energy (2*int(F-G)^2), cvm (mean sq CDF gap, rank-free), mmd
  (RBF gamma=0.5 on sd-std, subsample<=96). Candidates each @250: ts_wass/ts_energy/ts_cvm/ts_mmd (48f),
  ts_all (57f), paired vs champion (45f). Unit-checked: dims OK, finite, all 4 grow sharply under +3sd
  shift (energy .004->3.80, wass .08->2.98, cvm .0003->.31, mmd .003->.96). Ship |t|>=2 & d>0.
- E19 SCALE-SWEEP RESULT: DONE (5/6, last @350 running), ALL NOISE. base 0.5795; ms_40_60_130_250 t=-0.02;
  ms_60_90_130_250 t=+0.05; ms_60_130_190_250 t=+0.03; ms_60_100_160_250 t=+0.16; ms_40_90_160_250 t=+0.01.
  => TRAILING-WINDOW-VS-HISTORY REPRESENTATION IS SATURATED. 4th scale/spacing adds nothing. Champion
  (60,130,250) stands. STRATEGIC: our whole family (features E0-14, calib A, changepoint B/E12, window
  E11/16, multiscale E17/19, full-online E18) plateaus at ~0.575 LB. Leaders 64.5% w/ FEW submissions =
  CATEGORICALLY different method. Gap is REPRESENTATION not model capacity. Next big bets: (E21) proper
  NONPARAMETRIC TWO-SAMPLE distances window-vs-history (energy dist, MMD-RBF, Wasserstein-1, Cramer-von
  Mises, Anderson-Darling) -- we only have KS+moment diffs; (E22) neural seq model over raw online (1D-CNN/
  small transformer) -- top teams "Non deterministic" hints NN. This is ADIA Lab SB dataset (public winning
  solns exist = battery of 2-sample tests + GBM).
- ARTICLE CHECK (simomenaldo Medium, classical break methods): CUSUM=E12 dead, BinSeg/Bai-Perron supF
  = max-split Chow/LR Lever B dead, PELT/multi-break = overkill (single online-vs-history). Cointegration
  N/A (univariate). Only genuinely-new untested idea = TVP/Kalman filter (param random-walk; innovation/
  prediction-error variance spikes at break) -> E20 candidate to BUILD+CV after E19.
- E-PEF PAPER (Online SB detection in financial durations, 2025): method NOT applicable (log-ACD
  duration-specific; block-bootstrap critical values = false-alarm control, irrelevant to AUC ranking).
  TRANSFERABLE NUGGET = MAHALANOBIS distance of window feature vector from no-break (label-0) distribution:
  sqrt((f-mu0)^T Sigma0^-1 (f-mu0)), whitens by JOINT covariance -> captures coordinated multi-feature
  shift that marginal diffs + axis-aligned GBT splits miss. One scalar per scale. => E20 = Mahalanobis
  joint-deviation feature on top of champion; pure-numpy, fits infer loop; BUILD+CV after E19.
  concatenated -> strictly ADDS scales to champion's 130 view (clean nested paired vs base15@130).
  Candidates: ms_40_130, ms_20_60_130, ms_130_250, ms_40_130_250, ms_60_130_250. harness window
  = max(scales). ms_feat(w,ctx,scales) in _multiscale.py.
- E18 (idea, not built): FULL-ONLINE-vs-HISTORY. Fixed 130 trailing win discards online signal
  as it grows (online up to 1000). Add feature = two-sample contrast of ENTIRE online-so-far vs
  historical (grows/sharpens post-break). Needs notebook infer to keep full online, not just
  last 130 -> bigger change. Suspected leaders' lever.
- LEVER A (do first, highest EV, confirmed legal): CROSS-SERIES SCORE CALIBRATION.
  Use forward-only cross-series state (allowed, see below) to convert each raw break-score
  into a cross-series percentile/rank (or z vs running score-distribution at comparable
  online-depth) so series are comparable at each step t. Orthogonal to features. TS-AUC is
  purely cross-sectional-rank per step, so relative comparability > absolute calibration.
  Prototype an offline harness that post-processes stored per-(series,step) scores into
  ranks and measures TS-AUC lift, before touching the notebook.
- LEVER B (second): WITHIN-ONLINE CHANGEPOINT LOCALISATION. Replace/augment fixed
  window-vs-history with a max-split statistic: scan candidate split points in online
  history, take max standardized contrast (CUSUM / likelihood-ratio / KS) between the two
  sides. Window-length-agnostic, stays elevated post-break, targets the label directly.
  Try as a feature first, then possibly as new core signal.

## METHODOLOGY FIX (2026-08-11): always test MARGINAL vs CURRENT CHAMPION, not old base.
- Ship decision variable = delta of (champion + candidate) vs champion, NOT vs base14.
- stack_runs_spec t=+3.49 was vs base14 (base WITHOUT spec_entropy). ~all of that lift is
  the spec_entropy we ALREADY ship. Marginal of `runs` over base15 ~ +0.0021 and runs-alone
  is t=0.81 noise at W130 => NOT a clean ship (re-counts banked credit + noisy add-on).
- perm_entropy t=1.93 was ALSO vs base14; its marginal over base15 may be lower (correlated
  with spec_entropy). E14 (_explore_spectral.py) re-tests it correctly (on top of
  spec_entropy). Ship only if marginal |t|>=2 & delta>0 vs base15.

## EFFICIENCY WORKFLOW (user wants overlap, no idle time)
- PIPELINE: while user submits a shipped candidate to cloud, agent launches the NEXT
  exploration job so wall-clock overlaps. Keep an experiment queue.
- TWO-STAGE VALIDATION: fast 5-fold SCREEN to rank many candidates cheaply, then 10-fold
  CONFIRM only for finalists. Don't run big 10-fold batches for low-prior features.
- Prefer ONE targeted paired test (~10 min, 1 eval x10 folds) over 2.5h batch sweeps when a
  single question needs answering (e.g. base15+runs vs base15 to settle stack_runs_spec).

## Forum findings (2026-08-11, forum.crunchdao.com/c/competition/adia-lab/6)
- CROSS-SERIES STATE ALLOWED (staff enzo confirmed): infer() may keep a forward-only
  summary of ALREADY-COMPLETED test series and use it when scoring later series. Detect
  series end after the inner `for point in x_online` loop, record summary there. Must not
  use labels/future points/current-series final horizon; never revise past yields. Series
  order is deterministic but NOT guaranteed (code must work under random order too; OOS run
  once on new data). => LEVER: cross-series score calibration/normalisation (convert raw
  break-scores to cross-series percentile/rank to make series comparable at each step t).
- ONLINE LENGTH N NOT ACCESSIBLE by design (staff): cannot len(x_online); phase features
  t/N are OFF-LIMITS (exploiting = cheating). Any shared length prior is identical across
  series at fixed t so does not help cross-sectionally anyway.
- 2025 (one-shot) edition threads for feature ideas: "Fast features to achieve over 80%
  AUC" (Oct 2025), "Winning Solutions of Structural Break Challenge" (Nov 2025). pycatch22
  mentioned as a feature lib. NOTE: 2025 was one-shot (break always at boundary) so its
  80%+ numbers do NOT transfer directly to real-time TS-AUC.
- Gap: our 56.84% (rank ~229) vs top-50 >62.5%, top-10 64-64.95%. ~6-8pt gap; feature
  nibbles give +0.2pt each. Need a qualitatively different lever (cross-series calibration,
  better within-online changepoint localisation via max-split CUSUM/KS scan).
- Submission = `streaming_detector.ipynb` (supervised HGB, 14 two-sample features).
- `_exp_fast.py` = validation harness (TS-AUC + infer speed).

## Perf lessons (E4/E5)
- scipy.stats per-window ~7ms; pure-numpy equivalents ~0.36ms.
- Single-row predict_proba dominates infer cost; fewer trees + larger PREDICT_EVERY help.
- Best TS-AUC so far ~0.57 (supervised) vs 0.5065 baseline LB.
- CRITICAL: os.environ OMP_NUM_THREADS=1 does NOT apply on cloud (crunch imports numpy
  first; infer workers are separate processes). MUST wrap infer loop in
  `with threadpool_limits(limits=1):` (threadpoolctl, ships w/ sklearn) to cap threads at
  call time. Without it: 4 workers x16 OpenMP threads oversubscribe -> 5h+ hang.
- E6 (2026-08-10): PREDICT_EVERY 15->1 (predict every online step, no staleness).
  Sweep (_sweep_pe.py, 400 val): PE15=0.5636, PE5=0.5654, PE1=0.5663; PE1 ~7min/full
  test @12core = within budget. TS-AUC ranks series cross-sectionally per step, so held
  (stale) scores lose vs freshly-updated peers. Only inference budget constrains PE.
  RESULT: LB 55.48% rank 289 (from 296), 31min/16vCPU. Local +0.0027 -> LB +0.18pt.
- Improvement roadmap (agreed): 1)PREDICT_EVERY=1 DONE 2)more/denser training data DONE
  3)multi-scale + recent-vs-older window features 4)robust/rank-based stats 5)model tune.
  Prioritize free wins; only hard limit is inference budget. Validate on >=800 series.
- E7 (2026-08-10): MAX_TRAIN_SERIES 6000->10000, keep STEPS_PER_SERIES=10.
  Sweep _sweep_data.py (fixed 400 val, PE=1): 5000/10=0.5457, 9000/10=0.5648(best),
  9000/20=0.5582, 9000/40=0.5585, 9000/20 near-tau=0.5486. LESSON: more SERIES helps big
  (+0.019); more STEPS/series HURTS (redundant correlated rows); near-tau densify HURTS.
  Train-time only, infer unchanged. RESULT: LB 55.94% rank 268 (from 289); cloud trained
  9934 series/98.8k rows, adaptive parallelism=15 on 16vCPU, determinism passed, 42min.
  Progression: E0 rank388 -> E5 296 -> E6 289 -> E7 268.
- E8 (2026-08-10): +3 recent-vs-older split features (window halves: mean_z/log_var_ratio/
  KS between older half w[:n//2] and recent half w[n//2:], guard >=6 pts/half else 0).
  History-independent; fires when window straddles the break. Sweep _sweep_feats.py (PE=5,
  400 val): base=0.5370, +short=0.5410, +split=0.5430(best), +short+split=0.5422 (short
  doesn't stack). Confirm PE=1 (_confirm_split.py): base 0.5380 -> +split 0.5441 (+0.0061),
  infer +22%. Feature count 14->17. RESULT: REGRESSION -> LB 55.70% rank 269 (DOWN from
  E7 55.94%/268). Local +0.006 REVERSED on full test. REVERTED to 14 feats.
  METHODOLOGY LESSON: single 400-series val too noisy to trust ~0.006 deltas; use larger/
  multi-fold val or treat LB as ground truth for small changes. Kept E7 (14 feats) as best.
  Best submission = E7: 14 feats, PE=1, MAX_TRAIN_SERIES=10000, LB 55.94% rank 268.
- VALIDATION HARNESS (_val_harness.py, built after E8): trains 1 model/candidate on a
  fixed train pool (disjoint), evaluates on N_FOLDS=5 x FOLD_SIZE=400 DISJOINT folds.
  Reports per-fold TS-AUC, mean, std (=noise floor), + PAIRED delta vs base with
  t=mean_delta/(std_delta/sqrt(K)). DECISION RULE: only ship a feature change if paired
  gain > fold std and |t|>=2. Candidates = module-level fns (w,ctx)->list in CANDIDATES
  dict (base14, split17). Streaming parallelised via joblib loky N_JOBS=cpu-1, BLAS
  capped via threadpool_limits(1). Reuses helpers imported from _exp_fast. Run:
  .venv/bin/python _val_harness.py
  FIRST RUN (5x400 folds, 6000 pool, 778s): base14 mean 0.5511 std 0.0125 (folds
  0.5493 0.5708 0.5402 0.5409 0.5544 -> NOISE FLOOR ~0.0125). split17 delta mean
  +0.0009 t=0.31 -> NOISE (flips sign per fold). Confirms split was never real; harness
  would have rejected it pre-submission. Infer ~355s/candidate (2000 series, 15 jobs, PE=1).
- E9 (2026-08-10): robust/rank-based stats via _val_harness.py (5x400 folds, 6000 pool).
  Candidates base14 + {median-z & log-MAD (rob_loc); Mann-Whitney rank-loc z (mwu);
  Brown-Forsythe robust scale (bf); all four (rob_all)}. RESULTS (paired vs base, noise
  floor std 0.0125): rob_loc +0.0002 t=0.19; mwu -0.0004 t=-0.47; bf +0.0006 t=0.18;
  rob_all -0.0032 t=-0.88. ALL NOISE/REGRESSION -> NOT SHIPPED. Lesson: robust restatements
  are REDUNDANT with existing KS/mean_z/log_var_ratio (same location/scale/dist shift), not
  additive. Notebook stays E7 14-feat. Roadmap step 4 DONE (negative result, zero LB cost).
- E10 (2026-08-11): HGB hyperparam grid (_tune_hgb.py, 108 configs, GroupKFold row-level
  proxy AUC). Baseline proxy 0.5682 ranks #13/108; best +0.0013 (< proxy noise ~0.006).
  Model NOT the bottleneck -> NOT shipped.
- E11 (2026-08-11): WINDOW geometry 75->130. Proxy sweep rose monotonically 0.544(w40)->
  0.568(w75)->0.594(w130), plateau ~160-250, but proxy OVERSTATES big-window benefit;
  streaming TS-AUC is arbiter. 5-fold: w130 +0.0051 t=1.73 (all folds +). 10-fold
  (overnight): w130 vs 75 +0.0110 t=+3.96. Clears |t|>=2. SHIPPED WINDOW=130 (infer cost
  ~identical, free win). RESULT: LB 56.63% rank 235 (from E7 55.94%/268) = +0.69pt +33 ranks,
  BIGGEST single jump. Validates 10-fold harness (local +0.011 -> LB +0.69pt). Progression:
  E0 388 -> E5 296 -> E6 289 -> E7 268 -> E11 235. GOTCHA: first attempt the notebook
  reverted WINDOW->75 (formatter/auto-revert) and reproduced E7 55.94% EXACTLY; always
  re-verify WINDOW=130 in config cell before submit.
- E12 (2026-08-11): overnight 40-candidate feature battery (_overnight.py + _feat_lib.py,
  ~3.4h, 5x400 folds @ W75, base14 ref=0.5511). Ship-grade (|t|>=2,d>0): runs +0.0028
  t=2.48; combo_volatility(abschange/rv_ratio/curv/down_up_vol) +0.0050 t=2.40; spec_entropy
  +0.0054 t=2.39. Promising(1.5-2): perm_entropy 1.85, median_cross 1.84, longest_run 1.69,
  rv_ratio 1.62. All else noise/neg (distance/robust/cusum/wasserstein reinforce E9). Winners
  CLUSTER by family (run/level-crossing; complexity/spectral) -> real signal vs ~2 expected
  FPs from 40 tests. NOT shipped yet: battery ran @W75, must RE-CONFIRM top feats @W130
  (10 folds) before adding (window interaction). Files: _overnight_results.tsv/_summary.txt.
  Remaining next: W130 feature re-confirm (runs/spec_entropy/combo_volatility + promising,
  10 folds @ W130, individually + stacked) -> E13. E11 LB DONE (56.63%/235).
- E13 (2026-08-11): re-confirm E12 winners @WINDOW=130, 10 folds (_confirm_feats130.py,
  base14@130=0.5670 ref). RESULTS: spec_entropy +0.0050 t=+4.67 SHIP; stack_runs_spec
  (runs+spec_entropy) +0.0071 t=+3.49 SHIP; perm_entropy +0.0022 t=1.93 promising; runs
  ALONE +0.0011 t=0.81 NOISE; combo_volatility +0.0008 t=0.41 NOISE; rv_ratio -0.0001 noise.
  KEY: 2 of 3 W75 winners COLLAPSED at W130 (runs 2.48->0.81, combo_volatility 2.40->0.41) =
  window-specific artifacts -> re-confirm step vindicated. Only spec_entropy held+strengthened
  (2.39->4.67). SHIPPED spec_entropy alone as 15th feature (compute_features 14->15, added to
  FEATURE_NAMES; spectral entropy of window = -sum(p log p)/log(len) on rfft power). NOT
  shipping stack (runs' extra +0.0021 is noise-on-its-own -> E8 reversal risk).
  Follow-up E14 candidate: +runs on top of spec_entropy. (bug: stack_ship keyerror b/c
  combo_volatility is a COMBOS key not EXTRA; harmless, try/except caught it.)
  RESULT: LB 56.84% rank 229 (from E11 56.63%/235) = +0.21pt +6 ranks. Progression:
  E0 388 -> E7 268 -> E11 235 -> E13 229. Best = E13: 15 feats (base14+spec_entropy),
  WINDOW=130, PE=1, 10000 series.
- With per-worker 1 thread, raise INFER_PARALLELISM. Now adaptive:
  `max(1, (os.cpu_count() or 4) - 1)` so it auto-scales to any machine.
- E5 RESULT (2026-08-10): fix worked. Cloud run fast, determinism check passed,
  LB 55.30% -> rank 296/1144 (from E0 rank 388). Hang was thread oversub, not model.
- GPU is useless here: sklearn HGB is CPU-only; infer is single-row streaming predict
  (latency-bound) so GPU boosters (xgb/lgbm/catboost) would be slower, not faster.
  Speed levers are all CPU-side: threadpool_limits(1) + more cores + PREDICT_EVERY.
