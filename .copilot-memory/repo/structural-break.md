# structural-break2026 (CrunchDAO real-time)

## Conventions
- Document EVERY experiment in `README.md` (experiment-log table). Keep concise, no bloat.
- Append new runs to the table; record tests/observations that informed decisions.

## Environment
- Python venv: `.venv/bin/python` (system `python` not on PATH). Use absolute path.
- Available: scipy, sklearn, numpy, pandas, joblib, crunch, nbformat(5.11.0, since 2026-08-12). NOT: lightgbm.
- nbformat NOW installed -> create/edit notebooks via nbformat.read/write (validates on write); no longer
  need raw-JSON generator scripts. Still re-read notebook right before editing (auto-revert). nbformat
  validates STRUCTURE not cell CONTENTS -> won't fix byte-corruption inside a cell (e.g. baseline.ipynb
  `import joblib6ø+` garbage; fix via git checkout / crunch setup-notebook / surgical line replace).
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
- E21 RESULT (2026-08-12, full 10-fold, 167min): BEST NON-CHAMPION SIGNAL SINCE E17 but BELOW ship bar.
  vs champion 0.5795: ts_wass (W1) 0.5821 d=+0.0026 t=+1.83 promising; ts_energy 0.5819 d=+0.0024 t=+1.81
  promising; ts_cvm 0.5814 d=+0.0019 t=+1.44; ts_all(57f) 0.5819 d=+0.0024 t=+1.07; ts_mmd 0.5808 t=+0.87.
  READS: (1) Wasserstein-1 & energy are ~COLLINEAR (both CDF-gap integrals: W1=int|F-G|, energy=2int(F-G)^2)
  -> give the SAME +0.0025 signal; stacking them (ts_all) DILUTES (cvm/mmd noise drags t to 1.07). (2) Real
  distribution-SHAPE signal beyond mean/var exists but sits at t~1.8 (10 folds) = just under user bar
  t>=1.9-to-LB-test and |t|>=2-to-ship. (3) cvm/mmd weaker. NOT shipped yet. TWO-SAMPLE distance IS the
  first live lever since champion -> worth sharpening, NOT abandoning.
- E21 FOLLOW-UP OPTIONS (pick before spending LB slot): (a) confirm ts_wass at 20 folds -> if t firms
  >=1.9 ship to LB; (b) E22 = ISOLATE pure-shape W1: standardize BOTH window & history (subtract mean,
  /sd) before W1 so it's orthogonal to champion's mean/var feats -> may strengthen; (c) ts_wass single
  best scale only (drop redundant scales). Avoid stacking (collinear). Champion still stands.
- DECISION (2026-08-12): user chose to spend 2 LB slots rather than more slow local runs. Test the two
  DISTINCT signals on LB: E22a ts_wass first (strongest), then E22b ts_cvm (rank-based, orthogonal).
  Skip energy (collinear w/ wass). If ts_cvm doesn't help, drop it. NO config switch (hardcode per sub).
- E22a WIRED (2026-08-12): added _w1(a_sorted,b_sorted,sd) to notebook feature cell (ported EXACT from
  _twosample.py) + compute_features_ms now appends `_w1(np.sort(ws), ctx["sample_sorted"], ctx["sd"])`
  per scale -> 48 feats. VERIFIED: notebook compute_features_ms == harness ts_feat(...,dist_keys=("wass",))
  bit-exact (maxdiff 0.00e+00 over 20 trials, all finite, nfeat=48). Config unchanged (WINDOW=250,
  MS_SCALES=(60,130,250)). Shared by train+infer. READY TO SUBMIT. LB champion still 57.50%/214; CV
  expectation +0.0026 (~+0.25pt) IF it transfers (marginal t=1.83, could reverse like E14 perm_entropy).
  AWAITING LB RESULT for E22a.
- E22b RESULT (2026-08-12, sub #16): ts_cvm LB = 57.40% rank 218 = WORSE than champion 57.50%/214
  (-0.10pt). REVERSED as predicted (CV t=1.44 < hardened bar). Now 0-for-3 on marginal features
  (perm_entropy t=1.93, wass t=1.83, cvm t=1.44 all reversed). Two-sample distances FULLY DEAD as ship
  candidates. streaming_detector_cvm.ipynb should be treated as abandoned. Champion 57.50%/214 STANDS.
- E22b PARALLEL (2026-08-12): user may submit 2 notebooks concurrently. DECIDED: keep them INDEPENDENT
  (A=streaming_detector.ipynb champion+wass; B=streaming_detector_cvm.ipynb champion+cvm), NOT a
  wass+cvm STACK. Rationale: E21 ts_all (all 4 distances stacked) DILUTED to t=1.07 < ts_wass 1.83 =>
  GBT does not benefit from correlated distance features; LB can't resolve cvm's ~+0.15pt increment on
  top of wass (< LB noise). Only CV-validate a stack if BOTH beat champion on LB.
  streaming_detector_cvm.ipynb = copy of champion+wass with _w1 swapped for _cvm (rank-based, scale-free,
  no sd; append `_cvm(np.sort(ws), h)`), 48 feats. VERIFIED bit-exact vs harness ts_feat(dist_keys=("cvm",))
  (maxdiff 0.00e+00, all finite). BOTH notebooks ready to submit. AWAITING LB for E22a(wass) + E22b(cvm).
- E22a RESULT (2026-08-12, sub #15): ts_wass LB = 57.37% rank 217 = WORSE than champion 57.50%/214
  (-0.13pt, -3 ranks). REVERSED despite CV t=1.83. Notebook REVERTED to champion (45 feats, verified).
  *** HARDENED SHIP RULE ***: we are now 0-for-2 on marginal features (E14 perm_entropy t=1.93 reversed;
  E22a wass t=1.83 reversed). EVERYTHING that transferred had CV t>=3 (E13 spec_entropy 4.67, E17
  multiscale 3.06); everything t<2 reversed. => CV cannot resolve a +0.0026 edge (inside CV<->LB gap).
  NEW BAR: do NOT spend an LB slot unless CV t>=~2.5 (ideally >=3). The old "t>=1.9 = LB-testable" rule
  is RETIRED. E22b cvm (t=1.44) is BELOW even the old bar -> RECOMMEND HOLD, do NOT submit (will almost
  certainly reverse too); if already submitted in parallel just read it, expect failure. wass/cvm/energy
  two-sample distances = marginal, DEAD as ship candidates. Champion ms_60_130_250 57.50%/214 STANDS.
- IMPLICATION FOR NEXT LEVERS: marginal feature nibbles (t<2.5) are a waste of LB slots. Need levers big
  enough to clear t>=2.5-3 in CV (like WINDOW=130 and multiscale did). E23 transforms must be judged by
  this hardened bar before any submission.
- *** GOTCHA: data/X_train.parquet (218,514,418 bytes) is GITIGNORED and got deleted (disk 98% full,
  ~9.6G free -> likely cleanup). RESTORE with `.venv/bin/crunch download` (uses .crunchdao/token; needs
  network). Direct S3 URL 403s (needs auth). Other parquets (y_train, y_train_index, X_test.reduced,
  y_test*) persisted. If a harness dies with FileNotFoundError on X_train.parquet -> re-run crunch download.
- E23 series-transforms harness `_transforms.py` BUILT + smoke-OK (45->54 feats) + LAUNCHED full 10-fold
  (pid 240701, log _transforms.log, results _transforms_results.tsv). Added ctx["sample_ordered"] to
  _exp_fast.hist_context (additive; notebook has own hist_context so submission unaffected). Candidates:
  tf_cumsum/tf_diff/tf_mstd/tf_absdev/tf_all; each appends 3 discriminators (mean_z, log_var_ratio, KS)
  between transform(window) and transform(history) per scale, on z=(x-mu_h)/sd_h. Verdict thresholds
  hardened: SHIP(hard) only t>=2.5. AWAITING full results.
- E23 RESULT (2026-08-12, 81min, FULL 10-fold, baseline champion 0.5795): ALL NOISE, dead lever.
  tf_diff +0.0018 t=1.40 (best single); tf_cumsum +0.0010 t=0.53; tf_mstd -0.0006 t=-0.21;
  tf_absdev -0.0008 t=-0.32; tf_all(81f) +0.0023 t=0.53 (dilution again). Nothing near t>=2.5.
  CONFIRMS representation saturation (E19): cumsum/diff/mstd/absdev views add nothing the champion's
  base15 x 3 scales doesn't already capture. Series-transforms = DEAD. Do NOT submit. Champion STANDS.
- 2ND-PLACE REVIEW (2026-08-12, read full README): their 6 transforms = z-score/cumsum/dense-rank/abs/
  moving-avg/moving-std = EXACTLY our E23 transforms (which were noise for us). Their tests = KS(+have)
  + F-test/Levene(=our queued E24). Pipeline: ~2408 feats -> SHAP + LightGBM-gain SELECTION -> LightGBM
  (+1 TabPFN feat). CONCLUSION: no hidden feature magic; we converged. Their edge is STRUCTURAL, not a
  secret feature: (1) EASIER problem (2025 offline, KNOWN boundary -> clean before/after split; ours is
  real-time unknown boundary so same transforms wash out), (2) SCALE+SELECTION not hand-picked adds
  (we dilute when adding 3-9; they let SHAP/LGBM pick from 2408), (3) TabPFN meta-feat (heavy, dubious
  in per-step budget). IMPLICATION: chasing last-year pipeline = wrong problem structure. The 7pt gap to
  leaders (57.5->64.9) is a PARADIGM gap, not a feature-nibble gap. lightgbm NOT installed (would need it
  or use HGB importance/permutation for a large-bank+selection experiment).
- LINK REVIEW 2 (2026-08-12): user gave 2 more repos. emoen/...-Best-Solution = FORK of 2nd-place
  (identical, "4th public LB", nothing new). emoen/structural-break-challenge (fork of secabird) = DIFFERENT
  solution with NEW boundary-agnostic feature families we HAVE NOT tried, good for Option A bank:
  * Hurst exponent (long-memory/mean-reversion) - new signal type
  * AR(5) coefficients (richer than our ac1/ac2)
  * Anderson-Darling test (tail-sensitive dist test, > KS in tails)
  * Fligner-Killeen + Mood's median tests (robust variance/location; Option B cousins)
  * Sample entropy (we only have spectral/perm entropy)
  * (also: variance-ratio dynamics, wavelet db4, tsfresh, CatBoost/LGBM/XGB - some boundary-dependent, skip)
  Their boundary-relative feats (local window around known boundary, distance-to-true-boundary) DO NOT port
  (we're real-time unknown boundary). But the dist/complexity feats above port to window-vs-history.
  => Option A (E25) bank = champion base15 + {Hurst, AR(p) coefs, Anderson-Darling, Fligner-Killeen,
  Mood, sample entropy} x scales, window-vs-history, then permutation/importance SELECT top-K, validate
  subset vs champion at hardened t>=2.5. Option B (E24) = F-test + Levene (+Fligner) variance two-sample.
- TASKS IN FLIGHT (2026-08-12 ~17:06): (A) exp_bank_selection.ipynb CREATED (9 cells, valid, code cells
  compile OK, NOT executed per user) = E25 bank+ExtraTrees-importance-selection; keeps base45, selects
  NEW feats (hurst/AR4/anderson-darling/levene/fligner/mood/sampen x3 scales) with importance>=weakest
  base, validates subset vs champion at t>=2.5. (B) _vartests.py (E24) DONE 10-fold (66.8 min):
  ALL NOISE. vt_fligner d=+0.0013 t=0.42, vt_levene d=+0.0007 t=0.32, vt_lev_fli t=0.13, vt_ftest t=0.10
  (baseline ms_60_130_250 mean 0.5795). => NO winner, nothing wired into champion, nothing new submitted.
- E25 (Option A) LAUNCHED full 10-fold 2026-08-12 19:10 (pid 259387, runner _run_exp_a.py execs notebook
  code cells headless, log _exp_a.log, PYTHONWARNINGS=ignore to mute anderson_ksamp flood). SMOKE OK
  (EXIT0, 2.2min): bank=75 (45 base+30 new), ExtraTrees ranks AR coefs high alongside base; 2-fold verdict
  d=+0.0125 t=0.80 (unreliable). Selection threshold (imp>=weakest base) selected ALL 30 new at smoke.
  Expensive feats (anderson_ksamp, sampen O(n^2)) -> expect ~2-3h. AWAIT _exp_a.log final verdict; if
  selected subset clears t>=2.5 wire into champion, else bank champion (paradigm saturated).
- E25 FINAL VERDICT (2026-08-12, 71.7 min, _exp_a.log): NOISE / slightly WORSE. Selection kept ALL 30 new
  feats (weakest base imp=0.0073; ar4/ar3/fligner/levene/sampen_250 all rank in top-20 alongside base).
  75-feat model mean=0.5759 vs champion 45-feat 0.5795, delta=-0.0036, t=-1.04. The winners' structural
  recipe (large bank + importance selection) does NOT beat champion here -> trailing-window paradigm is
  SATURATED for feature-bank approaches. Option A = DEAD lever. Bank the champion on the feature axis.
  This is the 4th independent confirmation (perm_entropy/wass/cvm marginal + reversed, now bank=noise)
  that adding features to the stateless window snapshot is exhausted. Remaining hope = PARADIGM levers
  E26-E29 (target / data-scale / stateful-detector / model), especially E28 (stateful online detector).
- README BACKFILL (2026-08-12): E19-E25 rows added to experiment-log table (were only in memory). Table
  now current through E25.
- SEARCH STEP-UP (2026-08-12): created 4 NEW paradigm-lever notebooks (NOT executed; E25 still running,
  one 15-job harness at a time). All self-contained, reuse _overnight/_exp_fast/_feat_lib, compare vs
  champion on same folds at hardened t>=2.5, have SMOKE toggle, code cells compile. Built via nbformat
  generator _make_search_notebooks.py (run then deleted). To run each detached later: nohup runner or
  jupyter; they're notebooks so use a headless runner like _run_exp_a.py pattern or execute in VS Code.
  * E26 exp_hazard_target.ipynb - change TRAINING TARGET (frozen champion feats): ramp regression
    clip((t-tau)/H,0,1) + deweight-near-tau sample weights. Custom evaluate_tgt (regressor/weighted clf).
  * E27 exp_more_data.ipynb - scale TRAINING DATA breadth vs depth (E7: breadth wins). load_pool_n keeps
    val folds FIXED (reserved from END), varies N_TRAIN {6k,10k,MAX} x10 steps + depth control 6k x20.
    Cheapest lever - if breadth clears t>=2.5 just retrain champion on more series, no code change.
  * E28 exp_sequential_detector.ipynb - stateful UNSUPERVISED online detectors: CUSUM, Page-Hinkley,
    BOCPD (Gaussian const-hazard, score=1-P(r==t)). Real paradigm shift. If beats champion=huge; if
    weaker-but-complementary -> add detector score as a champion FEATURE (follow-up).
  * E29 exp_lightgbm_pipeline.ipynb - winners' toolchain. NEEDS `pip install lightgbm` (guarded HAS_LGBM).
    A) LGBM vs HGB on champion feats (model swap). B) bank(champion+Hurst+AR4) -> LGBM gain>0 select -> LGBM.
    Reuses _overnight build_training/fold_auc (LGBM has predict_proba). If both noise -> gap is problem
    structure not toolchain, bank champion.
- RECOMMENDED ORDER when cores free (after E25): E27 (cheapest, data-only) -> E26 (target) -> E28 (BOCPD)
  -> E29 (needs lightgbm install). Only ONE 15-job harness at a time.
- OVERNIGHT SEARCH LAUNCHED 2026-08-12 22:26 (pid 267600, runner _run_overnight.py, log _overnight_search.log,
  PYTHONWARNINGS=ignore). Runs E27->E26->E28->E29 SEQUENTIALLY in one process (fresh namespace each, try/except
  so a crash doesn't abort the rest, gc between) => only ONE Parallel(n_jobs) harness active at a time, no
  oversubscription. Full 10-fold (SMOKE unset). lightgbm 4.7.0 confirmed in venv.
  SMOKE=1 plumbing pass (2026-08-12 22:08, ALL 4 ran clean end-to-end, 4.5 min total). Smoke hints (UNRELIABLE
  per E24 lesson, confirm at full 10-fold):
    * E27 more_data breadth_MAX_x10 delta=+0.0697 t=+6.21 SHIP -- but smoke champ baseline was only 0.5160
      (300 series) vs breadth 9920 series, so effect is literally more-data not a fold artifact => genuinely
      promising. depth_6k_x20 control = noise (matches E7).
    * E26 hazard_target: all noise/worse (ramp/deweight).
    * E28 detectors standalone all BELOW champion (cusum/page_hinkley/bocpd) -- expected; follow-up = add as feature.
    * E29 lightgbm: A model-swap noise; B bank(60)+gain-select -> LGBM delta=+0.0109 t=+2.68 SHIP at smoke.
  NEXT: read _overnight_search.log final summary block in the morning; for any SHIP(hard) t>=2.5 at FULL folds,
  wire into champion + user local-test + submit; else mark dead. Expect E27 could be the free win (retrain on
  more series, no code change). Verify done: pgrep -af "[_]run_overnight\.py"; grep -A6 "SEARCH DONE" log.
- OVERNIGHT SEARCH FINAL VERDICTS (2026-08-13, done 120.9 min, _overnight_search.log). ALL FOUR = NOISE/WORSE.
  Champion baseline mean=0.5795 on all. NONE cleared bar. The smoke SHIP signals (E27 breadth, E29 bankB) BOTH
  evaporated at full folds -- textbook "never trust smoke" (E24 lesson) reconfirmed.
  * E27 more_data: breadth GENUINELY IMPOSSIBLE, not a bug. Dataset = 10000 total series; 10-fold x 400 val
    reserves 4000 => MAX_TRAIN = 6000 = exactly what champion already uses. breadth_10k/MAX both capped to 6000
    (delta=+0.0000 t=+inf artifact). Smoke's 9920 was artifact of tiny 2-fold(x40) val reservation + unfair
    300-vs-9920 champ compare. depth_6k_x20 (2x steps/series) delta=+0.0026 t=+0.82 NOISE (matches E7). To get
    more "breadth" you'd have to shrink val = trade reliability. DATA-SCALE LEVER = DEAD/exhausted.
  * E26 hazard_target: ramp20 t=+0.68, deweight20 t=+0.59, ramp40 t=-0.50. ALL NOISE. Target-shaping = DEAD.
  * E28 sequential_detector standalone: cusum 0.5227 (t=-11.60), page_hinkley 0.5202 (t=-10.23), bocpd 0.5427
    (t=-6.82) -- all FAR below champion 0.5795. Stateful online detectors as standalone rankers = DEAD. (Could
    add bocpd score as a champion feature, but 0.54-alone is weak + everything else saturated => not worth it.)
  * E29 lightgbm: A model-swap champion-feats nf=45 t=-2.18 (WORSE), B bank(60)+gain-select t=-1.03 NOISE.
    Toolchain swap = DEAD => gap is problem-structure not toolchain (confirms hypothesis).
  CONCLUSION: E26-E29 + E25(bank noise) + 3 reversed marginal feats = OVERWHELMING evidence the trailing-window
  real-time paradigm is at its CEILING as implemented. Champion E17 ms_60_130_250 (LB 57.50%/214) STANDS as the
  final submission on this approach. The ~7pt gap to leaders (~64.9%) is the fundamental problem framing (2025
  known-boundary offline vs 2026 real-time unknown-boundary), NOT any feature/target/data/model tweak. All
  local levers explored & exhausted. Runner _run_overnight.py + _run_exp_a.py can be deleted (kept for now).
- NEW DIRECTION (2026-08-13): ENSEMBLE + TUNING + VALIDATION FIX. User rejects "paradigm saturated"; believes
  a single HGB under-extracts the tested features. Plan: add a DECORRELATED 2nd model (logistic regression;
  RandomForest later), tune both, blend by weighted proba-average (rank-lift under TS-AUC). Also fix validation.
- VALIDATION DESIGN (confirmed from _overnight.load_pool): NOT k-fold CV. ONE fixed train set (6000), the
  held-out 4000 cut into 10 DISJOINT val chunks of 400 => 40% held out (NOT 10%; the 10 is #folds). Train does
  NOT rotate. Fold assignment = seeded random shuffle, NOT stratified => folds differ in difficulty; the PAIRED
  t-test (champion+cand on SAME folds) cancels that for comparisons, but absolute AUC is noisy fold-to-fold.
  SERIES LABEL = (yi.tau_index >= 0) i.e. has a structural break (tau_index=-1 => no break). Enables stratified folds.
- _ensemble_tune.py (built 2026-08-13): the new harness. Config: train 8000 / val 2000 (20%, down from 40%),
  10 STRATIFIED folds by break-rate (round-robin pos/neg => each fold break-rate identical, verified 0.500/0.500
  in smoke). champion_feat = base15 x (60,130,250) = 45 feats, WINDOW=250 (parity with E17 champion). Members:
  make_hgb(**over) + make_lr(C,class_weight)=Pipeline(StandardScaler,LogisticRegression). Sections: baselines
  (HGB, LR) -> blend weight-sweep w_lr in {.2,.3,.4,.5} (proba-avg, cached) -> HGB tuning grid (7) -> LR grid (5)
  -> best-HGB (x) best-LR blend sweep. Reports paired-t vs champion, hardened SHIP bar t>=2.5.
  KEY EFFICIENCY: precompute_fold computes the 45 val features ONCE per fold (cached); every model only does a
  cheap predict_proba on the cache => whole run (baselines+grids+blends) is short (feature precompute dominates,
  ~half an E-series streaming since val=2000). NOT an overnighter.
  CORRECTNESS PROVEN: fixed an off-by-one (warm-up scores from idx==MIN_N-1, matching _stream_one, not MIN_N).
  Parity check vs canonical _overnight.fold_auc => max abs diff = 0.0 (EXACT). Smoke ran all sections OK.
  DO NOT TRUST SMOKE NUMBERS: 300-series smoke gave champion HGB 0.4253 (below random) and LR "beating" it
  t=+17.59 -- pure undertraining artifact (canonical streaming reproduced same 0.44/0.41). LR is linear on
  engineered contrast feats; expect it MUCH weaker than HGB at full scale. Only full 10-fold t counts.
  NEXT: launch full `env PYTHONWARNINGS=ignore .venv/bin/python -u _ensemble_tune.py > _ensemble_tune.log 2>&1`.
  User constraint: daytime = no big runs; run is short but saturates 15 jobs briefly -> queue for evening or on ok.
- E30 ENSEMBLE+TUNE FINAL VERDICT (2026-08-13, 5.7 min, _ensemble_tune.log): ALL NOISE/WORSE. Champion HGB
  mean=0.5674 on the new 80/20 stratified val (2000 val; different design from 0.5795 40%-val, only within-run
  paired-t comparable). Results vs champion: LR alone 0.5322 t=-4.91 (much weaker); LR grid C{0.1,0.3,1,3}+bal
  all ~0.531 t~-4.9; HGB grid 7 configs best hgb_msl40 +0.0004 t=+0.33 (DEAD-FLAT, all within +/-0.0004);
  HGB+LR blend weight-sweep w_lr{.2,.3,.4,.5} ALL <=0 and MONOTONICALLY worse (best w=.2 -0.0006 t=-0.68);
  best-HGB(msl40) x best-LR(C3.0) blend also <=0 (w=.2 t=-0.01). NOTHING reached even t>=1.5.
  READ: (1) LR adds NO orthogonal signal -- any blend weight only degrades, so HGB already captures the
  linearly-separable structure. (2) HGB is at its tuning plateau (flat hill, not under/over-fit). MODEL +
  ENSEMBLE axis EXHAUSTED. The ceiling is the FEATURE SIGNAL, not the model. RandomForest deferred but
  expected same (tree ensemble, correlated with HGB). Champion E17 ms_60_130_250 (LB 57.50%/214) STANDS.
  Logged to README E30 row. This is the LAST untried classical lever on this paradigm.
- E31 (2026-08-13): SEQUENCE MODEL + PROPER JOINT TUNING. Built _seqmodel.py (torch 2.13.0+cpu installed,
  CPU). (A) 1D-CNN over raw history-standardized trailing window: window_vec(w,ctx)=clip((w-mu_h)/sd_h,+-8)
  left-padded to 250 + validity-mask channel => (2,250); trained BCE 14 epochs (loss 0.568->0.418), scored
  via SAME precompute_fold/score_fold cache path (CNNProba.predict_proba reshapes (N,500)->(N,2,250)) =>
  apples-to-apples paired-t vs champion HGB on identical folds. RESULT: CNN alone 0.5048 t=-6.62 (WORSE than
  champion 0.5674); every HGB+CNN blend NEGATIVE (w=.2 t=-2.09 ... w=.6 t=-5.29). Learned-representation lever
  DEAD: raw short online window is noisier than the 45 engineered window-vs-history contrasts; CNN can't
  out-extract them. (B) randomized JOINT HGB search (24 configs sampling lr/max_iter/leaves/msl/l2/max_features
  jointly, reusing champion cache). Best hgb#11 (lr0.047,it300,leaves31,msl40,l2=5.0,mf0.5) hit +0.0035 t=+3.51
  SHIP(hard) on SEED=42 -- but that was BEST-OF-24 selection bias. _confirm_tune.py re-validated hgb#11 vs
  champion on 5 INDEPENDENT pool seeds (42/7/123/2024/31337, model random_state fixed): seed42 +0.0035 t=3.51
  (reproduces), seed7 -0.0006 t=-0.41, seed123 +0.0014 t=0.96, seed2024 +0.0031 t=1.66, seed31337 +0.0004 t=0.26.
  SUMMARY mean delta +0.0016, positive 4/5, t>=2.5 on 1/5 => VERDICT NOT ROBUST (fold/search luck). Confirms the
  E30 flat-tuning verdict: no HGB config robustly beats champion. *** LESSON: a best-of-K random search t is
  SELECTION-BIASED; a single re-run at the SAME seed just reproduces it (deterministic) -- must re-validate the
  chosen config on INDEPENDENT fold seeds. Proper way to tune = select on MULTI-SEED MEAN delta, not single-seed
  max. *** Champion E17 STANDS. torch now in venv. _seqmodel.py has a latent bug: precompute_fold empty-fold
  fallback uses FEAT_DIM=45 so a fold with a <=14-len online series breaks vstack of 500-wide window rows (hit
  on first launch; avoided when no ultra-short series in folds). Files: _seqmodel.py, _confirm_tune.py.
  Also SEED-ENSEMBLE test (_seed_ens.py, K=5 champions rs={42,7,123,2024,31337}, avg proba) on 3 pool
  seeds: seed42 +0.0039 t=2.52, seed7 -0.0011 t=-1.11, seed123 +0.0049 t=4.10. mean +0.0026, positive 2/3
  => HOLD (not clean across seeds; dips negative on seed7). Principled variance-reduction, best mean of the
  ship options but still not robust. NOTHING shipped -- champion E17 stands.
- GPU-VALIDATION SETUP (2026-08-13): user moving validation to a separate GPU box for speed (only torch/CNN
  benefits; HGB+LGBM are CPU). Added requirements.txt (pinned: numpy2.5.1 pandas3.0.5 scikit-learn1.9.0
  scipy1.18.0 joblib1.5.3 threadpoolctl3.6.0 lightgbm4.7.0 matplotlib3.11.1 crunch-cli11.10.0; torch NOT
  pinned there -- install torch==2.13.0 from default PyPI = CUDA build on Linux, or --index-url .../cu124).
  Added check_env.py (PASS/WARN/FAIL checker: python ver, pkg versions vs baseline, torch build+CUDA+GPU
  tensor op, data/ files X_train/y_train/y_train_index, harness imports _exp_fast/_overnight/_ensemble_tune;
  exit0 unless FAIL). Data lives in ./data (DATA=os.path.join(HERE,"data") in _overnight & _exp_fast).
  Current laptop torch is 2.13.0+CPU so checker WARNs on GPU there -- flips to PASS on the CUDA box.
- E31 (2026-08-13) SEQUENCE MODEL + PROPER JOINT HGB TUNING. User pushed back on E30's shallow one-axis grid
  ("tuning should help") + wanted a sequence model. Built _seqmodel.py (1D-CNN over raw std window) + a
  RANDOMISED JOINT HGB search (24 configs sampling lr x max_iter x max_leaf_nodes x min_samples_leaf x l2 x
  max_features jointly). Installed torch 2.13.0+cpu (venv). Reuses _ensemble_tune load_pool_strat/folds/
  precompute_fold/score_fold (same 8000/2000 stratified 10x200 folds, champion baseline 0.5674) for apples-
  to-apples paired-t. CNN input = std trailing window z=(x-mu_h)/sd_h clipped +/-8, left-padded to 250 +
  mask channel -> (2,250); reused precompute_fold by passing window_vec as the "feat_fn" (returns flat 500 =
  2x250), CNNProba wrapper exposes predict_proba so score_fold works unchanged. Labels/warm-up identical.
  *** KEY RESULT (joint HGB search): hgb#11 mean=0.5709 delta=+0.0035 t=+3.51 SHIP(hard) vs champion 0.5674!
  Config: learning_rate=0.047, max_iter=300, max_leaf_nodes=31, min_samples_leaf=40, l2=5.0, max_features=0.5.
  The STANDOUT is max_features=0.5 (per-split COLUMN SUBSAMPLING, champion never used it -> decorrelates trees
  RF-style) + stronger l2=5.0 + max_iter=300. E30's one-axis grid NEVER tested max_features. Coherent pattern
  across 24: every high-lr/aggressive config LOST, every low-lr/high-reg config gained -> not obviously a
  multiple-comparison fluke, but MUST confirm on independent fold seed before spending LB slot (24-way search
  = multiple testing; hardened rule requires t>=2.5 AND we've been burned by CV->LB before). CAUTION: this t
  is on the 20%-val stratified design at 8000-train; deployed trains on ALL ~10000 so should transfer >= this.
  *** BUG FOUND+FIXED: precompute_fold empty-fold fallback hard-coded np.zeros((0,FEAT_DIM=45)); a val series
  with < MIN_N online points emits 0 rows -> (0,45) block can't vstack with 500-wide window rows -> CNN arm
  crashed (champion arm never tripped it, rows are 45-wide). FIX (in _ensemble_tune.precompute_fold): infer
  width from first non-empty part, coerce empty parts to (0,width). Benign for champion (width=45). Re-launched
  _seqmodel with SKIP_HGB=1 env (skip already-captured 24-config search) to get CNN + HGB-CNN blend. AWAITING.
  NEXT: (1) read CNN + blend verdict; (2) CONFIRM hgb#11 on independent fold seed(s) -> if t holds >=2.5, wire
  params into champion streaming_detector.ipynb (trivial: add max_features=0.5,l2=5.0,max_iter=300,msl40 to the
  HGB config) + user local-test + submit. This is the FIRST clear SHIP-bar signal since E17 multiscale.
- CAVEAT DISCOVERED (2026-08-13, user pushback): the "saturated" verdict rests on a LOCAL harness that
  FIXES 6000 train / 4000 val (40% holdout). But deployed train() uses ALL ~10000 series (no val split).
  => every candidate was validated at 6000-train while production runs at ~10000-train. Features that only
  earn signal near full data (AR/Hurst/sampen/variance families - high E25 importance but scored noise) get
  a SYSTEMATIC FALSE-NEGATIVE. So feature levers may NOT actually be dead; the harness may have masked them.
- FIX 1 (done): streaming_detector.ipynb MAX_TRAIN_SERIES 10000 -> 10**9 (consume ALL platform series, no cap).
- E30 = true k-fold CV re-validation (_cv_search.py + _bank_feats.py). Real 10-fold CV over ALL ~10000 series:
  each fold trains on ~9000 (approx deployed) + validates ~1000, rotating (every series validated once).
  Efficient: per-step feats identical on train/val paths & each series is val in exactly ONE fold -> build
  cheap sampled training rows once per candidate, refit HGB 10x on row subsets (secs), stream each series once.
  _bank_feats.py holds IMPORTABLE named builders (champion_feat, feat_ar/hurst/ar_hurst/levene/fligner/mood/
  sampen/bank_all) - MUST be importable, NOT partials in __main__ (loky workers can't un-pickle __main__ funcs
  or partials-of-__main__ -> BrokenProcessPool; first attempt failed exactly this way, fixed by moving to module).
  Candidates: champion(45) vs +ar4/+hurst/+levene/+fligner/+mood/+ar4_hurst/+bank_all/+sampen, paired-t over
  10 fold TS-AUCs, hardened bar t>=2.5. Incremental _cv_results.tsv (crash-safe), log _cv_search.log.
  SMOKE (800 series/2 folds) plumbing PASSED for 7/9 (t-values meaningless on 2 folds as always). Champion
  CV mean lower (~0.52) simply because SMOKE uses tiny data. If any family clears t>=2.5 at FULL 10-fold ->
  it was masked by the 6000-train harness => wire into champion + user local-test + submit. If all still noise
  at ~9000-train => feature axis genuinely saturated (verdict then TRUSTWORTHY). NEXT after E30: model tuning
  (never done - fixed HGB config) then ensembling (seed-average safest, then feature-subset/model diversity).
  Champion E17 (57.50%/214) stands. LESSON REINFORCED: 2-fold smoke vt_fligner t=15.33 -> 10-fold t=0.42
  (smoke deltas WORTHLESS; only large-fold t counts). Variance two-sample = DEAD lever (add to list).
- LOCAL HW: i7-1360P, nproc=16 LOGICAL but only 12 physical cores (4 P hyperthreaded=8thr + 8 E-cores).
  N_JOBS=15 saturates; NEVER run two 15-job harnesses (30 procs/16thr oversubscribes, skews results).
- exp_bank_selection.ipynb (Option A) is a RESEARCH/CV notebook (uses _overnight/load_pool on local
  parquet, NO train()/infer()) -> NOT cloud-submittable. Only streaming_detector.ipynb (champion) is.
- 2ND-PLACE SOLN ANALYSIS (2026-08-12, Farukcan Saglam aParsecFromFuture, LAST YEAR OFFLINE challenge):
  pipeline = 6 series TRANSFORMS (z-score, CUMSUM, DENSE RANK, abs, moving avg, moving std) -> stat
  features (mean/median/std/min/max/skew/autocorr/coeffvar/Q25/Q75) global+periodic + HYPOTHESIS TESTS
  comparing segments (F-test, LEVENE, KS) + 1 TabPFN feat = 2408 feats -> SHAP top200/500 + LGBM-gain
  top500 select -> LightGBM. CONVERGENCE: their "most critical part" = TWO-SAMPLE tests (KS/F/Levene)
  comparing segments = EXACTLY our E21/E22 lever (KS shipped; wass/energy/cvm now). Validates direction.
  KEY DIFFERENCE: they are OFFLINE with KNOWN boundary -> before-seg vs after-seg tests + ONE pred/series;
  we are REAL-TIME per-step AUC, unknown boundary -> trailing-window vs history (online analog). Cannot
  copy pipeline verbatim. UNTAPPED LEVERS THAT FIT US (feature-based, do before models, from their playbook):
  (E23) SERIES TRANSFORMS before feature extraction = apply our window-vs-history tests on transformed
  views (CUMSUM amplifies mean shift; DENSE RANK = robust/scale-free; z-score; abs; moving std) -- biggest
  new representational lever, cheap, straight from winning soln. (E24) proper F-test / LEVENE variance
  two-sample tests (we only have crude log_var_ratio). (minor) median/coeffvar/min/max descriptors.
  QUEUE these after E22 LB results. NOTE: TabPFN + SHAP-select + 2408-feat scale is their offline luxury;
  our per-step streaming budget is tighter but transforms+tests are affordable.
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
