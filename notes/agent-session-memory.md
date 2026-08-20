# Agent session memory — backup

Backup of the Copilot agent's session-scoped working notes for this project
(`/memories/session/night-bigrun-plan.md`). Session memory is ephemeral and lives outside git,
so this copy exists purely as a durable, restorable record.

**To restore:** copy the fenced content below back into `/memories/session/night-bigrun-plan.md`
(strip this markdown wrapper). Keep this file updated when the session notes change materially.

Last synced: 2026-08-20.

```text
# APPEND 2026-08-20b — per-offset headroom diagnostic + cumulative(CUSUM) lever = CLOSED
# - FIXED-WINDOW per-offset AUC on REAL (250-len windows both classes, tau>=250, pre-break negs from
#   SAME series => confound-free): champion is a TRANSITION detector. AUC PEAKS at offset 5-10 (~0.755)
#   then DECAYS to ~0.64 by offset 200. Opposite of synth §6c (which rose = generator artifact). Cause:
#   trailing 250 online window FORGETS pre-break baseline as it fills; real breaks subtle so persistent
#   level contrast weak. => headroom is the LATE-offset decay, NOT early detection. Do NOT chase a purer
#   level detector (persistent contrast is near floor).
# - "14 stats about history" = MISREAD (old _exp_fast prototype). Champion describes history richly:
#   hist_context ~10 moments + 200-pt empirical CDF (KS) + AR(8) model + per-series null bands (dep 24
#   wins, calib 20-32 wins) + hist ACF{1,2,5,10} + ARCH proxy. Richer-history axis = calib banks E68-71
#   = already CLOSED. Bottleneck is STATELESSNESS (cross-step memory), orthogonal to history description.
# - SCREEN (per-offset AUC, pooled OOF HGB, champ61 base):
#   * SHRINKAGE (n-adaptive shrunk mean_z/lvr, early): DEAD (+0.0002 overall, hurts [0,2]). mean_z already
#     studentizes by se=sd/sqrt(n); shrinking further biases to null. DROP.
#   * CUM naive (raw CUSUM): vs PRE-break negs looked great (+0.008 overall) but vs NO-BREAK negs (real
#     metric negs) HURTS early (-0.009 @[0,2]) bc K=1.0<null mean => "time-online" counter, cum grows
#     unbounded 11->125 on no-break => false-alarm accumulation.
#   * CUM null-calibrated leaky-CUSUM (_cum_feats.py: SW=40,CAP=6,K0=0.75,LEAK=0.99,GAMMA=0.97; med/MAD
#     null from history): STRONG 8-seed gate (N=1000+1000, metric dAUC vs NO-BREAK negs, all-pos &
#     mean>=0.0015 & t>=2.0) = FAIL. mean=+0.0016 t=2.91 BUT 7/8 pos (seed7 -0.0010) & LATE gain EVAPORATED
#     (+0.0003, 5/8). Bounding to be no-break-safe throttles the same persistence giving the late lift.
#   * CUM runmax-only variant (drop CUSUM, 2 decayed-running-max scales GAMMAS=(0.95,0.99)): gate = FAIL,
#     WORSE. mean=-0.0003 t=-0.33 5/8; late +0.0018 5/8. No robust signal.
#   => CROSS-STEP MEMORY LEVER CONCLUSIVELY CLOSED. Consistent with info-floor thesis. Nothing shipped.
# - Champion notebook UNCHANGED (still diverges: has never-scored haz_s20 E72 probe from failed submit
#   #106345 timeout; LB champion = b_no_accf 62.48% / rank ~67). Did NOT touch notebook, did NOT submit.
# - README: added subsection "Per-offset head-room on real data — the champion is a transition detector"
#   (before "## Experiment details"): transition-detector curve, shrinkage=dead, CUM both variants gate-fail,
#   "describe history more = calib banks E68-71 = closed; residual axis = cross-step memory = also marginal".
# - _cum_feats.py + _gate_cum.py were scratch; both REMOVED (lever closed).

# APPEND 2026-08-20 — real-data calibration + generator fix + failed submit
# - Measured REAL: H p50~2970 (p95 4808); O p50~506 (p5 59 short tail); tau_abs p5~11 (early breaks),
#   tau_rel ~uniform. Break effect sizes SUBTLE: |mean_shift| p50 0.05/p95 0.28 (90% <0.2 std);
#   var_ratio p50 0.99/p95 2.37 (57% within 0.8-1.25); kurt_online p50 0.08/p95 17 (rare heavy tails).
# - Step0 real probe: no recoverable mean/var/drift at low offset (AUC ~0.51). trend@0.43 was
#   SYNTH-ONLY artifact (old slope 10x too big). Low-offset gap = genuine info floor.
# - FIXED _synth_gen.py to match real: H integers(1100,4900); O integers(50,1000); tau floor 5;
#   gauss_mean shift lognormal(-3,1); gauss_var/ar_var sd=exp(N(0,0.2)); ar_coef/ma/arma = SMALL
#   perturbation of pre coeffs (_perturb_ar2); trend slope 0.0002-0.0015. Added _heavy_tail_overlay
#   (clustered stoch-vol, 20% series + 4% extreme tier s up to 1.3) -> kurt p50 0.09/p75 0.92/p95 10.8.
#   FIXED latent bugs: gauss_var used out-of-scope sd; _garch allowed alpha+beta>=1 -> inf (now clamp 0.98).
# - REDO per-family detectability on NEW synth (K_PER=80, offsets 5..80): gauss_mean/gauss_var/trend
#   COLLAPSE to ~0.5 (matches real!); only shape/dep families separable (spectral/tdist/garch/skew/arma);
#   pooled climbs slow 0.57->0.62->0.75, NO sharp elbow (old elbow~21 was artifact of too-easy synth).
# - NOTE: detectability harness (_series_with_post) calls gen_break/gen_nobreak DIRECTLY, NOT
#   generate_one -> heavy_tail_overlay NOT in harness. So kurtosis change does NOT require rerun.
# - FAILED SUBMIT: run #106345 = streaming_detector_depearly.ipynb (haz_s20 E72 probe) TIMED OUT
#   exit 124 at 1h29min but Quota Duration only 4.5min => likely HANG not compute-bound (daily budget 11h,
#   so NOT time-budget edge — earlier "edge" claim RETRACTED). haz_s20 got NO LB reading (inconclusive).
# - Multi-model-by-online-length idea: does NOT help. TS-AUC stratified by online step t; at each t ALL
#   series have same trailing window min(t+1,250) regardless of total length. No length flag fixes info floor.
# - README updated with real-data findings + synth-vs-real calibration tables (2 new subsections before
#   "## Experiment details"). Did NOT document code changes, only findings (per user).

# METRIC TS-AUC: computed PER absolute online step t (vertical slice), NOT one global AUC. At each t:
#   scores of all series alive at t, label 1 if t>=tau, cross-series rank-AUC, weight w(t)=n_pos*n_neg,
#   TS-AUC = sum w(t)*AUC(t)/sum w(t). Random=0.5. Champion window = x_on[max(0,t-249):t+1] ONLINE-ONLY.
# ENV: .venv/bin/python (3.13.7) absolute path; N_JOBS=15; notebook kernel WON'T attach -> validate headless
#   via scratch scripts (create/run/rm). Terminal needs requestUnsandboxedExecution=true. tmpdir
#   /home/endrem/.vscode/tmp/tmp_vscode_2. Ship gate: npos==n & mean_d>=0.0015; STRONG min_t>=2.0.


# NIGHT BIG-RUN (2026-08-16) — build a+b+c, launch overnight (8h budget)  [RECOVERED]

GOAL (user): build (a) stacking/model-diversity upgrade, (b) dependence/early-onset CUSUM lens,
(c) NEW Alphabot-style approach = 1000s of features across independent banks + 8 first-level
models + meta-learner. Kick off as ONE detached overnight run.

CHAMPION BASE = trail120 (E56): champion_feat(45) + _residual_reffit extras spec
refs=[{p:8,trail:None},{p:8,trail:120}] scales(60,130,250) => +6 => 51 cols. This is the LB
champion (61.58/94). All 3 deliverables baseline vs this 51-col champion.

ENGINE REUSE (all importable):
- _ensemble_tune: champion_feat(w,ctx), load_pool_strat() [uses ET.SEED], select_picks(X,ids),
  make_hgb(**over), WINDOW=250, SCALES=(60,130,250).
- _residual_reffit: _fit_refs(x_h,spec), _extra_feats(w,refs,spec), _nextra, _fit_ar, _ar_resid,
  SCALES. spec=dict(refs=[{p:8,trail:None},{p:8,trail:120}]).
- _residual_overnight: _eval(cols,Xtr,ytr,Rr,Rl,Rt,caches,y_val,seed) = fixed-CDF blend
  0.6*cdf(hgb TUNED)+0.4*cdf(ranker rank_xendcg). TUNED dict, NCHAMP=45. Returns per-fold AUC arr.
- _lambdarank: fit_ranker(objective,rows,labels,groups), sort_into_groups(rows,labels,tvals),
  RankerProba.
- _overnight: paired(cand,base)->(md,sd,t,d), N_JOBS, MIN_N.
- _exp_fast: hist_context(x_h,HIST_CAP), HIST_CAP, split_series, ts_auc_from, compute_features,
  _autocorr, _spectral_centroid, _fast_ks, WINDOW, MIN_N.
- ctx keys: mu,sd,var,skew,kurt,ac1,ac2,iqr,rng,sc,sample_sorted,sample_ordered.

SEED GATE: SEARCH {42,7,123} -> FRESH {31337,2025,555,99}. STRONG = pos on ALL seeds AND
mean_d>=0.0015 AND min_t>=2.0. Directional: mean_d ~+0.008 -> ~+1pt LB.
CAUSALITY: window w = x_on[max(0,t-250+1):t+1] (online trailing); history = x_h (pre-break).
Break at online index tau; label=1 if t>=tau. So window-vs-history = post-vs-pre but causal/growing.

FILES TO BUILD:
- _night_banks.py: trail120 base + dependence bank (studentized resid EDA-lever2, early-onset
  CUSUM EDA-lever1, ACF-lag contrasts, sup-F scan, arch-lite BDS proxy) + Alphabot banks
  (H stat-tests, M descriptors+fusions, R multiview+distances) + deterministic composite generator.
- _night_stack.py: eval_stack() = series-level KFold OOF meta-learner over diverse first-level
  models (leak-free), reuse fixed-CDF pct transform. build_rows returning sids for series-CV.
- _night_b.py (dep lens, uses _eval), _night_a.py (stack over champion51), _night_c.py (big bank
  + 8 models + meta). _night_run.sh sequential driver b->a->c, crash-guarded, budget guard.

LAUNCH: nohup env PYTHONWARNINGS=ignore .venv/bin/python -u ... (requestUnsandboxedExecution).
Smoke each with SMOKE=1 first. Reduce (c) to 2 search+2 fresh seeds, fewer/smaller folds; add
wall-clock budget guard so c stops gracefully near 8h.
STATUS: writing code.
```
