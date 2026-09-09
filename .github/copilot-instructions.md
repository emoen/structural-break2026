# structural-break2026 — AI project instructions

CrunchDAO ADIA Lab **real-time structural-break detection**. Causal streaming: score each
online point using only `x[:t+1]`. Metric = **TS-AUC** (cross-sectional AUC per online step,
weighted by `n_pos*n_neg`). Accumulating scores across steps gains nothing; only the
within-step ordering matters.

This file is the durable contract for how to work in this repo. Read it before acting.
Deep experiment history lives in memory (`/memories/repo/structural-break.md`,
`/memories/session/current-state.md`); consult it before proposing anything "new".

---

## Champion (current best)

- **Notebook:** `streaming_detector_stack4.ipynb` — **LB 62.97%**.
- **Ensemble:** 4 ACTIVE members, fixed-CDF **percentile blend**, weights sum to 1:
  `ranker 0.45, rf 0.40, xgb 0.10, catboost 0.05`. A 5th model (`hgb`) had weight `0.00`.
  As of 2026-09-09 `hgb` is **no longer trained** (quota optimization; it was fit-but-unused).
- **Features:** `base79` (79 cols) + **calibration block** = 158 cols. Calib = robust
  studentization (median / 1.4826·MAD) of each `base79` feature vs that series' own pre-break
  null (32 sliding history windows). `czrb_all79` is the shipped calib arm. Base79 first,
  byte-unchanged, then the 79 studentized cols.
- **Key config:** `WINDOW=250`, `MS_SCALES=(40,100,250)`, `HIST_CAP=150`, `STEPS_PER_SERIES=30`,
  `DENSE_K=60`, `PREDICT_EVERY=1`, `MAX_TRAIN_SERIES=10**9` (all platform series).
- **RF served as ONNX** (`skl2onnx`) in `infer()` — sklearn per-row `predict_proba` is ~300x
  slower and caused cloud timeouts. Keep RF as ONNX.

## Champion lineage (LB): 57.50 (multiscale) → 57.72 (tuned) → 58.78 (ensemble) →
60.56 → 61.32 → 61.58 → 62.50 (ranker+rf) → 62.70 → 62.76 (base79) → 62.93 (calib) → **62.97 (HIST_CAP)**.

---

## Non-negotiable workflow rules

1. **NEVER auto-submit to the leaderboard.** Prepare, verify offline, then the USER submits.
2. **BACKUP-FIRST on the champion.** Before ANY edit to `streaming_detector_stack4.ipynb`:
   `cp streaming_detector_stack4.ipynb backups/streaming_detector_stack4.bak_$(date +%Y%m%d_%H%M%S).ipynb`.
   Snapshots + the frozen `CHAMP_LB6297` copy live in `backups/`.
3. **FREEZE the champ before changing.** Keep a byte-exact frozen copy, edit the working
   notebook, then diff working-vs-frozen to isolate exactly what changed. (git HEAD is stale —
   predates the calibration work — so it is NOT a usable reference.)
4. **Any `train()` change must preserve the exact sampled rows** (RNG-preserving) or it moves the
   LB. Prove score-neutrality (e.g. byte-identical feature matrices) before trusting a refactor.
5. **One 15-job harness at a time.** Local HW = i7-1360P, 16 logical / 12 physical cores.
   `N_JOBS=15` saturates; two concurrent 15-job runs oversubscribe and skew results.
6. **Never tee crunch / sklearn runs through the VS Code terminal** — per-iteration warnings
   block joblib workers. Redirect to a file (`nohup ... > run.log 2>&1 &`) and `grep` it.
7. **Check `df -h .` before cache-heavy sweeps.** Probe caches can be 30GB+. Delete each
   `_*_cache` after the run.

## Environment

- Python venv: **`.venv/bin/python`** (system `python` is not on PATH; use the absolute path).
- Installed: numpy, pandas, scipy, scikit-learn, joblib, lightgbm, xgboost, catboost,
  onnxruntime, skl2onnx, torch (cpu), crunch-cli, nbformat. GPU is useless here (single-row
  streaming infer is latency-bound; sklearn HGB is CPU-only).
- Competition data in `data/` (`X_train.parquet` is gitignored — restore via
  `.venv/bin/crunch download`). `data/y_test.reduced.parquet` is a **code-check split only**
  (staff-confirmed) — never tune to it; it is NOT a validation signal.

---

## Validation methodology (how we decide to ship)

- **Local harness is directionally predictive** but LB-shrunk (`mean_d ~+0.008` local →
  ~+1pt LB). LB resolution ≈ 0.04pt.
- **Believe-rule / noise gate (current):** an ensemble-delta arm must have
  `mean_d >= +0.0015` AND be **positive on all screen seeds** `{42,7,123}`, THEN confirm on
  **fresh disjoint seeds** `{31337,2025,555,99}`. The believe-rule is a submission-scarcity
  noise gate on the ENSEMBLE delta — a member being weaker than the ensemble is expected;
  decorrelation is what matters.
- **Always re-baseline vs the CURRENT champion**, not an old base. A member confirmed vs a weak
  base can be redundant vs the stronger ensemble.
- **NEVER trust smoke numbers.** Smoke (2-fold / tiny data / reduced recipe) is a
  correctness/plumbing check ONLY. Deltas at smoke scale are worthless (repeatedly: smoke
  `t=15` → full `t=0.4`). Confirm at full folds / full `dense_k` before believing anything.
- **4/4 pool-positive means nothing without fresh-seed confirm** (seeds 2025/555/99 are
  historically adversarial and have correctly killed pool-luck winners).
- Best-of-K search deltas are **selection-biased**; re-validate the chosen config on independent
  seeds (select on multi-seed MEAN, not single-seed max).
- All new features must be **affine-invariant** (scale + location) and **causal** (history-only).
  `base79` already is. Include an affine self-check in every probe.

## Presenting findings (user preference)

- Report results as **clean markdown tables**, not prose.
- Show `config/block, mean_d, min_d, pos (n/N), t_agg, per-seed t (one column per seed), verdict`.
- Include the t score BOTH aggregated AND per-seed; label seed columns with the real seed numbers.
- Separate **SCREEN** vs **FRESH** tables.
- **Lead with absolute TS-AUC**, not just deltas.
- Spell out acronyms in user-facing text (say "random forest" / "leaderboard" / "gradient
  boosting member", not RF/LB/HGB).
- Do NOT verbally declare an axis "closed" to the user; say "what we tried didn't work."

---

## Notebook editing (`.ipynb`) — gotchas

- Prefer the notebook edit tool (edit by cell id). `edit_notebook_file` and
  `replace_string_in_file` operate on the **rendered** notebook source (Python), NOT the raw
  JSON — match the rendered text, not the `"\n"`-escaped JSON. Raw JSON line numbers differ from
  rendered line numbers.
- The `#VSC-...` cell ids (from `read_file` / editor context) are what the edit tool uses; they
  differ from the short `nbformat` ids stored in the file (`"id": "d192dea6"`). Known cell:
  train cell = `#VSC-eef603da`.
- After editing, validate: `.venv/bin/python -c "import json; json.load(open('nb.ipynb'))"`.
- Editing raw JSON strings with string-replace has historically corrupted notebooks — avoid.

## Crunch cloud constraints

- Cloud runs `train()` and `infer()` in **separate fresh processes**. Local `crunch test`
  forks infer workers from the live train proc → **libgomp/loky fork deadlocks** that CANNOT
  happen on cloud. For local end-to-end runs, shut down the loky reusable executor after the
  feature `Parallel(...)` and/or set `INFER_PARALLELISM=1`.
- Generic **"Something went wrong. Contact support"** with no traceback ≈ **out of quota /
  timeout** (Exit 124). Quota is computed at run start, does not reset mid-run, and failed heavy
  runs still consume it. Check the weekly quota graph before blaming the notebook.
- Crunch's notebook→`main.py` generator **only hoists plain top-level `import X`**. Imports
  inside `try/except` or functions are dropped (or not auto-detected as pip deps). Keep submission
  imports plain and top-level.
- `INFER_PARALLELISM` is RAM-adaptive (`_PER_WORKER_GB`); each ONNX-RF worker peaks ~8-12GB
  during session build. Free the RF blob (`saved["rf_onnx"] = None`) right after building the
  session.
- Only `streaming_detector_stack4.ipynb` (+ the `rankerrf` fallback) are cloud-submittable
  champions. `_*.py` scripts and `exp_*.ipynb` are local research harnesses, NOT submittable.

---

## What is CLOSED (do not reopen without a genuinely new angle)

The ~2.5pt gap to leaders is a **problem-structure / tau-location** gap (oracle: 84% of headroom
= not knowing tau), NOT a feature nibble. Exhausted axes (all NULL, see memory for E-numbers):

- **Marginal single features** (perm-entropy, wasserstein, cvm, Hurst/DFA, spectral-shape,
  variance two-sample, robust restatements, changepoint-as-feature) — feature addition saturated.
- **Learned nets on the window**: CNN / multi-scale CNN / TCN / ModernTCN / Hyena / S4-Mamba, all
  input lenses (z/cusum/z2/dz/adz/rank), capacity scaling, synthetic pretraining — base79
  saturates per-step single-series recent-window info (fingerprint ≈ −0.0011).
- **Sampling distribution** (metric-weighted / bimodal, ranker-side and rf-side) — uniform optimal.
- **Combiner / stacking / gating**: learned metas lose to the fixed-CDF blend 6×; per-step gating
  is real but causally unexploitable (cross-sectional rank couples all series → local gains
  annihilate — the E50 lesson).
- **Whitening** (wholesale stream replacement), **alt residual lenses** (AR order 2/4/16, robust
  Huber), **nonlinear dependence** on AR8 residual, **window-local changepoint locator**,
  **early-detection** focus (metric weight is mid-late, not early).
- Fine knobs: HIST_CAP (peak plateau 150–175, shipped), null_wins (peak 32), scale-set
  (40,100,250 optimal), DENSE_K/STEPS coarse-settled.

**Only living positive of the recent arc = the calibration axis** (studentize base79 vs each
series' own pre-break null). Remaining speculative bets: full-buffer global tau locator (plumbing),
additive whitened block alongside raw, and the untried 1st/2nd-place edge = **model diversity /
independent feature banks + stacking** (though learned metas have lost so far).

---

## Repo map

- `streaming_detector_stack4.ipynb` — champion. `streaming_detector_rankerrf.ipynb` — 2-member
  fallback (LB 62.61%, cloud-green). Other `streaming_detector_*` / `exp_*` = historical variants.
- `_*.py` — research harnesses & probes (e.g. `_calib_rankerrf.py` = the calibration probe that
  produces `czrb_all79`; `_stack_deep.py` = OOF stacking harness; `_compare_pred.py` = TS-AUC
  bit-compare of two prediction parquets; `_par_main.py`/`_ser_main.py` = parallel/serial crunch
  end-to-end, safe to edit).
- `backups/` — champion snapshots (backup-first target). `data/` — competition parquets.
- `README.md` — the experiment-log table (E-numbered). **Document every experiment there**
  (concise, append rows); record tests/observations that informed decisions.
- Memory: `/memories/repo/structural-break.md` (long history), `/memories/session/current-state.md`
  (live plan), `/memories/repo/first-place-2025-alphabot.md`, `/memories/presentation.md`.
