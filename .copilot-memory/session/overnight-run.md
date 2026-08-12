# Overnight battery run (launched 2026-08-11 00:11)

Process: `_overnight.py` PID 103399, nohup+disown, UNSANDBOXED. Survives access revocation.
ETA ~4h. Keep machine awake/plugged in.

## Files
- `_overnight.py` runner, `_feat_lib.py` feature library (34 EXTRA groups + 6 combos)
- Output (workspace root): `_overnight_results.tsv` (1 flushed row/candidate),
  `_overnight_summary.txt` (ranked at end), `_overnight.log` (stdout)

## What it tests
- Section A: base14 @ WINDOW=75 vs 130 across 10 folds (settles borderline 5-fold t=1.73).
  Smoke (2-fold) already showed 130 winning t=+2.45.
- Section B: base14 vs base14+group for 40 feature candidates @ WINDOW=75 on first 5 folds
  (== _val_harness folds, so base14 reference == 0.5511). Paired t vs base14.

## Ship rule (morning)
- WINDOW=130: ship if 10-fold t>=2 & positive (free on inference). -> document E11.
- Feature groups: ship any with |t|>=2 & delta>0. -> document E12.
- Review: `cat _overnight_summary.txt`; verify process done (pgrep) or check DONE line in log.

## Reminders
- Notebook = streaming_detector.ipynb (E7 config, WINDOW=75). Re-read before editing.
- Record LB result before next experiment. Update README.md + /memories/repo/structural-break.md.
