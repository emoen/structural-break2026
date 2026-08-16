#!/usr/bin/env bash
# Overnight NN fusion queue. Runs sequentially; each writes its own log.
# Order = priority (most informative first) in case it doesn't all finish.
#   1. scalar_pre  : synthetic-pretrain K10 S80 (the missing number: synthetic's contribution)
#   2. embed_pre   : + embedding fusion (64-d vector instead of scalar prob)
#   3. k15_pre     : max stack (more seeds + more data) with pretrain
#   4. k15_nopre   : control (same, no pretrain) -> isolates synthetic's effect
set -u
cd /home/endrem/prosjekt/structural-break2026
MASTER=_overnight_nn_master.log
: > "$MASTER"

run() {
  local name="$1" envs="$2" out="$3"
  echo "=== START $name $(date '+%F %T') ===" | tee -a "$MASTER"
  # shellcheck disable=SC2086
  env $envs .venv/bin/python -u _nn_fusion.py > "$out" 2>&1
  echo "=== END $name rc=$? $(date '+%F %T') ===" | tee -a "$MASTER"
  grep -E 'champion=|delta=' "$out" | tail -1 | tee -a "$MASTER"
  echo "" | tee -a "$MASTER"
}

run scalar_pre "PRETRAIN=1 SYNTH_DIR=synthetic ENSEMBLE=10 STEPS=80"          _ov_scalar_pre.log
run embed_pre  "PRETRAIN=1 EMBED=1 SYNTH_DIR=synthetic ENSEMBLE=10 STEPS=80"   _ov_embed_pre.log
run k15_pre    "PRETRAIN=1 SYNTH_DIR=synthetic ENSEMBLE=15 STEPS=120"          _ov_k15_pre.log
run k15_nopre  "ENSEMBLE=15 STEPS=120"                                          _ov_k15_nopre.log

echo "=== ALL DONE $(date '+%F %T') ===" | tee -a "$MASTER"
echo "--- summary ---" | tee -a "$MASTER"
grep -H 'champion=' _ov_*.log | tee -a "$MASTER"
