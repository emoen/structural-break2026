#!/usr/bin/env bash
# Focused embed-fusion queue (embed64 is the winning lever: t=2.56 last night).
# Priority: isolate synthetic, then robustness, then push the stack.
#   1. embed_nopre_k10s80 : embed WITHOUT synthetic  -> does embed need synth? (synth hurt scalar)
#   2. embed_pre_k10s80_s7: winner config, DIFFERENT seed -> robustness of t=2.56
#   3. embed_pre_k15s120  : more seeds + more data + synth -> push higher
set -u
cd /home/endrem/prosjekt/structural-break2026
MASTER=_ov2_master.log
: > "$MASTER"

run() {
  local name="$1" envs="$2" out="$3"
  echo "=== START $name $(date '+%F %T') ===" | tee -a "$MASTER"
  # shellcheck disable=SC2086
  env $envs .venv/bin/python -u _nn_fusion.py > "$out" 2>&1
  echo "=== END $name rc=$? $(date '+%F %T') ===" | tee -a "$MASTER"
  grep -E 'champion=' "$out" | tail -1 | tee -a "$MASTER"
  echo "" | tee -a "$MASTER"
}

run embed_nopre_k10s80  "EMBED=1 ENSEMBLE=10 STEPS=80"                                        _ov2_embed_nopre_k10s80.log
run embed_pre_k10s80_s7 "PRETRAIN=1 EMBED=1 SYNTH_DIR=synthetic ENSEMBLE=10 STEPS=80 SEED=7"  _ov2_embed_pre_k10s80_s7.log
run embed_pre_k15s120   "PRETRAIN=1 EMBED=1 SYNTH_DIR=synthetic ENSEMBLE=15 STEPS=120"        _ov2_embed_pre_k15s120.log

echo "=== ALL DONE $(date '+%F %T') ===" | tee -a "$MASTER"
grep -H 'champion=' _ov2_*.log | tee -a "$MASTER"
