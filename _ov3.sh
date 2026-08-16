#!/usr/bin/env bash
# Q1: does the NN feature add on top of the SHIPPED TUNED champion (max_features=0.5)?
# Memory-safe: K10/S80 (proven to fit 31GB), no synthetic pretrain (didn't robustly help + saves RAM).
# Two seeds on embed => robustness check (embed t=2.56 was a seed artifact on the OLD champion).
set -u
cd /home/endrem/prosjekt/structural-break2026
MASTER=_ov3_master.log
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

run tuned_embed_s42  "TUNED=1 EMBED=1 ENSEMBLE=10 STEPS=80 SEED=42"  _ov3_tuned_embed_s42.log
run tuned_embed_s7   "TUNED=1 EMBED=1 ENSEMBLE=10 STEPS=80 SEED=7"   _ov3_tuned_embed_s7.log
run tuned_scalar_s42 "TUNED=1 ENSEMBLE=10 STEPS=80 SEED=42"          _ov3_tuned_scalar_s42.log

echo "=== ALL DONE $(date '+%F %T') ===" | tee -a "$MASTER"
grep -H 'champion=' _ov3_*.log | tee -a "$MASTER"
