#!/usr/bin/env bash
# Overnight big-run driver: runs the three experiments SEQUENTIALLY, cheapest first
# (b -> a -> c) so a and b always complete even if the wide (c) run is slow. Each phase
# is crash-guarded: a failure in one phase is logged and the next phase still runs.
#
# A hard TOTAL deadline (default 8h) is enforced: b and a run first, then c is handed its
# REMAINING budget via C_BUDGET so the whole run finishes on time. c's own budget guard then
# stops it gracefully and logs a partial verdict if needed.
set -u

cd "$(dirname "$0")"
PY=.venv/bin/python
export PYTHONWARNINGS=ignore

TOTAL_BUDGET="${TOTAL_BUDGET:-28800}"   # 8h in seconds
C_FLOOR="${C_FLOOR:-1800}"              # skip c if less than this remains
START=$(date +%s)

remaining() { echo $(( START + TOTAL_BUDGET - $(date +%s) )); }

run() {
    local name="$1"; local script="$2"
    echo "===== [$(date '+%F %T')] START $name ($script), $(remaining)s of budget left ====="
    env "$PY" -u "$script"
    local rc=$?
    echo "===== [$(date '+%F %T')] END   $name rc=$rc ====="
}

echo "########## NIGHT BIG-RUN START $(date '+%F %T') SMOKE=${SMOKE:-0} budget=${TOTAL_BUDGET}s ##########"
run "b dependence/early-onset" _night_b.py
run "a model-diversity stack"  _night_a.py

REM=$(remaining)
if [ "$REM" -lt "$C_FLOOR" ]; then
    echo "########## SKIP c: only ${REM}s left (< ${C_FLOOR}s floor) ##########"
else
    export C_BUDGET=$(( REM - 300 ))   # leave a 5-min margin
    run "c Alphabot wide-stack (C_BUDGET=${C_BUDGET}s)" _night_c.py
fi
echo "########## NIGHT BIG-RUN DONE  $(date '+%F %T') ##########"
