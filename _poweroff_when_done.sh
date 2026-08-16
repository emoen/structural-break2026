#!/usr/bin/env bash
# Waits for the TabPFN fresh-seed job to finish, then powers off the machine.
PID=578052
LOG=/home/endrem/prosjekt/structural-break2026/_poweroff_watch.log
echo "$(date) watching PID $PID" > "$LOG"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "$(date) PID $PID exited; syncing and powering off" >> "$LOG"
sync
sleep 5
systemctl poweroff >> "$LOG" 2>&1 \
  || /usr/sbin/poweroff >> "$LOG" 2>&1 \
  || /sbin/shutdown -h now >> "$LOG" 2>&1
echo "$(date) poweroff command returned $?" >> "$LOG"
