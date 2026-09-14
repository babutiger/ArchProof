#!/usr/bin/env bash
AR=/home/xu/mymac_remotedir/CCS/artifact
cd "$AR" || exit 1
echo "driver sha: $(sha256sum archproof/run_e1_final.py | cut -c1-16)"
setsid nohup bash "$AR/logs/rerun_20260913/run_e1_seeded_twice.sh" > "$AR/logs/rerun_20260913/e1_seeded_twice.log" 2>&1 < /dev/null &
sleep 8
pgrep -af "run_e1_seeded_twice|run_e1_final" | grep -v pgrep | cut -c1-100
echo "--- log"; cat "$AR/logs/rerun_20260913/e1_seeded_twice.log"
