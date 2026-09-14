#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact
export PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD OMP_NUM_THREADS=8
echo "== start $(date)"; python logs/rerun_20260913/legacy_map_71.py 2>/dev/null; echo "== exit $? $(date)"
