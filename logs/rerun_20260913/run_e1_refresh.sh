#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact; export PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD OMP_NUM_THREADS=8
cp -f benchmark/e1_final_results.json logs/rerun_20260913/old_records/e1_final_results.beforeH3fix.json
echo "== start $(date)"; bash scripts/00_build_benchmark_models.sh; python archproof/run_e1_final.py 2>/dev/null | tail -3; echo "== exit $? $(date)"
