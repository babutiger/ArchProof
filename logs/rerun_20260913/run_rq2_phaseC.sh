#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact
export PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "== start $(date)"; nice -n 5 python scripts/run_rq2_phaseC_slice.py; echo "== exit $? $(date)"
