#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact
export PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD OMP_NUM_THREADS=8 TORCH_HOME=$PWD/models/torchvision_weights HF_ENDPOINT=https://hf-mirror.com
echo "== start $(date)"; python scripts/run_tau_sys_sweep.py 2>&1 | grep -v "Warning\|onnxruntime:"; echo "== exit ${PIPESTATUS[0]} $(date)"
