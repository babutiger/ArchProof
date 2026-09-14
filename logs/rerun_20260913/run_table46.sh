#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact
export PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD OMP_NUM_THREADS=8 TORCH_HOME=$PWD/models/torchvision_weights
echo "== start $(date)"; bash reproduce/table46_appx-torchvision.sh; echo "== exit $? $(date)"
