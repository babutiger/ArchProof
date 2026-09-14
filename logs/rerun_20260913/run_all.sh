#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact
export TORCH_HOME=$PWD/models/torchvision_weights PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD
echo "== env: $(python -c 'import sys,torch,torchvision,onnx,onnxruntime,numpy;print(sys.version.split()[0],torch.__version__,torchvision.__version__,onnx.__version__,onnxruntime.__version__,numpy.__version__)')"
echo "== start $(date)"
echo "== [1/3] table07 (run_e1_final.py, 22+97 models) =="; bash reproduce/table07_f1-baseline.sh; echo "== table07 exit $? $(date)"
echo "== [2/3] table06 (run_e2_expanded.py) =="; bash reproduce/table06_verdict-dist.sh; echo "== table06 exit $? $(date)"
echo "== [3/3] e9 backbones (run_e9_backbones.py) =="; python archproof/run_e9_backbones.py; echo "== e9 exit $? $(date)"
echo "== done $(date)"
