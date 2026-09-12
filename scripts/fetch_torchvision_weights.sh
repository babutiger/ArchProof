#!/usr/bin/env bash
# Restore the local torchvision pretrained-weight cache (models/torchvision_weights/).
#
# The weights are bundled under models/torchvision_weights/hub/checkpoints/ and
# the table scripts point TORCH_HOME there, so a normal run uses the local copy
# with NO download. This script only REPAIRS that mirror if a file is missing
# (download-if-absent: files already present are skipped). Needs network.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=models/torchvision_weights/hub/checkpoints
mkdir -p "$OUT"

python - "$OUT" <<'PY'
import sys, os, urllib.request
from torchvision.models import get_model_weights
out = sys.argv[1]
# architectures whose ONNX appear in the clean panel (torchvision families)
names = ["alexnet","densenet121","densenet169","efficientnet_b0","googlenet",
         "inception_v3","mnasnet1_0","mobilenet_v2","mobilenet_v3_small",
         "regnet_y_400mf","resnet18","resnet34","resnet50","resnet101",
         "resnext50_32x4d","shufflenet_v2_x1_0","squeezenet1_0","vgg11","vgg16",
         "wide_resnet50_2"]
got = skip = fail = 0
for n in names:
    try:
        url = get_model_weights(n).DEFAULT.url
    except Exception as e:
        print(f"  [skip] {n}: no DEFAULT weights ({e})"); fail += 1; continue
    dst = os.path.join(out, os.path.basename(url))
    if os.path.exists(dst):
        print(f"  [have] {os.path.basename(dst)}"); skip += 1; continue
    print(f"  [get ] {os.path.basename(dst)} ...")
    try:
        urllib.request.urlretrieve(url, dst); got += 1
    except Exception as e:
        print(f"  [fail] {n}: {e}"); fail += 1
print(f"downloaded={got} already-present={skip} failed={fail} -> {out}")
PY
echo "done: $(du -sh "$OUT" 2>/dev/null | cut -f1) in $OUT"
