import os, sys, tempfile
sys.path.insert(0,(os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torchvision.models as tvm, onnx
from collections import Counter
for ctor in ["efficientnet_b0","mobilenet_v3_small"]:
    print(f"### {ctor}",flush=True)
    m=getattr(tvm,ctor)(weights=None).eval()
    for opset in [11,13,17]:
        p=os.path.join(tempfile.mkdtemp(),f"{ctor}_{opset}.onnx")
        try:
            torch.onnx.export(m, torch.randn(1,3,224,224), p, opset_version=opset, keep_initializers_as_inputs=True)
            g=onnx.load(p).graph
            c=Counter(n.op_type for n in g.node)
            pool=", ".join(f"{k}={c[k]}" for k in ["GlobalAveragePool","ReduceMean","ReduceSum","AveragePool","Sigmoid","HardSigmoid","Mul","Conv"] if c.get(k))
            print(f"  opset{opset}: {pool}",flush=True)
        except Exception as e:
            print(f"  opset{opset}: ERROR {type(e).__name__}: {str(e)[:80]}",flush=True)
print("\n若任一opset出现ReduceMean/ReduceSum -> 论文caption的机制描述对它的导出成立(版本差异,非错);若所有opset都是GlobalAveragePool -> caption机制描述与导出不符。")
