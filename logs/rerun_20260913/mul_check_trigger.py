import os, sys, glob, pickle, onnx, numpy as np, onnxruntime as ort, torch
from onnx import numpy_helper, helper
AR=os.environ["ARCHPROOF_ROOT"]; sys.path.insert(0, AR)
import utils
p=os.path.join(AR,"models/backdoor_graphs/op_sep_tar.onnx"); m=onnx.load(p); g=m.graph
mul3=[n for n in g.node if n.name=="/Mul_3"][0]; const_name=[i for i in mul3.input if not i.startswith("/Relu")][0]
mz=onnx.load(p); cz=[n for n in mz.graph.node if const_name in n.output][0]
val=numpy_helper.to_array(cz.attribute[0].t); cz.attribute[0].t.CopyFrom(numpy_helper.from_array(np.zeros_like(val), cz.attribute[0].t.name))
mul4=[n for n in g.node if n.name=="/Mul_4"][0]; sub_name=[i for i in mul4.input if "Flatten" not in i][0]
for extra in (sub_name, "/Mul_3_output_0"): m.graph.output.append(helper.make_tensor_value_info(extra, onnx.TensorProto.FLOAT, None))
s=ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]); sz=ort.InferenceSession(mz.SerializeToString(), providers=["CPUExecutionProvider"]); inp=s.get_inputs()[0].name
def contrib(x):
    o=s.run(None,{inp:x}); oz=sz.run(None,{inp:x})[0]; return np.abs(o[0]-oz).max(), np.abs(o[1]).max(), np.abs(o[2]).max()
b=pickle.load(open(os.path.join(AR,"data/cifar-10-batches-py/test_batch"),"rb"),encoding="bytes")
imgs=b[b"data"][:300].reshape(-1,3,32,32).astype(np.float32)/255.0
# apply the benchmark's own trigger if a helper exists, else a 3x3 chequerboard of 1/0 in the corner
helper_names=[n for n in dir(utils) if "trigger" in n.lower() and ("add" in n.lower() or "apply" in n.lower() or "insert" in n.lower() or "poison" in n.lower())]
print("trigger-apply helpers in utils:", helper_names)
def add_cheq(x):
    x=x.copy(); pat=np.indices((3,3)).sum(0)%2; x[:,:, :3, :3]=pat[None,None].astype(np.float32); return x
xt=add_cheq(imgs)
ind=utils.op_indicator_trigger(torch.from_numpy(xt)); print("detector fires on triggered imgs (min/max indicator):", float(ind.min()), float(ind.max()), "| pixel range of triggered inputs:", float(xt.min()), float(xt.max()))
vals=[contrib(xt[i:i+1]) for i in range(xt.shape[0])]
c=np.array([v[0] for v in vals]); sub=np.array([v[1] for v in vals]); gate=np.array([v[2] for v in vals])
print(f"triggered CIFAR imgs (n={len(c)}): |M-M_zeroed|inf  max={c.max():.4f} min={c.min():.4f} | Sub multiplier max={sub.max():.4f} | gate Mul out max={gate.max():.4f}")
print("certificate eps = 0.99999994 ; violated on", int((c>0.99999994).sum()), "of", len(c), "triggered inputs")
xc=np.clip(xt,0,0.95); vals=[contrib(xc[i:i+1]) for i in range(xc.shape[0])]; print("same inputs clipped to 0.95: max contribution", max(v[0] for v in vals))
