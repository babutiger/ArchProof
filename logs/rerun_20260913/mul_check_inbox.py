import os, sys, onnx, numpy as np, onnxruntime as ort
from onnx import numpy_helper, helper
AR=os.environ["ARCHPROOF_ROOT"]
p=os.path.join(AR,"models/backdoor_graphs/op_sep_tar.onnx"); m=onnx.load(p); g=m.graph
# payload = the Constant feeding /Mul_3
mul3=[n for n in g.node if n.output[0]=="/Mul_3_output_0" or n.name=="/Mul_3"][0]
const_name=[i for i in mul3.input if not i.startswith("/Relu")][0]
cn=[n for n in g.node if const_name in n.output][0]
val=numpy_helper.to_array(cn.attribute[0].t); print("payload const:", const_name, "shape", val.shape, "min/max", val.min(), val.max())
mz=onnx.load(p); cz=[n for n in mz.graph.node if const_name in n.output][0]
cz.attribute[0].t.CopyFrom(numpy_helper.from_array(np.zeros_like(val), cz.attribute[0].t.name))
# expose the Sub tensor feeding /Mul_4 and the gate Mul output as extra outputs
mul4=[n for n in g.node if n.name=="/Mul_4"][0]; sub_name=[i for i in mul4.input if "Flatten" not in i][0]
for extra in (sub_name, "/Mul_3_output_0"):
    m.graph.output.append(helper.make_tensor_value_info(extra, onnx.TensorProto.FLOAT, None))
s=ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]); sz=ort.InferenceSession(mz.SerializeToString(), providers=["CPUExecutionProvider"])
inp=s.get_inputs()[0].name
def contrib(x):
    o=s.run(None,{inp:x}); oz=sz.run(None,{inp:x})[0]
    return np.abs(o[0]-oz).max(), np.abs(o[1]).max(), np.abs(o[2]).max()
rng=np.random.RandomState(0); UB=0.95
cands={"all 0.95 (box corner)":np.full((1,3,32,32),UB,np.float32),"all 0":np.zeros((1,3,32,32),np.float32),"uniform[0,0.95]":rng.uniform(0,UB,(1,3,32,32)).astype(np.float32)}
# CIFAR test images clipped to the box, if bundled
try:
    import pickle, glob
    b=pickle.load(open(glob.glob(os.path.join(AR,"data/cifar-10-batches-py/test_batch"))[0],"rb"),encoding="bytes")
    imgs=b[b"data"][:200].reshape(-1,3,32,32).astype(np.float32)/255.0
    cands["200 CIFAR test imgs clipped to 0.95 (max over imgs)"]=np.clip(imgs,0,UB)
except Exception as e: print("no cifar:", e)
print(f'{"input":50s} {"|M-M_zeroed|inf":>16s} {"|Sub| (post-gate multiplier)":>28s} {"|gate Mul out|":>14s}')
for k,x in cands.items():
    if x.shape[0]>1:
        vals=[contrib(x[i:i+1]) for i in range(x.shape[0])]; best=max(vals,key=lambda t:t[0]); print(f"{k:50s} {best[0]:16.4f} {best[1]:28.4f} {best[2]:14.4f}")
    else:
        c=contrib(x); print(f"{k:50s} {c[0]:16.4f} {c[1]:28.4f} {c[2]:14.4f}")
print("certificate eps = 0.99999994 ; box = [0, 0.95]")
onnx.save(mz, "/tmp/op_sep_tar_zeroed.onnx")
