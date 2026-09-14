import os, sys, onnx, numpy as np
AR=os.environ["ARCHPROOF_ROOT"]; sys.path.insert(0, AR)
from archproof.verify_phaseC import verify_model_phaseC
p=os.path.join(AR,"models/backdoor_graphs/op_sep_tar.onnx")
r=verify_model_phaseC(p)
print("verdict:", r.verdict_phaseC, "eps:", r.epsilon_phaseC)
for k in ("gate_report","additive_certified","admitted","gates","epsilon_blowup"):
    if hasattr(r,k): print(k, "=", getattr(r,k))
m=onnx.load(p); g=m.graph
prod={o:n for n in g.node for o in n.output}
inits={i.name for i in g.initializer}
# find Mul nodes and walk from each Mul to the output, printing op chain
def chain(t, depth=0, seen=None):
    seen=seen or set(); out=[]
    cons=[n for n in g.node if t in n.input]
    for n in cons:
        out.append(n.op_type+("(init)" if any(i in inits for i in n.input) else ""))
        for o in n.output: out+=chain(o, depth+1, seen)
    return out
for n in g.node:
    if n.op_type=="Mul":
        srcs=[("init" if i in inits else prod[i].op_type if i in prod else "input") for i in n.input]
        print("Mul", n.name or n.output[0], "inputs:", srcs, "-> downstream:", " > ".join(chain(n.output[0])[:12]))
