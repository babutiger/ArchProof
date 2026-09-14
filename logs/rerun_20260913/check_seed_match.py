import os, sys, torch, onnx
from onnx import numpy_helper
AR=os.environ.get("ARCHPROOF_ROOT", os.getcwd()); sys.path.insert(0, AR)
from backdoored_models import op_sep_tar_backdoor, op_int_tar_backdoor_001
from archproof.handcrafted_gdp import H1_SignGated, H2_AvgPoolGated, H3_MulIndicatorGated
cases=[("op_sep_tar",op_sep_tar_backdoor),("op_int_tar_L001",op_int_tar_backdoor_001),("H1_SignGated",H1_SignGated),("H2_AvgPoolGated",H2_AvgPoolGated),("H3_MulIndicatorGated",H3_MulIndicatorGated)]
def inits(name):
    g=onnx.load(os.path.join(AR,"models/backdoor_graphs",name+".onnx")).graph
    return {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in g.initializer}
def compare(m, ini):
    sd={k:v.detach().float() for k,v in m.state_dict().items()}
    matched=[k for k in sd if k in ini]
    if not matched:  # fall back: match by shape order
        return None
    diffs=[(sd[k]-ini[k].float()).abs().max().item() for k in matched if sd[k].shape==ini[k].shape]
    return len(matched), len(sd), max(diffs) if diffs else float("nan")
for name,fn in cases:
    ini=inits(name)
    r0=compare(fn().eval(), ini)                       # unseeded, as run_e1_final.py does today
    torch.manual_seed(0); r1=compare(fn().eval(), ini) # seeded 0 before construction
    print(f"{name:22s} n_init={len(ini):3d}  unseeded(matched/params/maxdiff)={r0}  seed0={r1}")
