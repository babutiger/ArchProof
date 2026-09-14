import os, sys, torch, onnx
from onnx import numpy_helper
AR=os.environ.get("ARCHPROOF_ROOT", os.getcwd()); sys.path.insert(0, AR)
from archproof.handcrafted_gdp import H1_SignGated, H2_AvgPoolGated, H3_MulIndicatorGated
g=onnx.load(os.path.join(AR,"models/backdoor_graphs/H1_SignGated.onnx")).graph
ini={i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in g.initializer}
def md(m): 
    sd=m.state_dict(); return max((sd[k].float()-ini[k].float()).abs().max().item() for k in ini)
hyp={}
torch.manual_seed(0); hyp["seed0 fresh"]=md(H1_SignGated())
torch.manual_seed(0); torch.randn(1,3,32,32); hyp["seed0, randn(1,3,32,32) first"]=md(H1_SignGated())
torch.manual_seed(0); H1_SignGated(); hyp["seed0, 2nd construction"]=md(H1_SignGated())
torch.manual_seed(0); H2_AvgPoolGated(); H3_MulIndicatorGated(); hyp["seed0 after H2,H3"]=md(H1_SignGated())
for s in (1,42,1234,2024,2025):
    torch.manual_seed(s); hyp[f"seed{s} fresh"]=md(H1_SignGated())
for k,v in hyp.items(): print(f"{k:38s} maxdiff={v:.4g}")
print("H1 initializer names:", list(ini))
