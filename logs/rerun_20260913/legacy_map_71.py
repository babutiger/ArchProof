import sys, os, json, warnings, collections, time; warnings.filterwarnings("ignore"); sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from archproof.verify import verify_model
from verdict_class import map_to_3class
nat = json.load(open("benchmark/baselines_natural49_complete.json"))["natural49_models"]
IN = {"op_sep_tar","op_sep_un","op_sha_tar","op_sha_un","op_int_tar","op_int_un","op_int_tar_L01","op_int_tar_L001","op_int_tar_L0001","H2_AvgPoolGated","H3_MulIndicatorGated"}
items = [(f[:-5], "backdoor-in-class" if f[:-5] in IN else "backdoor-out-of-class", f"models/backdoor_graphs/{f}") for f in sorted(os.listdir("models/backdoor_graphs")) if f.endswith(".onnx")]
items += [(n, "clean-natural49", f"models/clean_panel/{n}.onnx") for n in sorted(nat)]
tally = collections.defaultdict(collections.Counter); rows=[]
for n, sub, p in items:
    t0=time.time(); l = verify_model(p)
    v = map_to_3class(l.verdict, n_syn=l.n_gdp_syntactic, n_adm=l.n_gdp_admitted)
    c = "pos" if "POSITIVE" in v else "neg" if "NEGATIVE" in v else "unc"; tally[sub][c] += 1
    rows.append(dict(name=n, subset=sub, legacy=l.verdict, n_syn=l.n_gdp_syntactic, n_adm=l.n_gdp_admitted, mapped=v, sec=round(time.time()-t0,1)))
    print(f"[{len(rows):2d}/71] {n:22s} {sub:22s} legacy={l.verdict:18s} n_syn={l.n_gdp_syntactic} n_adm={l.n_gdp_admitted} -> {v}", flush=True)
print("=== legacy verify_model + map_to_3class tally ===")
for s in sorted(tally): print(f"   {s:24s} pos={tally[s]['pos']:2d} neg={tally[s]['neg']:2d} unc={tally[s]['unc']:2d}")
json.dump(rows, open("logs/rerun_20260913/legacy_map_71.json","w"), indent=1)
