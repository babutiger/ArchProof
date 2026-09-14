#!/usr/bin/env bash
# Run the patched e1 driver twice and compare the two records byte-for-byte on
# every eps_star/delta: proves the PGD columns are now deterministic.
source ~/anaconda3/etc/profile.d/conda.sh; conda activate archproof_repro
cd /home/xu/mymac_remotedir/CCS/artifact; export PYTHONPATH=$PWD ARCHPROOF_ROOT=$PWD OMP_NUM_THREADS=8
OUT=logs/rerun_20260913
bash scripts/00_build_benchmark_models.sh
for i in 1 2; do
  echo "== run $i start $(date)"
  python archproof/run_e1_final.py 2>/dev/null | grep -v "^  [a-zA-Z]" | tail -25
  cp -f benchmark/e1_final_results.json $OUT/e1_seeded_run$i.json
  echo "== run $i exit $? $(date)"
done
python - <<'PY'
import json
a=json.load(open("logs/rerun_20260913/e1_seeded_run1.json")); b=json.load(open("logs/rerun_20260913/e1_seeded_run2.json"))
ra={r["name"]:r for r in (a if isinstance(a,list) else a.get("results",[]))}; rb={r["name"]:r for r in (b if isinstance(b,list) else b.get("results",[]))}
diff=[k for k in ra if ra[k]!=rb[k]]
print("models:",len(ra),"records differing between run1 and run2:",len(diff),diff[:10])
bd=[r for r in ra.values() if r["type"]=="backdoor"]
print("graph maxdiff (module vs onnxruntime) max over 22:",max(r.get("pgd_graph_maxdiff") or 0 for r in bd),"unmatched:",[r["name"] for r in bd if r.get("pgd_graph_unmatched")])
for k in ("H1_SignGated","H2_AvgPoolGated","H3_MulIndicatorGated","op_sep_tar"): print(k, ra[k]["verdict"], ra[k]["eps_star"], ra[k]["delta"])
PY
echo "== done $(date)"
