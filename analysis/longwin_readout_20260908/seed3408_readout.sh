#!/usr/bin/env bash
# One-shot readout of the seed-3408 replicate (run after 4200590/4200591 finish).
V=/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results
cd ~/synthetic-RLVL
echo "### graded greedy seed3408"; python3 analysis/longwin_readout_20260908/graded_table.py $V/qwen25_longwin_graded_deduction_20260906_seed3408
echo "### per-class seed3408"; python3 analysis/longwin_readout_20260908/pw_per_class.py $V/qwen25_longwin_graded_deduction_20260906_seed3408
echo "### passk seed3408"; python3 - <<PY
import json,os
P="$V/qwen25_longwin_passk_20260907_seed3408"
print("arm,depth,pass1,pass16,maj16,sampled_tokens")
for arm in ["control","longdoc","logic_band25","nl_exact_band25","condensed_logic_band25"]:
    for d in [5,10,15,20,25]:
        f=f"{P}/{arm}/metrics_synthrlvl_deduction_bp_cot_d{d}.json"
        if os.path.exists(f):
            m=json.load(open(f)); print(f"{arm},{d},{m[pass_at_k][1]:.4f},{m[pass_at_k][16]},{m[maj_at_k][16]},{m[sampled][mean_response_tokens]:.0f}")
PY
echo "### bootstrap seed3408"; python3 analysis/longwin_readout_20260908/passk_bootstrap.py $V/qwen25_longwin_passk_20260907_seed3408
