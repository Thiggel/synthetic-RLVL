#!/usr/bin/env python3
"""Check every midtraining number printed in the paper against its source JSON.

Run on alex, where the eval bundles live, with the two .tex files copied to
/tmp. Prints one line per mismatch and a summary; exit status is non-zero if
anything disagrees.
"""
import glob
import json
import os
import re
import sys

V = "/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results"
ARMS = ["control", "longdoc", "logic_band25", "nl_exact_band25", "condensed_logic_band25"]
TEX = {}
for name in ("main", "appendix_details"):
    TEX[name] = open("/tmp/%s.tex" % name).read()

problems = []
checked = 0


def close(a, b, tol=0.051):
    return abs(a - b) < tol


def graded(root, suffix):
    out = {}
    for a in ARMS:
        d = glob.glob("%s/qwen25_7b_longwin_%s_2p5b_dolci_100k_lr5em6%s" % (root, a, suffix))
        if not d:
            continue
        fs = sorted(glob.glob(d[0] + "/**/results*.json", recursive=True))
        if not fs:
            continue
        r = json.load(open(fs[-1]))["results"]
        out[a] = {k.replace("synthrlvl_deduction_", ""): v["exact_match,none"]
                  for k, v in r.items() if k.startswith("synthrlvl_deduction_")}
    return out


def row_of(tex, label, lead):
    """Return the numeric cells of the table row beginning with `lead`."""
    i = tex.index("\\label{%s}" % label)
    block = tex[i:tex.index("\\end{tabular}", i)]
    for line in block.split("\\\\"):
        # a rule can be glued to the front of a row
        line = re.sub(r"^\\(top|mid|bottom)rule", "", line.strip()).strip()
        if line.startswith(lead):
            cells = [c.strip() for c in line.split("&")[1:]]
            nums = []
            for c in cells:
                m = re.search(r"-?\d+\.\d+", c)
                if m:
                    nums.append(float(m.group()))
                elif not nums:
                    continue  # a second lead cell, e.g. the condition name
                else:
                    nums.append(None)
            return nums
    return None


# --- Table 3: ProofWriter greedy, seed 3407 ------------------------------
g7 = graded(V + "/qwen25_longwin_graded_deduction_20260906", "")
tex_names = {"control": "\\ctrl{}", "longdoc": "\\longdoc{}", "logic_band25": "\\formal{}",
             "nl_exact_band25": "\\nat{}", "condensed_logic_band25": "\\condensed{}"}
for arm in ARMS:
    nums = row_of(TEX["main"], "tab:transfer-pw", tex_names[arm])
    if nums is None:
        problems.append("tab:transfer-pw: no row for %s" % arm)
        continue
    for j, depth in enumerate(["pw_d0", "pw_d1", "pw_d2", "pw_d3", "pw_d5"]):
        src = g7[arm][depth] * 100
        checked += 1
        if not close(nums[j], src):
            problems.append("tab:transfer-pw %s %s: paper %.1f vs source %.1f"
                            % (arm, depth, nums[j], src))

# --- Table 4: near-transfer greedy + pass@16, seed 3407 ------------------
for arm in ARMS:
    for metric, lead in (("greedy", "greedy & " + tex_names[arm]),
                         ("pass16", "\\passk{16} & " + tex_names[arm])):
        nums = row_of(TEX["main"], "tab:transfer-passk", lead)
        if nums is None:
            problems.append("tab:transfer-passk: no %s row for %s" % (metric, arm))
            continue
        for j, d in enumerate([5, 10, 15, 20, 25]):
            if nums[j] is None:
                continue
            if metric == "greedy":
                src = g7[arm]["bp_cot_d%d" % d] * 100
            else:
                f = "%s/qwen25_longwin_passk_20260908_eosfix/%s/metrics_synthrlvl_deduction_bp_cot_d%d.json" % (V, arm, d)
                if not os.path.exists(f):
                    continue
                src = json.load(open(f))["pass_at_k"]["16"] * 100
            checked += 1
            if not close(nums[j], src):
                problems.append("tab:transfer-passk %s %s d%d: paper %.1f vs source %.1f"
                                % (metric, arm, d, nums[j], src))

# --- Appendix: sampled downstream, seed 3407 -----------------------------
DOWN = V + "/qwen25_longwin_passk_downstream_20260908"
rows = [("GSM8K", "standard", "gsm8k", "pass_at_k"),
        ("MATH-500", "standard", "hendrycks_math500", "pass_at_k"),
        ("HotpotQA", "multihop", "synthrlvl_longbench_hotpotqa_tagged", "pass_at_k"),
        ("2WikiMultihopQA", "multihop", "synthrlvl_longbench_2wikimqa_tagged", "pass_at_k"),
        ("MuSiQue", "multihop", "synthrlvl_longbench_musique_tagged", "pass_at_k")]
for lead, group, task, key in rows:
    nums = row_of(TEX["appendix_details"], "tab:sampled-downstream", lead)
    if nums is None:
        problems.append("tab:sampled-downstream: no row for %s" % lead)
        continue
    nums = [n for n in nums if n is not None]
    for j, arm in enumerate(ARMS):
        f = "%s/%s/%s/metrics_%s.json" % (DOWN, group, arm, task)
        if not os.path.exists(f):
            continue
        src = json.load(open(f))[key]["16"] * 100
        checked += 1
        if j < len(nums) and not close(nums[j], src):
            problems.append("tab:sampled-downstream %s %s: paper %.1f vs source %.1f"
                            % (lead, arm, nums[j], src))

print("checked %d printed values" % checked)
if problems:
    print("\nMISMATCHES (%d):" % len(problems))
    for p in problems:
        print("  " + p)
    sys.exit(1)
print("all printed midtraining values match their source files")
