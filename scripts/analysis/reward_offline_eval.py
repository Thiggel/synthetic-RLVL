#!/usr/bin/env python
"""Offline check of the Stage-2 reward arms on existing policy outputs (no training).

For every <run>/rl_gate_dolci/generations.jsonl (greedy gate outputs) this recomputes
formal_rewards.components and reports per model and arm: the fraction of outputs with
nonzero reward, the mean, and the mean reward of valid vs non-valid outputs. It also
prints the highest-scoring non-valid outputs under the dense arms, to look for reward
hacking before any GRPO run. Writes reports/figures/stage2_reward_offline.png and
analysis/stage2_reward_offline.json.
"""
import collections
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from formal_rewards import components  # noqa: E402

ROOT = "/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925"
TEST = "/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl"
ARMS = {"correct": lambda c: c["correct"], "correct_x_valid": lambda c: c["correct"] * c["valid"],
        "gvc": lambda c: (c["grammatical"] + c["valid"] + c["correct"]) / 3, "valid": lambda c: c["valid"],
        "lines": lambda c: c["lines"]}


def main():
    test = {json.loads(line)["id"]: json.loads(line) for line in open(TEST)}
    res, worst = {}, []
    for f in sorted(glob.glob(f"{ROOT}/*/rl_gate_dolci/generations.jsonl")):
        run = f.split("/")[-3]
        m = run.split("_")[1] + " " + run.split("_")[4].replace("p0", "p").replace("p", "p")
        rows = [json.loads(line) for line in open(f)]
        comps = []
        for r in rows:
            rec = {**test[r["id"]], "system_answerable": test[r["id"]]["system_answerable"]}
            c = components(rec, r["generation"])
            comps.append(c)
            if not c["valid"] and c["lines"] > 0:
                worst.append((c["lines"], c["lines_raw"], m, r["id"], r["generation"][:1500]))
        d = {}
        for a, fn in ARMS.items():
            v = [fn(c) for c in comps]
            d[a] = {"nonzero": sum(x > 0 for x in v) / len(v), "mean": sum(v) / len(v)}
        d["lines_valid_mean"] = (lambda xs: sum(xs) / max(1, len(xs)))([c["lines"] for c in comps if c["valid"]])
        d["lines_invalid_mean"] = (lambda xs: sum(xs) / max(1, len(xs)))([c["lines"] for c in comps if not c["valid"]])
        d["mean_n_ok"] = sum(c["n_ok"] for c in comps) / len(comps)
        d["mean_n_parsed"] = sum(c["n_parsed"] for c in comps) / len(comps)
        d["mean_n_steps"] = sum(c["n_steps"] for c in comps) / len(comps)
        res[m] = d
        print(m, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()})
                             for k, v in d.items()}), flush=True)
    out = HERE.parent / "analysis" / "stage2_reward_offline.json"
    out.write_text(json.dumps(res, indent=1))
    worst.sort(key=lambda x: -x[0])
    for w in worst[:6]:
        print("\n=== lines=%.3f raw=%d %s %s\n%s" % w)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in res if res[m]["correct"]["mean"] >= 0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.6))
    w = 0.8 / len(ARMS)
    for j, a in enumerate(ARMS):
        ax.bar([i + j * w for i in range(len(models))], [res[m][a]["nonzero"] for m in models], w, label=a)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(models))], models)
    ax.set_ylabel("fraction of gate outputs\nwith nonzero reward")
    ax.set_title("Reward density per arm on greedy gate outputs (950 items)")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(HERE.parent / "reports" / "figures" / "stage2_reward_offline.png", dpi=130)


if __name__ == "__main__":
    main()
