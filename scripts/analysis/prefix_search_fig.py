#!/usr/bin/env python
"""Figure: in-system prompts solved per round of checker-guided prefix resampling (ei_prefix_search.py stats.json)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stats = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                       "/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925/"
                       "qwen35_2b_dolci_rlvlgen_p50_lr5em6_seed3407/ei_search_r1/stats.json"))
out = Path(__file__).resolve().parents[2] / "reports/figures/stage2_prefix_search.png"
rounds = stats["rounds"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
cum = 0
for bench in stats["per_bench"]:
    ys = [r["solved"].get(bench, 0) for r in rounds]
    a.plot([r["round"] for r in rounds], ys, marker="o", label=bench.removeprefix("dolci_"))
a.set_xlabel("round (0 = whole-proof sampling)"); a.set_ylabel("prompts with an in-system proof")
a.set_xticks([r["round"] for r in rounds]); a.legend(); a.grid(alpha=.3)
a.set_title("cumulative prompts solved")
samp = [r["n_samples"] for r in rounds]
b.bar([r["round"] for r in rounds], [r["n_valid"] / s * 100 for r, s in zip(rounds, samp)], color="C2")
b.set_xlabel("round"); b.set_ylabel("valid samples (%)"); b.set_title("yield per sample")
b.set_xticks([r["round"] for r in rounds]); b.grid(alpha=.3, axis="y")
fig.suptitle("2B p50: checker-guided prefix resampling on 3,191 RL prompts")
fig.tight_layout(); fig.savefig(out, dpi=130)
print(out, json.dumps([(r["round"], r["n_samples"], r["n_valid"], r["solved"]) for r in rounds]))
