#!/usr/bin/env python
"""Held-out gate (rl_gate_dolci, 950 items, greedy) of the Stage-2 GRPO checkpoints.

Reads <run>/<ckpt>/rl_gate_dolci/generations.jsonl (scripts/slurm/jobs/grpo_gate_ckpts_in_6964.sh)
and rescores every generation with scripts/formal_rewards.components, i.e. with the hardened
Stage-2 validity (>= 1 checked derived line under the conclusion, no circular `given`).
The eval's own `valid` (rlvl strict ok + grounded) is reported as valid_eval next to it.
Writes analysis/stage2_gate_ckpts.json and reports/figures/stage2_gate_ckpts.png.
Run with .venv_rlvl_grpo and PYTHONPATH=RLVL-next/gen:RLVL-next/rlvl/python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from formal_rewards import components  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
SFT = DATA / "formal_mixture_sft_20260925"
GRPO = DATA / "grpo_formal_20260928"
TEST = DATA / "datasets/rl_gate_dolci_instruct_20260928/test.jsonl"
MODELS = [
    ("SFT p0", SFT / "qwen35_2b_dolci_rlvlgen_p00_lr5em6_seed3407"),
    ("SFT p50", SFT / "qwen35_2b_dolci_rlvlgen_p50_lr5em6_seed3407"),
    ("G0 correct (p0) @50", GRPO / "2b_p0_G0_correct_bal/checkpoint-50"),
    ("G0 correct (p0) @100", GRPO / "2b_p0_G0_correct_bal/checkpoint-100"),
    ("G0 correct (p0) @150", GRPO / "2b_p0_G0_correct_bal/checkpoint-150"),
    ("G0 correct (p0) final", GRPO / "2b_p0_G0_correct_bal/final"),
    ("G1 correct @100", GRPO / "2b_p50_G1_correct_bal/checkpoint-100"),
    ("G1 correct @150", GRPO / "2b_p50_G1_correct_bal/checkpoint-150"),
    ("G1 correct final", GRPO / "2b_p50_G1_correct_bal/final"),
    ("G5 lines @50", GRPO / "2b_p50_G5_lines_bal/checkpoint-50"),
    ("G5 lines @100", GRPO / "2b_p50_G5_lines_bal/checkpoint-100"),
    ("G3b gvc hardened @50", GRPO / "2b_p50_G3b_gvc_hard/checkpoint-50"),
    ("G3b gvc hardened @100", GRPO / "2b_p50_G3b_gvc_hard/checkpoint-100"),
    ("G3b gvc hardened @150", GRPO / "2b_p50_G3b_gvc_hard/checkpoint-150"),
    ("G3b gvc hardened final", GRPO / "2b_p50_G3b_gvc_hard/final"),
    ("G5c lines x format @50", GRPO / "2b_p50_G5c_linesfmt/checkpoint-50"),
    ("G5c lines x format @100", GRPO / "2b_p50_G5c_linesfmt/checkpoint-100"),
    ("G6 frac_hard @50", GRPO / "2b_p50_G6_frachard/checkpoint-50"),
    ("G6 frac_hard @100", GRPO / "2b_p50_G6_frachard/checkpoint-100"),
    ("G6 frac_hard @150", GRPO / "2b_p50_G6_frachard/checkpoint-150"),
    ("G6 frac_hard final", GRPO / "2b_p50_G6_frachard/final"),
    ("G6g frac_hard + gen @50", GRPO / "2b_p50_G6g_frachard_gen/checkpoint-50"),
    ("G6g frac_hard + gen @100", GRPO / "2b_p50_G6g_frachard_gen/checkpoint-100"),
    ("G6g frac_hard + gen @150", GRPO / "2b_p50_G6g_frachard_gen/checkpoint-150"),
    ("G6g frac_hard + gen final", GRPO / "2b_p50_G6g_frachard_gen/final"),
    ("G6h frac_hard hardened @50", GRPO / "2b_p50_G6h_frachard_hardened/checkpoint-50"),
    ("G6h frac_hard hardened @100", GRPO / "2b_p50_G6h_frachard_hardened/checkpoint-100"),
    ("G6h frac_hard hardened @150", GRPO / "2b_p50_G6h_frachard_hardened/checkpoint-150"),
    ("G6i frac_hard restate @50", GRPO / "2b_p50_G6i_frachard_restate/checkpoint-50"),
    ("G6i frac_hard restate @100", GRPO / "2b_p50_G6i_frachard_restate/checkpoint-100"),
    ("G6i frac_hard restate @150", GRPO / "2b_p50_G6i_frachard_restate/checkpoint-150"),
    ("G7 cvf @50", GRPO / "2b_p50_G7_cvf/checkpoint-50"),
    ("G7 cvf @100", GRPO / "2b_p50_G7_cvf/checkpoint-100"),
    ("G7 cvf @150", GRPO / "2b_p50_G7_cvf/checkpoint-150"),
    ("G7g cvf gen-only @50", GRPO / "2b_p50_G7g_cvf_genonly/checkpoint-50"),
    ("G7g cvf gen-only @100", GRPO / "2b_p50_G7g_cvf_genonly/checkpoint-100"),
    ("G7g cvf gen-only @150", GRPO / "2b_p50_G7g_cvf_genonly/checkpoint-150"),
    ("G6h frac_hard hardened final", GRPO / "2b_p50_G6h_frachard_hardened/final"),
    ("G6i frac_hard restate final", GRPO / "2b_p50_G6i_frachard_restate/final"),
    ("G7 cvf final", GRPO / "2b_p50_G7_cvf/final"),
    ("G7g cvf gen-only final", GRPO / "2b_p50_G7g_cvf_genonly/final"),
    ("SFT p50 + lemma catalog (bf16 DDP)", SFT / "qwen35_2b_p50_cont_lc_lr5em6_seed3407/final"),
    ("SFT p50 + lemma catalog (fp32 master)", SFT / "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407/final"),
    ("SFT p50 + control", SFT / "qwen35_2b_p50_cont_ct_lr5em6_seed3407/final"),
    ("SFT p50 x3 @781", SFT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407/checkpoint-781"),
    ("SFT p50 x3 @1562", SFT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407/checkpoint-1562"),
    ("SFT p50 x3 final", SFT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407/final"),
]
KEYS = ["has_proof", "grammatical", "format_ok", "valid_eval", "circular", "n_taut", "valid", "in_system", "correct"]


def main():
    test = {json.loads(ln)["id"]: json.loads(ln) for ln in open(TEST)}
    res = {}
    for name, d in MODELS:
        f = d / "rl_gate_dolci/generations.jsonl"
        if not f.exists():
            continue
        tot, per = {k: 0.0 for k in KEYS}, {}
        for ln in open(f):
            g = json.loads(ln)
            c = components(test[g["id"]], g["generation"])
            c["valid_eval"] = float(bool(g["valid"]))
            b = per.setdefault(g["bench"], {k: 0.0 for k in KEYS + ["n"]})
            for k in KEYS:
                tot[k] += c[k]
                b[k] += c[k]
            b["n"] += 1
        n = sum(b["n"] for b in per.values())
        res[name] = {"all": {k: v / n for k, v in tot.items()},
                     "per_bench": {bn: {k: v / b["n"] for k, v in b.items() if k != "n"} for bn, b in per.items()}}
        print(name, {k: round(v, 3) for k, v in res[name]["all"].items()})
    (REPO / "analysis/stage2_gate_ckpts.json").write_text(json.dumps(res, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("grammatical", "grammatical"), ("format_ok", "format ok"), ("valid_eval", "valid (eval: strict ok + grounded)"),
            ("valid", "valid (hardened)"), ("in_system", "in-system (hardened)"), ("correct", "correct")]
    fig, ax = plt.subplots(figsize=(14, 4.6))
    w = 0.8 / len(res)
    for i, (name, r) in enumerate(res.items()):
        xs = [j + (i - (len(res) - 1) / 2) * w for j in range(len(show))]
        ys = [r["all"][k] for k, _ in show]
        bars = ax.bar(xs, ys, w, label=name)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.2f}" if y >= 0.01 else f"{y:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(show)), [lbl for _, lbl in show], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of 950 held-out items (greedy)")
    ax.set_title("Stage-2 gate on GRPO checkpoints (2B): the valid gain of G5 is mostly circular `given` proofs")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    fig.tight_layout()
    out = REPO / "reports/figures/stage2_gate_ckpts.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
