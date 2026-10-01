#!/usr/bin/env python3
"""Every Dolci-gate eval rescored on the clean gate subset (2026-10-01).

analysis/gate_contamination.py found that 237 of the 950 gate prompts have a near-duplicate (12-gram coverage
>= .3) among the training-prompt pool (Dolci train + GSM8K train) that RL and EI draw from. Here every
rl_gate_dolci{,_k16}/generations.jsonl under rlvl_data is rescored on the clean items (713) and on the
contaminated ones: per-sample has_proof, valid, valid_correct, correct. If training on near-duplicates
inflated the gate, the gain from an intervention is larger on the contaminated items than on the clean ones.
Writes analysis/gate_clean_rescore.{md,json} and reports/figures/gate_clean_rescore.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
CONTAM = DATA / "datasets/rl_gate_dolci_instruct_20260928/contamination.json"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis/gate_clean_rescore"
OUT_FIG = REPO / "reports/figures/gate_clean_rescore"
METRICS = ("has_proof", "valid", "valid_correct", "correct")
SFT = "formal_mixture_sft_20260925/"
G = "grpo_formal_20260928/"
# the figure: interventions from the L1 base (qwen35_2b_p50_cont_lc_fp32m), greedy gate
FIG_RUNS = {"L1 base": SFT + "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407/final",
            "G8 (cvf)": G + "2b_lcfp32m_G8_cvf/final",
            "G10 @500 (cvf)": G + "2b_lcfp32m_G10_cvf_cont600/checkpoint-500",
            "L1 correct @250": G + "L1_correct/checkpoint-250", "L1 correct @500": G + "L1_correct/checkpoint-500",
            "SFT c": SFT + "qwen35_2b_lc_libext_c_lr5em6_seed3407/final",
            "SFT l": SFT + "qwen35_2b_lc_libext_l_lr5em6_seed3407/final",
            "SFT e": SFT + "qwen35_2b_lc_libext_e_lr5em6_seed3407/final",
            "SFT le": SFT + "qwen35_2b_lc_libext_le_lr5em6_seed3407/final"}


def rescore(f: Path, clean: set[str]) -> dict:
    acc = {s: {m: 0 for m in METRICS} | {"n": 0} for s in ("all", "clean", "contaminated")}
    for line in open(f):
        r = json.loads(line)
        for s in ("all", "clean" if r["id"] in clean else "contaminated"):
            acc[s]["n"] += 1
            for m in METRICS:
                acc[s][m] += bool(r[m])
    return {s: {m: (v / a["n"] if m != "n" else v) for m, v in a.items()} for s, a in acc.items() if a["n"]}


def main() -> None:
    c = json.loads(CONTAM.read_text())
    clean = set(c["clean_ids"])
    res = {}
    for f in sorted(DATA.glob("**/rl_gate_dolci*/generations.jsonl")):
        res[str(f.parent.relative_to(DATA))] = rescore(f, clean)
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# Dolci gate rescored on the clean subset", "",
             f"clean = {len(clean)} of {c['n_gate']} gate items without a near-duplicate (12-gram coverage >= "
             f"{c['threshold']}) in the training-prompt pool (analysis/gate_contamination.md). Per-sample rates; "
             "`_k16` = 16 samples at T=1, else greedy.", "",
             "| eval | valid all | valid clean | valid contam. | v·c all | v·c clean | correct all | correct clean | "
             "correct contam. |", "|---|" + "---:|" * 8]
    for k, r in res.items():
        a, cl, co = r["all"], r["clean"], r.get("contaminated", {})
        lines.append(f"| {k.replace('/rl_gate_dolci', ' ').replace(SFT, '').replace(G, '')} | {a['valid']:.4f} | "
                     f"{cl['valid']:.4f} | {co.get('valid', 0):.4f} | {a['valid_correct']:.4f} | "
                     f"{cl['valid_correct']:.4f} | {a['correct']:.3f} | {cl['correct']:.3f} | "
                     f"{co.get('correct', 0):.3f} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fr = {n: res[p + "/rl_gate_dolci"] for n, p in FIG_RUNS.items() if p + "/rl_gate_dolci" in res}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for ax, m in zip(axes, ("valid", "valid_correct", "correct")):
        x = range(len(fr))
        for i, (s, col) in enumerate((("clean", "tab:blue"), ("contaminated", "tab:red"), ("all", "tab:gray"))):
            ax.bar([j + (i - 1) * .27 for j in x], [r[s][m] for r in fr.values()], .27, color=col,
                   label=f"{s} (n={next(iter(fr.values()))[s]['n']})")
        ax.set_xticks(list(x), list(fr), rotation=35, ha="right", fontsize=8)
        ax.set_title(f"Dolci gate, greedy: {m}")
        ax.grid(alpha=.3, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("Gate items with a near-duplicate in the training-prompt pool (contaminated) vs the rest (clean)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=140)


if __name__ == "__main__":
    main()
