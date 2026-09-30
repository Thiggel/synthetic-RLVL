#!/usr/bin/env python3
"""Continued-SFT arms c / l / e / le vs their base (2026-10-01; scripts/data/build_libext_ei_mixture.py).

Question: does the Dolci gate (real prompts) gain validity from new lemma families (l), from the policy's own
checker-passing proofs on real prompts (e, expert iteration), or from both (le), beyond what more SFT on fresh
generator rows gives (c, control)? All arms: same base (qwen35_2b_p50_cont_lc_fp32m), same 6k Dolci rows, 20k rows.
Inputs, per run (scripts/slurm/jobs/sft_eval_suite.slurm, new lemma library):
  final/rl_gate_dolci/summary.json       greedy, 950 gate prompts
  final/rl_gate_dolci_k16/summary.json   16 samples at T=1.0 (per-sample rates, pass@k)
  formal_eval/summary.json               generator test, default families (2000)
  formal_eval_math/summary.json          generator test, new families (1000)
Missing runs/evals are skipped. Writes analysis/libext_ei_arms.{md,json}, reports/figures/libext_ei_arms.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SFT = Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925")
RUNS = {"base": SFT / "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407",
        **{a: SFT / f"qwen35_2b_lc_libext_{a}_lr5em6_seed3407" for a in ("c", "l", "e", "le")}}
LABEL = {"base": "base (L1 init)", "c": "c: +fresh gen", "l": "l: +new families", "e": "e: +EI proofs",
         "le": "le: +both"}
BENCHES = ["dolci_wordprob", "dolci_math", "dolci_yesno", "dolci_dapo", "dolci_knowledge"]
KS = ["@1", "@2", "@4", "@8", "@16"]
REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "analysis/libext_ei_arms.md"
OUT_JSON = REPO / "analysis/libext_ei_arms.json"
OUT_FIG = REPO / "reports/figures/libext_ei_arms"


def load(run: Path) -> dict | None:
    f = {k: run / p / "summary.json" for k, p in (("greedy", "final/rl_gate_dolci"), ("k16", "final/rl_gate_dolci_k16"),
                                                    ("gen", "formal_eval"), ("math", "formal_eval_math"))}
    if not all(x.is_file() for x in f.values()):
        return None
    s = {k: json.loads(x.read_text()) for k, x in f.items()}
    g, k16 = s["greedy"]["overall"]["all"], s["k16"]
    return {"greedy": {m: g[m] for m in ("has_proof", "grammatical", "valid", "correct", "valid_correct")},
            "sample": {m: k16["overall"]["all"][m] for m in ("has_proof", "grammatical", "valid", "correct",
                                                              "valid_correct")},
            "pass_at_k": {m: k16["pass_at_k"]["overall"][m] for m in ("valid", "valid_correct", "correct")},
            "sample_per_bench": {b: {m: k16["per_bench"][b]["all"][m] for m in ("valid", "valid_correct", "correct")}
                                 for b in BENCHES},
            "gen_test": {m: s["gen"]["overall"][m] for m in ("valid", "faithful", "answer_acc")},
            "math_test": {m: s["math"]["overall"][m] for m in ("valid", "faithful", "answer_acc")},
            "math_per_family": {f_: v["valid"] for f_, v in s["math"]["per_family"].items()}}


def main() -> None:
    res = {a: r for a, run in RUNS.items() if (r := load(run)) is not None}
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")
    t = [("gate greedy valid", lambda r: r["greedy"]["valid"]),
         ("gate greedy valid·correct", lambda r: r["greedy"]["valid_correct"]),
         ("gate greedy correct", lambda r: r["greedy"]["correct"]),
         ("gate T=1 valid / sample", lambda r: r["sample"]["valid"]),
         ("gate T=1 valid·correct / sample", lambda r: r["sample"]["valid_correct"]),
         ("gate T=1 correct / sample", lambda r: r["sample"]["correct"]),
         ("gate valid@16", lambda r: r["pass_at_k"]["valid"]["@16"]),
         ("gate valid·correct@16", lambda r: r["pass_at_k"]["valid_correct"]["@16"]),
         ("gate mixed@8 (valid)", lambda r: r["pass_at_k"]["valid"]["mixed@8"]),
         ("gen test valid", lambda r: r["gen_test"]["valid"]),
         ("gen test answer acc", lambda r: r["gen_test"]["answer_acc"]),
         ("new-family test valid", lambda r: r["math_test"]["valid"])]
    t += [(f"T=1 valid / sample, {b[6:]}", lambda r, b=b: r["sample_per_bench"][b]["valid"]) for b in BENCHES]
    lines = ["# Continued-SFT arms c / l / e / le vs base (new lemma library)", "",
             "| metric | " + " | ".join(LABEL[a] for a in res) + " |", "|---|" + "---:|" * len(res)]
    lines += [f"| {name} | " + " | ".join(f"{f(r):.4f}" for r in res.values()) + " |" for name, f in t]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, m in zip(axes[:2], ("valid", "valid_correct")):
        for a, r in res.items():
            ax.plot([int(k[1:]) for k in KS], [r["pass_at_k"][m][k] for k in KS], marker="o", label=LABEL[a])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (samples at T=1.0)")
        ax.set_title(f"Dolci gate {m}@k")
        ax.legend(fontsize=7)
    ax = axes[2]
    w = .8 / len(res)
    for i, (a, r) in enumerate(res.items()):
        ax.bar([j + i * w for j in range(len(BENCHES))], [r["sample_per_bench"][b]["valid"] for b in BENCHES], w,
               label=LABEL[a])
    ax.set_xticks([j + .4 - w / 2 for j in range(len(BENCHES))], [b[6:] for b in BENCHES])
    ax.set_title("Dolci gate: valid per sample (T=1), by source")
    ax.legend(fontsize=7)
    ax = axes[3]
    names = ["gate greedy correct", "gate T=1 correct / sample", "gen test valid", "new-family test valid"]
    fs = dict(t)
    for i, (a, r) in enumerate(res.items()):
        ax.bar([j + i * w for j in range(len(names))], [fs[n](r) for n in names], w, label=LABEL[a])
    ax.set_xticks([j + .4 - w / 2 for j in range(len(names))], ["gate correct\n(greedy)", "gate correct\n(T=1)",
                                                                  "gen test\nvalid", "new-family\ntest valid"])
    ax.set_ylim(0, 1)
    ax.set_title("side effects: correctness and in-domain validity")
    ax.legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle("Continued SFT from the L1 base: new lemma families (l), self-distilled real-prompt proofs (e), "
                 "both (le), control (c)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=140)


if __name__ == "__main__":
    main()
