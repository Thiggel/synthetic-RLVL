#!/usr/bin/env python3
"""Continued-SFT arms c / l / e / le / e2 vs their base (2026-10-01; scripts/data/build_libext_ei_mixture.py).

Question: does the Dolci gate (real prompts) gain validity from new lemma families (l), from the policy's own
checker-passing proofs on real prompts (e, expert iteration), or from both (le), beyond what more SFT on fresh
generator rows gives (c, control)? All arms: same base (qwen35_2b_p50_cont_lc_fp32m), same 6k Dolci rows, 20k rows.
Inputs, per run (scripts/slurm/jobs/sft_eval_suite.slurm, new lemma library):
  final/rl_gate_dolci/summary.json       greedy, 950 gate prompts
  final/rl_gate_dolci_k16/summary.json   16 samples at T=1.0 (per-sample rates, pass@k)
  formal_eval/summary.json               generator test, default families (2000)
  formal_eval_math/summary.json          generator test, new families (1000)
Missing runs/evals are skipped. Once c/l/e/le are all in, the 2x2 contrasts (EI and new families, each with and
without the other, and their interaction) get paired-bootstrap 95% CIs over gate prompts, on all 950 and on the 713
clean items (analysis/gate_contamination.md).
EI round 2 (2026-10-02): e2 = e with the harvest from an RL'd teacher (G12 checkpoint-100); contrast e2 - e.
e2s = e2's harvest cut to e's size per bench: e2s - e is teacher quality at fixed quantity, e2 - e2s quantity.
EI round 3 (2026-10-03): e3 = e with the harvest from G16 checkpoint-250 (no-proof penalty), one proof per prompt
(4874 rows / prompts) vs e2's up to several per prompt (6776 rows / 2512 prompts): e3 - e2 mixes teacher and data shape.
EI round 4 (2026-10-03): e4 = e with the harvest from G16 checkpoint-750, re-filtered with the hardened reward (premise
numbers stated), up to 4 proofs per prompt (5221 rows / 1679 prompts): e4 - e3 mixes teacher, filter and data shape.
Writes analysis/libext_ei_arms.{md,json}, reports/figures/libext_ei_arms.{png,pdf}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SFT = Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925")
RUNS = {"base": SFT / "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407",
        **{a: SFT / f"qwen35_2b_lc_libext_{a}_lr5em6_seed3407" for a in ("c", "l", "e", "le", "e2", "e2s", "e3", "e4")}}
LABEL = {"base": "base (L1 init)", "c": "c: +fresh gen", "l": "l: +new families", "e": "e: +EI proofs",
         "le": "le: +both", "e2": "e2: +EI from RL teacher", "e2s": "e2s: e2 harvest cut to e size",
         "e3": "e3: +EI from G16@250, 1/prompt", "e4": "e4: +EI from G16@750, hardened, <=4/prompt"}
BENCHES = ["dolci_wordprob", "dolci_math", "dolci_yesno", "dolci_dapo", "dolci_knowledge"]
KS = ["@1", "@2", "@4", "@8", "@16"]
REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "analysis/libext_ei_arms.md"
OUT_JSON = REPO / "analysis/libext_ei_arms.json"
OUT_FIG = REPO / "reports/figures/libext_ei_arms"
CONTAM = Path("/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/contamination.json")
CONTRASTS = {"EI, no new families (e - c)": lambda x: x["e"] - x["c"],
             "EI, with new families (le - l)": lambda x: x["le"] - x["l"],
             "new families, no EI (l - c)": lambda x: x["l"] - x["c"],
             "new families, with EI (le - e)": lambda x: x["le"] - x["e"],
             "interaction (le - e) - (l - c)": lambda x: x["le"] - x["e"] - x["l"] + x["c"],
             "RL teacher for EI (e2 - e)": lambda x: x["e2"] - x["e"],
             "teacher quality at e's size (e2s - e)": lambda x: x["e2s"] - x["e"],
             "harvest quantity (e2 - e2s)": lambda x: x["e2"] - x["e2s"],
             "EI round 3 (e3 - e2)": lambda x: x["e3"] - x["e2"],
             "EI round 4 (e4 - e3)": lambda x: x["e4"] - x["e3"]}


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


def per_prompt(run: Path, evl: str, m: str) -> dict[str, float]:
    acc: dict[str, list[bool]] = {}
    for line in open(run / "final" / evl / "generations.jsonl"):
        r = json.loads(line)
        acc.setdefault(r["id"], []).append(bool(r[m]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def contrasts(n_boot: int = 2000) -> dict:
    """2x2 contrasts of per-prompt rates, paired bootstrap over gate prompts: {eval/metric/subset: {name: [est, lo, hi]}}."""
    clean = set(json.loads(CONTAM.read_text())["clean_ids"])
    rng = np.random.default_rng(0)
    out = {}
    for evl in ("rl_gate_dolci", "rl_gate_dolci_k16"):
        for m in ("valid", "valid_correct"):
            pp = {a: per_prompt(RUNS[a], evl, m) for a in ("c", "l", "e", "le", "e2", "e2s", "e3", "e4")}
            for sub in ("all", "clean"):
                ids = sorted(i for i in pp["c"] if sub == "all" or i in clean)
                x = {a: np.array([v[i] for i in ids]) for a, v in pp.items()}
                boot = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]
                out[f"{evl}/{m}/{sub}"] = {
                    k: [float(f({a: v.mean() for a, v in x.items()}))]
                    + [float(q) for q in np.percentile([f({a: v[b].mean() for a, v in x.items()}) for b in boot],
                                                       (2.5, 97.5))]
                    for k, f in CONTRASTS.items()}
    return out


def main() -> None:
    res = {a: r for a, run in RUNS.items() if (r := load(run)) is not None}
    con = contrasts() if all(a in res for a in ("c", "l", "e", "le", "e2", "e2s", "e3", "e4")) else {}
    OUT_JSON.write_text(json.dumps(res | {"contrasts": con}, indent=1) + "\n")
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
    lines = ["# Continued-SFT arms c / l / e / le / e2 vs base (new lemma library)", "",
             "| metric | " + " | ".join(LABEL[a] for a in res) + " |", "|---|" + "---:|" * len(res)]
    lines += [f"| {name} | " + " | ".join(f"{f(r):.4f}" for r in res.values()) + " |" for name, f in t]
    if con:
        cols = list(con)
        lines += ["", "## 2x2 contrasts (paired bootstrap over gate prompts, 95% CI)", "",
                  "`rl_gate_dolci` = greedy, `_k16` = per sample at T=1; clean = 713 items without a near-duplicate "
                  "in the training-prompt pool.", "",
                  "| contrast | " + " | ".join(c.replace("rl_gate_dolci", "gate") for c in cols) + " |",
                  "|---|" + "---:|" * len(cols)]
        lines += [f"| {k} | " + " | ".join("{:+.4f} [{:+.4f}, {:+.4f}]".format(*con[c][k]) for c in cols) + " |"
                  for k in CONTRASTS]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 5 if con else 4, figsize=(25 if con else 20, 4.4))
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
    if con:
        ax = axes[4]
        for i, (c, col) in enumerate((("rl_gate_dolci_k16/valid/clean", "tab:blue"),
                                      ("rl_gate_dolci_k16/valid_correct/clean", "tab:green"))):
            v = con[c]
            y = [j + (i - .5) * .3 for j in range(len(CONTRASTS))]
            ax.errorbar([v[k][0] for k in CONTRASTS], y, xerr=[[v[k][0] - v[k][1] for k in CONTRASTS],
                                                               [v[k][2] - v[k][0] for k in CONTRASTS]],
                        fmt="o", color=col, capsize=3, label=c.split("/")[1] + " / sample (T=1), clean 713")
        ax.axvline(0, color="k", lw=.8)
        ax.set_yticks(range(len(CONTRASTS)), list(CONTRASTS), fontsize=8)
        ax.invert_yaxis()
        ax.set_title("2x2 contrasts, 95% paired-bootstrap CI")
        ax.legend(fontsize=7, loc="lower left")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle("Continued SFT from the L1 base: new lemma families (l), self-distilled real-prompt proofs (e), "
                 "both (le), control (c), EI from an RL'd teacher (e2)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=140)


if __name__ == "__main__":
    main()
