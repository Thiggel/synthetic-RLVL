#!/usr/bin/env python3
"""Prompt-filtered GRPO (G11) vs unfiltered (G10), both from G8 final with the cvf reward (2026-10-01).

G11 trains only on prompts whose sampled cvf rate under G8 final is in (0, 1) (scripts/rl_prompt_filter.py,
n=16): 1438 prompts, of which 1257 generator, 145 Dolci wordprob, 30 yesno, 6 math. G10 trains on all 4091
(gen 1200, dolci_math 1200, wordprob 1200, yesno 491). G11 died of CUDA OOM at step 580/600 (no final), so the
comparison stops at step 500. Gate = 950 held-out Dolci items, greedy, frozen pre-libext checker; "clean" = the
713 items without a near-duplicate in the training-prompt pool (analysis/gate_contamination.md).
Per checkpoint: valid, valid_correct, correct (Answer line), ans_correct (the proof's `ans`), has_proof (all / clean / per bench), and what follows </proof>
(tail classes of g10_format_drift.py). From the rollout logs: cvf and format_ok per 10-step bin.
Writes analysis/g11_vs_g10.{md,json} and reports/figures/g11_vs_g10.{png,pdf}.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g10_format_drift import COLORS as TAIL_COLORS, TAILS, tail  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
RUNS = DATA / "grpo_formal_20260928"
G8 = RUNS / "2b_lcfp32m_G8_cvf/final"
ARMS = {"G10 (all prompts)": RUNS / "2b_lcfp32m_G10_cvf_cont600",
        "G11 (prompt-filtered)": RUNS / "2b_lcfp32m_G11_cvf_filtered"}
ARM_COLORS = {"G10 (all prompts)": "tab:blue", "G11 (prompt-filtered)": "tab:orange"}
STEPS = (200, 400, 500)
CONTAM = DATA / "datasets/rl_gate_dolci_instruct_20260928/contamination.json"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis/g11_vs_g10"
OUT_FIG = REPO / "reports/figures/g11_vs_g10"
METRICS = ("valid", "valid_correct", "correct", "ans_correct", "has_proof")
BIN = 10


def gate(ckpt: Path, clean: set[str]) -> dict | None:
    f = ckpt / "rl_gate_dolci/generations.jsonl"
    if not f.is_file():
        return None
    d = pd.DataFrame([json.loads(x) for x in open(f)])
    d["tail"] = d.generation.map(tail)
    d["clean"] = d.id.isin(clean)
    r = {"all": {m: float(d[m].mean()) for m in METRICS}, "clean": {m: float(d[d.clean][m].mean()) for m in METRICS},
         "bench": {b: {m: float(g[m].mean()) for m in METRICS} for b, g in d.groupby("bench")},
         "tail": {t: float((d["tail"] == t).mean()) for t in TAILS}}
    return r


def per_step(run: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(run / "completions/completions_*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=["step", "cvf", "format_ok", "valid"]) for f in fs], ignore_index=True)
    d["b"] = d.step // BIN * BIN
    g = d.groupby("b")
    return g[["cvf", "format_ok", "valid"]].mean()[g.size() >= g.size().max() / 2]


def main() -> None:
    clean = set(json.loads(CONTAM.read_text())["clean_ids"])
    base = gate(G8, clean)
    res: dict = {"gates": {}, "train": {}}
    for arm, run in ARMS.items():
        pts = {0: base}
        for s in STEPS:
            g = gate(run / f"checkpoint-{s}", clean)
            if g:
                pts[s] = g
        if (run / "final").is_dir() and (g := gate(run / "final", clean)):
            st = run / "final/trainer_state.json"
            pts[json.loads(st.read_text())["global_step"] if st.is_file() else 600] = g  # G10: --max-steps 600
        res["gates"][arm] = pts
        res["train"][arm] = per_step(run).reset_index().to_dict(orient="records")
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")

    benches = sorted(base["bench"])
    lines = ["# G11 (prompt-filtered GRPO) vs G10 (unfiltered), both from G8 final, cvf reward", "",
             "Dolci gate, 950 items, greedy, frozen pre-libext checker. clean = 713 items without a near-duplicate "
             "in the training pool. Step 0 = G8 final. (scripts/analysis/g11_vs_g10.py)", "",
             "| arm | step | valid | valid clean | v·c | v·c clean | correct | correct clean | proof ans correct | has_proof | "
             + " | ".join(f"valid {b.replace('dolci_', '')}" for b in benches) + " | `<formal>` loop | Answer: x |",
             "|---|---:|" + "---:|" * (10 + len(benches))]
    for arm, pts in res["gates"].items():
        for s, r in sorted(pts.items()):
            a, c = r["all"], r["clean"]
            lines.append(f"| {arm} | {s} | {a['valid']:.3f} | {c['valid']:.3f} | {a['valid_correct']:.3f} | "
                         f"{c['valid_correct']:.3f} | {a['correct']:.3f} | {c['correct']:.3f} | {a['ans_correct']:.3f} | "
                         f"{a['has_proof']:.3f} | "
                         + " | ".join(f"{r['bench'][b]['valid']:.3f}" for b in benches)
                         + f" | {r['tail'][TAILS[3]]:.3f} | {r['tail'][TAILS[0]]:.3f} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    for ax, m in zip(axes[0], ("valid", "valid_correct", "correct")):
        for arm, pts in res["gates"].items():
            s = sorted(pts)
            ax.plot(s, [pts[i]["all"][m] for i in s], "o-", color=ARM_COLORS[arm], label=f"{arm}, all 950")
            ax.plot(s, [pts[i]["clean"][m] for i in s], "s--", color=ARM_COLORS[arm], alpha=.6,
                    label=f"{arm}, clean 713")
        ax.set_title(f"Dolci gate (greedy): {m}")
        ax.set_xlabel("GRPO step (0 = G8 final)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    ax = axes[1, 0]
    cols = [("G8 final", base)] + [(f"{a.split()[0]} @{max(STEPS)}", res["gates"][a].get(max(STEPS))) for a in ARMS]
    cols = [(n, r) for n, r in cols if r]
    w = .8 / len(cols)
    for i, (n, r) in enumerate(cols):
        ax.bar([j + (i - (len(cols) - 1) / 2) * w for j in range(len(benches))],
               [r["bench"][b]["valid"] for b in benches], w, label=n,
               color=["tab:gray", "tab:blue", "tab:orange"][i])
    ax.set_xticks(range(len(benches)), [b.replace("dolci_", "") for b in benches])
    ax.set_title("gate valid per bench")
    ax.grid(alpha=.3, axis="y")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    bars = [("G8", base)] + [(f"{a.split()[0]}@{s}", res["gates"][a][s]) for a in ARMS for s in STEPS
                             if s in res["gates"][a]]
    bottom = [0.0] * len(bars)
    for t, c in zip(TAILS, TAIL_COLORS):
        v = [r["tail"][t] for _, r in bars]
        ax.bar(range(len(bars)), v, bottom=bottom, color=c, label=t)
        bottom = [b + x for b, x in zip(bottom, v)]
    ax.set_xticks(range(len(bars)), [n for n, _ in bars], rotation=30, ha="right", fontsize=8)
    ax.set_title("gate (greedy): what follows </proof>")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1, 2]
    for arm, rows in res["train"].items():
        t = pd.DataFrame(rows)
        ax.plot(t.b + BIN / 2, t.cvf, color=ARM_COLORS[arm], label=f"{arm}: cvf")
        ax.plot(t.b + BIN / 2, t.format_ok, ":", color=ARM_COLORS[arm], label=f"{arm}: format_ok")
    ax.set_title("training rollouts (own prompt pools, not comparable in level)")
    ax.set_xlabel("GRPO step")
    ax.set_ylim(-.02, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=7)
    fig.suptitle("G11: GRPO only on prompts G8 solves sometimes (87% generator) vs G10: all prompts (71% Dolci)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=130)


if __name__ == "__main__":
    main()
