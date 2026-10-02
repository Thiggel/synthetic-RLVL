#!/usr/bin/env python3
"""What the L1 policies write, over GRPO steps (2026-09-30; G13 and G14 added 2026-10-01, G15 and G16 2026-10-02).

Reads the per-step rollout logs <run>/completions/completions_*.parquet of each L1 arm (256 rollouts per step:
32 prompts x 8, training prompts with training rewards) and bins them into 25-step buckets, split by prompt source
(generator vs Dolci). Style of a completion, first match wins: `<proof>` block (formal), `<analysis>` block,
other text containing \\boxed, other text. Per bucket: the style mix, and within `<proof>` completions the share
with an `ans` line, grammatical and valid; correct and mean length for all completions.
Writes analysis/l1_format_shift.json and reports/figures/l1_format_shift.{png,pdf}.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = Path("/vol/tmp2/laitenbf/rlvl_data/grpo_formal_20260928")
ARMS = {"correct only": RUNS / "L1_correct", "cvf (x format_ok from step 51)": RUNS / "L1_cvf",
        "G13: le SFT + cvf_fmt (new lib)": RUNS / "2b_le_G13_cvffmt",
        "G14: G13 + overlong penalty, unmasked": RUNS / "2b_le_G14_cvffmt_overlong",
        "G15: G14 recipe from e2 SFT (EI round 2)": RUNS / "2b_e2_G15_cvffmt_overlong",
        "G16: G15 + no-proof penalty": RUNS / "2b_e2_G16_cvffmt_overlong_noproof"}
REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "analysis/l1_format_shift.json"
OUT_FIG = REPO / "reports/figures/l1_format_shift"
BIN = 25
STYLES = ["<proof>", "<analysis>", "other, \\boxed", "other"]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:gray"]


def style(c: str) -> str:
    if "<proof>" in c:
        return STYLES[0]
    if "<analysis>" in c:
        return STYLES[1]
    return STYLES[2] if "\\boxed" in c else STYLES[3]


def load(run: Path) -> pd.DataFrame | None:
    fs = sorted(glob.glob(str(run / "completions/completions_*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["step", "completion", "correct", "valid", "grammatical", "correct_gen"])
                   for f in fs], ignore_index=True)
    d["b"] = d.step // BIN * BIN
    d["src"] = d.correct_gen.notna().map({True: "gen", False: "dolci"})
    d["style"] = d.completion.map(style)
    body = d.completion.str.extract(r"<proof>(.*?)(?:</proof>|$)", flags=re.S)[0]
    d["ans"] = body.str.contains(r"(?m)^ans ", na=False)
    d["chars"] = d.completion.str.len()
    return d


def summarize(d: pd.DataFrame) -> dict:
    out = {}
    for src, g in d.groupby("src"):
        rows = []
        full = g.groupby("b").size().max()
        for b, x in g.groupby("b"):
            if len(x) < full / 2:  # the running bucket, too few steps yet
                continue
            p = x[x["style"] == STYLES[0]]
            rows.append({"step": int(b), "n": len(x), "style": {s: float((x["style"] == s).mean()) for s in STYLES},
                         "correct": float(x.correct.mean()), "valid": float(x.valid.mean()),
                         "chars": float(x.chars.mean()),
                         "proof_ans": float(p.ans.mean()) if len(p) else None,
                         "proof_grammatical": float(p.grammatical.mean()) if len(p) else None,
                         "proof_valid": float(p.valid.mean()) if len(p) else None})
        out[src] = rows
    return out


def main() -> None:
    res = {}
    for arm, run in ARMS.items():
        d = load(run)
        if d is not None:
            res[arm] = summarize(d)
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")

    fig, axes = plt.subplots(len(res), 4, figsize=(20, 4.4 * len(res)), squeeze=False)
    for row, (arm, r) in zip(axes, res.items()):
        for ax, src in zip(row[:2], ("gen", "dolci")):
            s = [x["step"] + BIN / 2 for x in r.get(src, [])]
            ax.stackplot(s, *[[x["style"][k] for x in r[src]] for k in STYLES], labels=STYLES, colors=COLORS,
                         alpha=.75)
            ax.set_ylim(0, 1)
            ax.set_title(f"{arm}: completion style, {'generator' if src == 'gen' else 'Dolci'} prompts")
            ax.legend(fontsize=7, loc="lower left")
        ax = row[2]
        for src, ls in (("gen", "-"), ("dolci", "--")):
            s = [x["step"] + BIN / 2 for x in r.get(src, [])]
            for k, c in (("correct", "tab:green"), ("valid", "tab:red")):
                ax.plot(s, [x[k] for x in r[src]], ls, color=c, label=f"{k} ({src})")
            ax.plot(s, [x["proof_ans"] for x in r[src]], ls, color="tab:purple", label=f"<proof> has `ans` line ({src})")
            ax.plot(s, [x["proof_grammatical"] for x in r[src]], ls, color="tab:brown",
                    label=f"<proof> grammatical ({src})")
        ax.set_ylim(-.02, 1.02)
        ax.set_title(f"{arm}: reward components and proof health")
        ax.legend(fontsize=6, ncol=2, loc="center right")
        ax = row[3]
        for src, ls in (("gen", "-"), ("dolci", "--")):
            s = [x["step"] + BIN / 2 for x in r.get(src, [])]
            ax.plot(s, [x["chars"] for x in r[src]], ls, color="k", label=src)
        ax.set_title(f"{arm}: mean completion length (chars)")
        ax.legend(fontsize=7)
        for ax in row:
            ax.set_xlabel("GRPO step")
            ax.grid(alpha=.3)
    fig.suptitle(f"L1: what the policy writes on its training prompts ({BIN}-step buckets, 256 rollouts per step)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=130)
    for arm, r in res.items():
        for src, rows in r.items():
            for x in rows:
                print(arm, src, x["step"], {k: round(v, 3) for k, v in x["style"].items()}, round(x["correct"], 3),
                      round(x["valid"], 3), None if x["proof_ans"] is None else round(x["proof_ans"], 3),
                      round(x["chars"]))


if __name__ == "__main__":
    main()
