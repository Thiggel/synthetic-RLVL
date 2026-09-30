#!/usr/bin/env python3
"""Why sampled proofs on the Dolci RL gate fail: the first failing checker stage per sample, and a
heuristic split of the parse errors (2026-09-30).

Input: <model>/rl_gate_dolci_k16/generations.jsonl (950 prompts x 16 samples at T=1.0, scored by
scripts/eval_formal_bench_vllm.py with the frozen pre-libext checker). Categories, in checker order:
no proof, parse, rule (unknown or misapplied lemma), quote (quote not in the prompt), literal (a number in
a `given` missing from its quote), other checker errors (type/scope/label/answer/leak), checked but not
valid, valid. Parse errors are split heuristically by the text at the error position (the checker message shows
it around a <here> marker), first match wins: English prose inside a formula ("n is the smallest"), a
malformed justification (no `; rule`, or an error after the `;`: bad rule name or arguments), nested step numbers or case splits ("2.2.1", "case n = 2:"),
a `?` placeholder, notation the language lacks (sets, sums, lists, |x|, subscripts, %, ==, +=, chained
comparisons, tuples, **, '), juxtaposition (multi-word names like `total outcomes`, implicit products
like `2(x+1)`), other.
Writes analysis/gate_error_breakdown.json and reports/figures/gate_error_breakdown.{png,pdf}.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
SFT = DATA / "formal_mixture_sft_20260925"
X3 = SFT / "qwen35_2b_dolci_rlvlgen_p50_x3_lr5em6_seed3407"
MODELS = [("SFT p25 fp32m", SFT / "qwen35_2b_dolci_rlvlgen_p25_lr5em6_seed3407_fp32m/final"),
          ("SFT x3 @781", X3 / "checkpoint-781"),
          ("SFT x3 @1562", X3 / "checkpoint-1562"),
          ("SFT p50 + lemma catalog", SFT / "qwen35_2b_p50_cont_lc_fp32m_lr5em6_seed3407/final"),
          ("GRPO G8 final", DATA / "grpo_formal_20260928/2b_lcfp32m_G8_cvf/final")]
STAGES = ["no proof", "parse", "rule", "quote", "literal", "other error", "checked, not valid", "valid"]
PARSE = ["prose in formula", "malformed `; rule`", "nested / case-split steps", "`?` placeholder",
         "unsupported notation", "juxtaposition", "other parse"]
BENCH_SETS = {"in-domain Dolci (math, wordprob, yesno)": ("dolci_math", "dolci_wordprob", "dolci_yesno"),
              "all 950 gate prompts": None}
REPO = Path(__file__).resolve().parents[2]
OUT_JSON = REPO / "analysis/gate_error_breakdown.json"
OUT_FIG = REPO / "reports/figures/gate_error_breakdown"
_CTX = re.compile(r'"(.*)⟨here⟩(.*)"', re.S)
_NOTATION = re.compile(r"[{}\[\]|_^%…#']|\*\*|\.\.|\\|--|\bsum\b|\bmin\b|\bmax\b")


def stage(r: dict) -> str:
    if r["valid"]:
        return "valid"
    if not r["has_proof"]:
        return "no proof"
    e = r.get("error")
    code = e.get("code") if isinstance(e, dict) else None
    if code in ("parse", "rule", "quote", "literal"):
        return code
    return "other error" if code else "checked, not valid"


def parse_kind(msg: str) -> str:
    m = _CTX.search(msg)
    if not m:
        return "other parse"
    before, after = m.group(1).replace("\\n", "\n"), m.group(2).replace("\\n", "\n")
    line = before.rsplit("\n", 1)[-1]  # the failing line up to the error
    win = re.search(r"[A-Za-z_]*$", before).group() + after[:16]  # the word under the error, and what follows
    if re.search(r"\b(is|are|the|of|where|such|that|with|must|can|has|to|for|if|then)\b [a-z]", win):
        return "prose in formula"
    if re.match(r"\s*(given|def|arith|logic)\b", after) or ";" in line or (after[:1] == "\n" and "goal" not in line):
        return "malformed `; rule`"
    if re.match(r"\d+(\.\d+)+ ", line) or re.match(r"\s*(\d+\.\d+|case\b)", after) or ":\n" in after[:12]:
        return "nested / case-split steps"
    if "?" in line.split("=")[-1][-3:] or re.match(r"\s*\?", after):
        return "`?` placeholder"
    if _NOTATION.search(before[-4:] + after[:4]) or re.search(r"(==|\+=|<=?\s*\w+\s*<|\(\s*-?\w+\s*,)", before[-12:] + after[:6]):
        return "unsupported notation"
    if re.search(r"[\w)] ?$", before) and re.match(r" ?[\w(]", after):
        return "juxtaposition"
    return "other parse"


def main() -> None:
    old = json.loads(OUT_JSON.read_text()) if OUT_JSON.is_file() else {}
    res = {}
    for name, d in MODELS:
        f = d / "rl_gate_dolci_k16/generations.jsonl"
        if not f.is_file():  # model dir gone (deleted checkpoint): keep the recorded breakdown
            if name in old:
                res[name] = old[name]
            continue
        rows = [json.loads(ln) for ln in open(f)]
        res[name] = {}
        for bs, benches in BENCH_SETS.items():
            sel = [r for r in rows if benches is None or r["bench"] in benches]
            st = collections.Counter(stage(r) for r in sel)
            pk = collections.Counter(parse_kind(r["error"]["msg"]) for r in sel if stage(r) == "parse")
            res[name][bs] = {"n": len(sel), "stage": {s: st[s] / len(sel) for s in STAGES},
                             "parse": {p: pk[p] / len(sel) for p in PARSE}}
    OUT_JSON.write_text(json.dumps(res, indent=1) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    bs = next(iter(BENCH_SETS))
    names = list(res)
    for ax, key, cats, cmap in [(axes[0], "stage", STAGES, "tab10"), (axes[1], "parse", PARSE, "Set2")]:
        left = [0.0] * len(names)
        colors = plt.get_cmap(cmap).colors
        for i, c in enumerate(cats):
            v = [res[n][bs][key][c] for n in names]
            ax.barh(names, v, left=left, color=colors[i % len(colors)], label=c)
            for y, (lo, x) in enumerate(zip(left, v)):
                if x >= .04:
                    ax.text(lo + x / 2, y, f"{x:.2f}", ha="center", va="center", fontsize=7)
            left = [a + b for a, b in zip(left, v)]
        ax.set_xlabel("fraction of samples")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=4)
    axes[0].set_title("first failing checker stage")
    axes[1].set_title("parse errors by kind (share of all samples)")
    axes[0].invert_yaxis()
    fig.suptitle(f"Dolci RL gate, 16 samples at T=1.0: why proofs fail ({bs}, {res[names[0]][bs]['n']} samples)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}.{ext}", dpi=150)
    for n in names:
        for b in BENCH_SETS:
            print(n, "|", b, "|", {k: round(v, 3) for k, v in res[n][b]["stage"].items()},
                  {k: round(v, 3) for k, v in res[n][b]["parse"].items()})


if __name__ == "__main__":
    main()
