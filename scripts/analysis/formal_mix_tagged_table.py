#!/usr/bin/env python
"""Tables for the format-tagged benchmark eval of the 2026-09-25 formal-mixture sweep.

Reads <sweep-root>/qwen35_<m>_dolci_rlvlgen_pXX_lr5em6_seed3407/<subdir>/summary.json
(scripts/eval_formal_bench_vllm.py) and writes to --out-dir:
  tagged_long.csv   model, x, bench, subset (all | answerable), metric, value
  tagged_<model>.md per model: for each bench, has_proof / grammatical / valid /
                    correct / in_system (%) per mixture fraction
  tagged_curves.{pdf,png}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RUN_RE = re.compile(r"qwen35_(?P<m>[0-9.]+b)_dolci_rlvlgen_p(?P<x>\d\d)_lr5em6_seed3407$")
MODELS = ["0.8b", "2b", "9b"]
SHOW = ["has_proof", "grammatical", "valid", "grounded", "correct", "valid_correct", "in_system", "valid_wrong"]
ROWS = ["overall", "deduction", "pw_all", "pw_d0", "pw_d1", "pw_d2", "pw_d3", "pw_d5", "folio", "bbh",
        "bbh_web_of_lies", "bbh_formal_fallacies", "bbh_boolean_expressions", "bbh_navigate",
        "bbh_multistep_arithmetic_two", "bbh_object_counting", "gsm8k", "gpqa_diamond", "gpqa_quant", "standard",
        "arc_challenge", "logiqa", "mmlu", "multihop", "hotpotqa", "2wikimqa", "musique"]
PLOT = [("overall", "all"), ("overall", "answerable"), ("pw_all", "answerable"), ("pw_all", "all"),
        ("folio", "answerable"), ("bbh", "answerable"), ("gsm8k", "all"), ("gpqa_diamond", "all"),
        ("standard", "all"), ("multihop", "all")]


def lookup(s: dict, name: str) -> dict | None:
    if name == "overall":
        return s["overall"]
    return s["per_group"].get(name) or s["per_bench"].get(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, default=Path("/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925"))
    ap.add_argument("--subdir", default="formal_bench_tagged")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/formal_mixture_sweep_20260925/tagged"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    T = {}
    for f in sorted(args.sweep_root.glob(f"*/{args.subdir}/summary.json")):
        m = RUN_RE.match(f.parent.parent.name)
        if m:
            T[(m["m"], int(m["x"]))] = json.load(open(f))
    with open(args.out_dir / "tagged_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "x", "bench", "subset", "metric", "value", "n"])
        for (mdl, x), s in sorted(T.items(), key=lambda kv: (MODELS.index(kv[0][0]), kv[0][1])):
            for name in ["overall", *s["per_group"], *s["per_bench"]]:
                v = lookup(s, name)
                for sub in ("all", "answerable"):
                    d = v[sub]
                    for k in SHOW + ["f1"]:
                        if k in d:
                            w.writerow([mdl, x, name, sub, k, f"{100 * d[k]:.2f}", d["n"]])
    for mdl in MODELS:
        xs = sorted(x for (m_, x) in T if m_ == mdl)
        if not xs:
            continue
        out = [f"# {mdl}: format-tagged benchmarks (% of items)\n",
               "cells: grammatical / valid / correct / in-system (valid proof whose own `ans` is right);",
               "`answerable` restricts to items with a yes/no or numeric reference.\n"]
        for sub in ("all", "answerable"):
            out += [f"\n## {sub}\n", "| bench | n | " + " | ".join(f"{x}%" for x in xs) + " |",
                    "|---|---:|" + "---|" * len(xs)]
            for name in ROWS:
                vals = [lookup(T[(mdl, x)], name) for x in xs]
                vals = [v[sub] if v else None for v in vals]
                if all(v is None or v["n"] == 0 for v in vals):
                    continue
                n = next(v["n"] for v in vals if v)
                cells = ["–" if not v or v["n"] == 0 else
                         f"{100*v['grammatical']:.0f} / {100*v['valid']:.0f} / "
                         f"{100*v.get('f1', v['correct']):.0f} / {100*v['in_system']:.0f}" for v in vals]
                out.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
        (args.out_dir / f"tagged_{mdl}.md").write_text("\n".join(out) + "\n")
        print("\n".join(out))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = ["grammatical", "valid", "correct", "in_system"]
    fig, axs = plt.subplots(len(PLOT), len(metrics), figsize=(3.0 * len(metrics), 2.2 * len(PLOT)), squeeze=False)
    for i, (name, sub) in enumerate(PLOT):
        for j, k in enumerate(metrics):
            ax = axs[i][j]
            for mdl in MODELS:
                pts = []
                for (m_, x), s in T.items():
                    v = lookup(s, name)
                    if m_ == mdl and v and v[sub]["n"]:
                        pts.append((x, 100 * v[sub].get("f1" if k == "correct" else k, v[sub][k])))
                if pts:
                    ax.plot(*zip(*sorted(pts)), marker="o", ms=3, label=mdl)
            ax.set_title(f"{name} [{sub}] {k}", fontsize=7)
            ax.tick_params(labelsize=6)
    axs[0][0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"tagged_curves.{ext}", dpi=120)


if __name__ == "__main__":
    main()
