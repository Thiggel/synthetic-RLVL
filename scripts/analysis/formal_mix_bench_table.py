#!/usr/bin/env python
"""Downstream benchmarks of the 2026-09-25 formal-CoT mixture sweep.

Reads lm-eval output written by scripts/slurm/jobs/gruenau_formal_mix_bench_2026-09-26.slurm
(<results-root>/<suite>/qwen35_<m>_dolci_rlvlgen_pXX_lr5em6_seed3407/, only
suites with a .complete marker) and scores the benchmarks of the "Lead into
Gold" paper exactly as scripts/analysis/overview_table.py does:
  - BBH: 3-shot CoT, all subtasks, the eight chain subtasks (macro), the rest,
    web of lies separately;
  - LongBench HotpotQA / 2Wiki / MuSiQue: F1 after truncating at the first newline;
  - GPQA-quant: the 73-item question-text regex subset of GPQA-Diamond
    (appendix_details.tex, "Benchmark subsets"), scored from the log-samples;
  - general benchmarks with the named metric (acc_norm where the paper used it).
Writes to --out-dir: bench_long.csv (model, x, benchmark, score), one wide
markdown table per model (bench_<model>.md) and bench_curves.{pdf,png}.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import gzip
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lm_eval_tasks" / "synthrlvl_ood"))
from utils import qa_f1_score  # noqa: E402

RUN_RE = re.compile(r"qwen35_(?P<m>[0-9.]+b)_dolci_rlvlgen_p(?P<x>\d\d)_lr5em6_seed3407$")
MODELS = ["0.8b", "2b", "9b"]
METRIC = {"gsm8k": "exact_match,strict-match", "mmlu": "acc,none", "arc_challenge": "acc_norm,none",
          "agieval_logiqa_en": "acc_norm,none", "hellaswag": "acc_norm,none", "piqa": "acc_norm,none",
          "winogrande": "acc,none", "humaneval": "pass@1,create_test", "mbpp": "pass_at_1,none"}
CH = ["web_of_lies", "tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_five_objects",
      "tracking_shuffled_objects_seven_objects", "logical_deduction_three_objects",
      "logical_deduction_five_objects", "logical_deduction_seven_objects", "formal_fallacies"]
QU = re.compile(r"\b\d+(\.\d+)?\s*(nm|mm|cm|km|kg|mol|ml|eV|keV|MeV|GeV|K|Hz|MHz|GHz|deg|°|%|pc|AU|M_|R_)", re.I)
QW = re.compile(r"\b(what is the (value|ratio|factor|energy|wavelength|number|probability|concentration|pH"
                r"|mass|radius|velocity|temperature)|calculate|compute|how many|by what factor|what fraction)\b", re.I)
STOPS = ("<|im_end|>", "</s>", "\n")


def is_quant(q: str) -> bool:
    return bool(QU.search(q) or (re.search(r"\d", q) and QW.search(q)))


def trunc(t: str) -> str:
    cut = len(t)
    for s in STOPS:
        i = t.find(s)
        if i != -1:
            cut = min(cut, i)
    return t[:cut].strip()


def samples(d: str, pat: str):
    for f in glob.glob(f"{d}/**/samples_{pat}_*.jsonl*", recursive=True):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            for line in fh:
                yield json.loads(line)


def results(d: str) -> dict:
    fs = sorted(glob.glob(f"{d}/**/results_*.json", recursive=True))
    return json.load(open(fs[-1]))["results"] if fs else {}


def score(v: dict, want: str | None):
    if want and want in v:
        return 100 * v[want]
    for k in ("exact_match,get-answer", "exact_match,none", "acc,none"):
        if k in v:
            return 100 * v[k]
    vals = [x for k, x in v.items() if k != "alias" and "stderr" not in k and isinstance(x, float)]
    return 100 * vals[0] if vals else None


def cell(root: Path, run: str, any_suite: bool = False) -> dict:
    out = {}
    def done(suite):
        d = root / suite / run
        return str(d) if (d / ".complete").exists() or (any_suite and d.is_dir()) else None

    if d := done("deduction"):
        for k, v in results(d).items():
            if m := re.match(r"synthrlvl_deduction_pw_d(\d)$", k):
                out[f"PW d{m[1]}"] = score(v, "exact_match,none")
    if d := done("cot"):
        for k, v in results(d).items():
            if m := re.match(r"synthrlvl_deduction_pw_cot_d(\d)$", k):
                out[f"PW CoT d{m[1]}"] = score(v, "exact_match,none")
    if d := done("folio_gpqa"):
        r = results(d)
        if "synthrlvl_folio" in r:
            out["FOLIO"] = score(r["synthrlvl_folio"], "exact_match,none")
        if "synthrlvl_gpqa_diamond" in r:
            out["GPQA-Diamond"] = score(r["synthrlvl_gpqa_diamond"], "exact_match,none")
        qs, rs = [], []
        for row in samples(d, "synthrlvl_gpqa_diamond"):
            (qs if is_quant(row["doc"]["question"]) else rs).append(float(row["exact_match"]))
        if qs:
            out["GPQA-quant"] = 100 * st.mean(qs)
            out["GPQA-rest"] = 100 * st.mean(rs)
            out["_gpqa_quant_n"] = len(qs)
    if d := done("bbh_nochat"):
        sub = {k[len("bbh_cot_fewshot_"):]: score(v, "exact_match,get-answer")
               for k, v in results(d).items() if k.startswith("bbh_cot_fewshot_")}
        sub = {k: v for k, v in sub.items() if v is not None}
        if sub:
            out["BBH (all)"] = st.mean(sub.values())
            ch = [sub[k] for k in CH if k in sub]
            out["BBH chain-8"] = st.mean(ch) if len(ch) == len(CH) else None
            out["BBH other"] = st.mean([v for k, v in sub.items() if k not in CH])
            for k in CH:
                if k in sub:
                    out[f"BBH {k}"] = sub[k]
    if d := done("multihop"):
        for task, name in (("hotpotqa", "HotpotQA"), ("2wikimqa", "2WikiMQA"), ("musique", "MuSiQue")):
            sc = []
            for row in samples(d, f"synthrlvl_longbench_{task}_standard"):
                resp = row.get("filtered_resps") or row.get("resps")
                while isinstance(resp, list) and resp:
                    resp = resp[0]
                pred = trunc(str(resp))
                sc.append(max((qa_f1_score(pred, str(a)) for a in row["doc"]["answers"]), default=0.0))
            if sc:
                out[name] = 100 * st.mean(sc)
    for suite, names in (("standard", ["gsm8k", "mmlu", "arc_challenge", "agieval_logiqa_en", "hellaswag",
                                       "piqa", "winogrande"]), ("code_nochat", ["humaneval", "mbpp"])):
        if d := done(suite):
            r = results(d)
            for t in names:
                if t in r:
                    out[t] = score(r[t], METRIC[t])
    return out


ORDER = (["PW d0", "PW d1", "PW d2", "PW d3", "PW d5", "PW CoT d0", "PW CoT d1", "PW CoT d2", "PW CoT d3",
          "PW CoT d5", "FOLIO", "BBH (all)", "BBH chain-8", "BBH other"] + [f"BBH {k}" for k in CH]
         + ["HotpotQA", "2WikiMQA", "MuSiQue", "GPQA-Diamond", "GPQA-quant", "GPQA-rest", "gsm8k", "mmlu",
            "arc_challenge", "agieval_logiqa_en", "hellaswag", "piqa", "winogrande", "humaneval", "mbpp"])
PLOT = ["PW d3", "PW d5", "PW CoT d3", "FOLIO", "BBH (all)", "BBH chain-8", "BBH web_of_lies", "HotpotQA",
        "2WikiMQA", "MuSiQue", "GPQA-Diamond", "GPQA-quant", "gsm8k", "mmlu", "arc_challenge",
        "agieval_logiqa_en", "hellaswag", "winogrande", "humaneval", "mbpp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path,
                    default=Path("/vol/tmp2/laitenbf/rlvl_data/lm_eval_results/formal_mix_20260926"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/formal_mixture_sweep_20260925/bench"))
    ap.add_argument("--include-incomplete", action="store_true",
                    help="also read suites without .complete (LIMIT smoke runs)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = set()
    for sd in args.results_root.glob("*"):
        runs |= {p.name for p in sd.glob("*") if RUN_RE.match(p.name)}
    T = collections.defaultdict(dict)  # (model, x) -> {bench: score}
    for run in sorted(runs):
        m = RUN_RE.match(run)
        T[(m["m"], int(m["x"]))] = cell(args.results_root, run, args.include_incomplete)
    with open(args.out_dir / "bench_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "x", "benchmark", "score"])
        for (mdl, x), d in sorted(T.items(), key=lambda kv: (MODELS.index(kv[0][0]), kv[0][1])):
            for b in ORDER:
                if d.get(b) is not None:
                    w.writerow([mdl, x, b, f"{d[b]:.2f}"])
    for mdl in MODELS:
        xs = sorted(x for (m_, x) in T if m_ == mdl)
        if not xs:
            continue
        lines = ["| benchmark | " + " | ".join(f"{x}%" for x in xs) + " |",
                 "|---|" + "---:|" * len(xs)]
        for b in ORDER:
            vals = [T[(mdl, x)].get(b) for x in xs]
            if all(v is None for v in vals):
                continue
            base = T[(mdl, 0)].get(b) if 0 in xs else None
            cells = []
            for x, v in zip(xs, vals):
                if v is None:
                    cells.append("–")
                elif x == 0 or base is None:
                    cells.append(f"{v:.1f}")
                else:
                    cells.append(f"{v:.1f} ({v - base:+.1f})")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        (args.out_dir / f"bench_{mdl}.md").write_text("\n".join(lines) + "\n")
        print(f"## {mdl}\n" + "\n".join(lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = 5
    rows = -(-len(PLOT) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, b in zip(axs.flat, PLOT):
        for mdl in MODELS:
            pts = sorted((x, d[b]) for (m_, x), d in T.items() if m_ == mdl and d.get(b) is not None)
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=3, label=mdl)
        ax.set_title(b, fontsize=9)
        ax.set_xlabel("% synthetic", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(PLOT):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"bench_curves.{ext}", dpi=130)


if __name__ == "__main__":
    main()
