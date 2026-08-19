#!/usr/bin/env python3
"""Three-way multi-hop readout for the document-preserving 2.5B rerun.

Replicates the methodology of the ACCEPTED 5B readout
(analysis/dolmino_threeway_post_sft_readout_20260806.md) so the two are
directly comparable:

  * raw qa_f1 macros per protocol (standard / tagged), averaged over the three
    LongBench benchmarks;
  * tag_found rates on the tagged protocol;
  * compliance-corrected ("fallback") rescoring of the tagged protocol, using
    the repo's own extract_answer(..., allow_raw_fallback=True) -- the tagged
    metric itself scores with allow_raw_fallback=False, so a correct bare
    answer emitted without the <answer> wrapper is scored 0;
  * paired example-level bootstrap (10k resamples, pooled across benchmarks).

Scoring functions are imported from the task module rather than reimplemented.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

from lm_eval_tasks.synthrlvl_ood.utils import extract_answer, qa_f1_score

BENCHMARKS = ("hotpotqa", "2wikimqa", "musique")
CONDITIONS = ("control", "logic", "nl_exact")
RUN_TEMPLATE = "qwen25_7b_dolmino_{condition}_docpack_2p5b_dolci_100k_lr5em6_multihop"


def sample_rows(run_dir: Path, task: str) -> list[dict]:
    paths = sorted(glob.glob(str(run_dir / "**" / f"samples_{task}_*.jsonl"), recursive=True))
    if not paths:
        raise SystemExit(f"missing samples for {task} under {run_dir}")
    with open(paths[0], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def raw_text(row: dict) -> str:
    resps = row.get("filtered_resps") or row.get("resps") or [""]
    value = resps[0]
    while isinstance(value, list) and value:
        value = value[0]
    return "" if value is None else str(value)


def fallback_f1(row: dict) -> float:
    extracted = extract_answer(raw_text(row), allow_raw_fallback=True)
    answers = row.get("doc", {}).get("answers") or []
    return max((qa_f1_score(extracted, str(a)) for a in answers), default=0.0)


def bootstrap(deltas: list[float], resamples: int, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    mean = sum(deltas) / n
    means = []
    for _ in range(resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return mean, means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    # per-condition, per-protocol pooled per-example scores (order-aligned by doc_id)
    pooled: dict[tuple[str, str], list[float]] = {}
    macros: dict[tuple[str, str], float] = {}
    tag_rates: dict[str, float] = {}

    for condition in CONDITIONS:
        run_dir = args.root / RUN_TEMPLATE.format(condition=condition)
        audit = json.loads((run_dir / "multihop_audit.json").read_text(encoding="utf-8"))
        if not audit.get("accepted") or not audit.get("require_full"):
            raise SystemExit(f"unaccepted or limited audit: {run_dir}")

        for protocol in ("standard", "tagged"):
            per_bench_means, pooled_scores = [], []
            for bench in BENCHMARKS:
                task = f"synthrlvl_longbench_{bench}_{protocol}"
                rows = sorted(sample_rows(run_dir, task), key=lambda r: r["doc_id"])
                scores = [float(r["qa_f1_score"]) for r in rows]
                per_bench_means.append(sum(scores) / len(scores))
                pooled_scores.extend(scores)
            macros[(condition, protocol)] = sum(per_bench_means) / len(per_bench_means)
            pooled[(condition, protocol)] = pooled_scores

        # tagged fallback rescoring + tag rate
        fb_bench_means, fb_pooled, tag_flags = [], [], []
        for bench in BENCHMARKS:
            task = f"synthrlvl_longbench_{bench}_tagged"
            rows = sorted(sample_rows(run_dir, task), key=lambda r: r["doc_id"])
            scores = [fallback_f1(r) for r in rows]
            fb_bench_means.append(sum(scores) / len(scores))
            fb_pooled.extend(scores)
            tag_flags.extend(float(r.get("tag_found", 0.0)) for r in rows)
        macros[(condition, "tagged_fallback")] = sum(fb_bench_means) / len(fb_bench_means)
        pooled[(condition, "tagged_fallback")] = fb_pooled
        tag_rates[condition] = sum(tag_flags) / len(tag_flags)

    lines = ["# Docpack 2.5B rerun: three-way multi-hop readout", ""]
    lines.append(f"Pooled n per condition/protocol: {len(pooled[('control','tagged')])}")
    lines.append("")
    lines.append("| protocol | control | logic | nl_exact |")
    lines.append("| --- | --- | --- | --- |")
    for protocol in ("standard", "tagged", "tagged_fallback"):
        cells = " | ".join(f"{macros[(c, protocol)]:.4f}" for c in CONDITIONS)
        lines.append(f"| {protocol} | {cells} |")
    lines.append("")
    lines.append("| condition | tag_found rate |")
    lines.append("| --- | --- |")
    for condition in CONDITIONS:
        lines.append(f"| {condition} | {tag_rates[condition]:.4f} |")
    lines.append("")
    lines.append(f"Paired example-level bootstrap ({args.resamples} resamples, seed {args.seed}):")
    lines.append("")
    lines.append("| protocol | contrast | delta | ci_low | ci_high | significant |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for protocol in ("standard", "tagged", "tagged_fallback"):
        for a, b in (("logic", "control"), ("logic", "nl_exact"), ("nl_exact", "control")):
            deltas = [x - y for x, y in zip(pooled[(a, protocol)], pooled[(b, protocol)])]
            mean, lo, hi = bootstrap(deltas, args.resamples, args.seed)
            sig = "yes" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "no"
            lines.append(f"| {protocol} | {a}-{b} | {mean:+.4f} | {lo:+.4f} | {hi:+.4f} | {sig} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
