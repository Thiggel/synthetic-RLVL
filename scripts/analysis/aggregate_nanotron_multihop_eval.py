#!/usr/bin/env python3
"""Aggregate audited Nanotron context-provided multi-hop QA evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RUN_TAGS = {
    "control": "qwen25_7b_midtrain_control_p0_4p3b_step8192",
    "logic_p15": "qwen25_7b_midtrain_logic_p15_bp_unique_v2_4p3b_step8192",
    "nl_p15": "qwen25_7b_midtrain_nl_exact_p15_bp_unique_v2_4p3b_step8192",
}


def _protocol(task: str) -> str:
    return "strict_tagged" if task.endswith("_tagged") else "standard_short_answer"


def _benchmark(task: str) -> str:
    for name in ("hotpotqa", "2wikimqa", "musique"):
        if name in task:
            return name
    return task


def _generation(row: dict[str, Any]) -> str:
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    responses = row.get("resps")
    if isinstance(responses, list) and responses:
        value = responses[0]
        if isinstance(value, list) and value:
            return str(value[0])
        return str(value)
    return ""


def _prompt(row: dict[str, Any]) -> str:
    arguments = row.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    for value in arguments.values():
        if not isinstance(value, dict):
            continue
        for key, candidate in value.items():
            if str(key).startswith("arg_") and isinstance(candidate, str):
                return candidate
    return ""


def _selected_samples(run_dir: Path, task: str) -> list[dict[str, Any]]:
    paths = sorted(run_dir.rglob(f"samples_{task}_*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one sample file for {task} under {run_dir}, found {len(paths)}")
    rows = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: list[dict[str, Any]] = []
    for label, predicate in (
        ("correct", lambda row: float(row.get("qa_f1_score", row.get("score", 0.0))) >= 0.999),
        ("partial", lambda row: 0.0 < float(row.get("qa_f1_score", row.get("score", 0.0))) < 0.999),
        ("incorrect", lambda row: float(row.get("qa_f1_score", row.get("score", 0.0))) == 0.0),
    ):
        match = next((row for row in rows if predicate(row)), None)
        if match is None:
            continue
        prompt = _prompt(match)
        selected.append(
            {
                "case": label,
                "doc_id": match.get("doc_id"),
                "qa_f1": float(match.get("qa_f1_score", match.get("score", 0.0))),
                "target": match.get("target"),
                "generation": _generation(match),
                "prompt_tail": prompt[-2000:],
            }
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for condition, tag in RUN_TAGS.items():
        for mode in ("direct", "instruction"):
            run_dir = args.root / f"{tag}_{mode}"
            audit_path = run_dir / "multihop_audit.json"
            if not audit_path.is_file():
                raise SystemExit(f"missing audit: {audit_path}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not audit.get("accepted") or not audit.get("require_full"):
                raise SystemExit(f"unaccepted or limited audit: {audit_path}")
            for task, metrics in audit["task_metrics"].items():
                rows.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "protocol": _protocol(task),
                        "benchmark": _benchmark(task),
                        "task": task,
                        "qa_f1": metrics.get("qa_f1_score,none"),
                        "qa_exact_match": metrics.get("qa_exact_match,none"),
                        "tag_found": metrics.get("tag_found,none"),
                        "extracted_nonempty": metrics.get("extracted_nonempty,none"),
                    }
                )
                for sample in _selected_samples(run_dir, task):
                    samples.append({"condition": condition, "mode": mode, "task": task, **sample})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sample_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop_samples.json"
    sample_path.write_text(json.dumps(samples, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    lines = [
        "# Qwen2.5 BranchProof-v2 Multi-Hop QA",
        "",
        "Context-provided LongBench evaluation; this does not test retrieval or proof validity.",
        "",
        "The standard-short-answer rows reproduce the stock LongBench prompt and 32-token decoding. "
        "The strict-tagged rows test transfer to the synthetic `<answer>...</answer>` response contract.",
        "",
        "| condition | mode | protocol | mean QA F1 | mean exact match | tag found |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for condition in RUN_TAGS:
        for mode in ("direct", "instruction"):
            for protocol in ("standard_short_answer", "strict_tagged"):
                selected = [
                    row
                    for row in rows
                    if row["condition"] == condition
                    and row["mode"] == mode
                    and row["protocol"] == protocol
                ]
                mean_f1 = sum(float(row["qa_f1"]) for row in selected) / len(selected)
                if protocol == "strict_tagged":
                    mean_em = sum(float(row["qa_exact_match"]) for row in selected) / len(selected)
                    mean_tag = sum(float(row["tag_found"]) for row in selected) / len(selected)
                    em_text, tag_text = f"{mean_em:.3f}", f"{mean_tag:.3f}"
                else:
                    em_text = tag_text = "--"
                lines.append(
                    f"| {condition} | {mode} | {protocol} | {mean_f1:.3f} | "
                    f"{em_text} | {tag_text} |"
                )
    md_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": True, "rows": len(rows), "samples": len(samples), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
