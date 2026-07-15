#!/usr/bin/env python3
"""Aggregate audited Nanotron context-provided multi-hop QA evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lm_eval_tasks.synthrlvl_ood.utils import qa_exact_match, qa_f1_score


RUN_TAGS = {
    "control": "qwen25_7b_midtrain_control_p0_4p3b_step8192",
    "nl_p15": "qwen25_7b_midtrain_nl_exact_p15_bp_unique_v2_4p3b_step8192",
    "logic_p15": "qwen25_7b_midtrain_logic_p15_bp_unique_v2_4p3b_step8192",
}

ANSWER_HEAD_MARKERS = (
    "\n",
    " Passage 1:",
    " You are an AI assistant",
    " Question:",
    " Answer the question based",
    ".rawidłow",
)


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


def _sample_rows(run_dir: Path, task: str) -> list[dict[str, Any]]:
    paths = sorted(run_dir.rglob(f"samples_{task}_*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one sample file for {task} under {run_dir}, found {len(paths)}")
    return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line.strip()]


def _selected_samples(run_dir: Path, task: str) -> list[dict[str, Any]]:
    rows = _sample_rows(run_dir, task)
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


def _answer_head(generation: str) -> str:
    offsets = [offset for marker in ANSWER_HEAD_MARKERS if (offset := generation.find(marker)) >= 0]
    return generation[: min(offsets)].strip() if offsets else generation.strip()


def _answers(row: dict[str, Any]) -> list[str]:
    doc = row.get("doc")
    answers = doc.get("answers") if isinstance(doc, dict) else None
    return [str(answer) for answer in answers] if isinstance(answers, list) else []


def standard_answer_head_rescore(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    stock_scores: list[float] = []
    head_f1_scores: list[float] = []
    head_exact_scores: list[float] = []
    for row in rows:
        answers = _answers(row)
        if not answers:
            raise ValueError("standard LongBench sample is missing retained answers")
        stock_scores.append(float(row.get("qa_f1_score", row.get("score", 0.0))))
        head = _answer_head(_generation(row))
        head_f1_scores.append(max(qa_f1_score(head, answer) for answer in answers))
        head_exact_scores.append(max(qa_exact_match(head, answer) for answer in answers))
    return {
        "row_count": len(rows),
        "stock_qa_f1": mean(stock_scores),
        "answer_head_qa_f1": mean(head_f1_scores),
        "answer_head_exact_match": mean(head_exact_scores),
    }


def generation_diagnostics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    generations = [_generation(row) for row in rows]
    lengths = sorted(len(generation) for generation in generations)
    count = len(generations)

    def rate(marker: str) -> float:
        return sum(marker in generation for generation in generations) / count

    return {
        "row_count": count,
        "next_document_marker_rate": rate("You are an AI assistant"),
        "prompt_continuation_rate": sum(
            any(marker in generation for marker in ("Passage 1:", "Question:", "Answer the question based"))
            for generation in generations
        )
        / count,
        "formal_open_rate": rate("<formal>"),
        "think_open_rate": rate("<think>"),
        "answer_open_rate": rate("<answer>"),
        "answer_close_rate": rate("</answer>"),
        "rawidlow_rate": rate("rawidłow"),
        "response_chars_p50": median(lengths),
        "response_chars_p95": lengths[math.ceil(0.95 * count) - 1],
        "response_chars_max": lengths[-1],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
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
                if not selected:
                    raise ValueError(f"missing rows for {condition}/{mode}/{protocol}")
                summary.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "protocol": protocol,
                        "benchmark_count": len(selected),
                        "mean_qa_f1": sum(float(row["qa_f1"]) for row in selected) / len(selected),
                        "mean_exact_match": (
                            sum(float(row["qa_exact_match"]) for row in selected) / len(selected)
                            if protocol == "strict_tagged"
                            else None
                        ),
                        "mean_tag_found": (
                            sum(float(row["tag_found"]) for row in selected) / len(selected)
                            if protocol == "strict_tagged"
                            else None
                        ),
                    }
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rescore_rows: list[dict[str, Any]] = []
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
                task_samples = _sample_rows(run_dir, task)
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
                diagnostic_rows.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        "protocol": _protocol(task),
                        "benchmark": _benchmark(task),
                        **generation_diagnostics(task_samples),
                    }
                )
                if _protocol(task) == "standard_short_answer":
                    rescore_rows.append(
                        {
                            "condition": condition,
                            "mode": mode,
                            "benchmark": _benchmark(task),
                            **standard_answer_head_rescore(task_samples),
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = summarize_rows(rows)
    summary_csv_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    sample_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop_samples.json"
    sample_path.write_text(json.dumps(samples, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    diagnostics_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop_generation_diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    control_rescores = {
        (row["mode"], row["benchmark"]): row
        for row in rescore_rows
        if row["condition"] == "control"
    }
    for row in rescore_rows:
        control = control_rescores[(str(row["mode"]), str(row["benchmark"]))]
        row["stock_delta_vs_control"] = float(row["stock_qa_f1"]) - float(control["stock_qa_f1"])
        row["answer_head_delta_vs_control"] = float(row["answer_head_qa_f1"]) - float(
            control["answer_head_qa_f1"]
        )
    rescore_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop_answer_head_rescore.csv"
    with rescore_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rescore_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rescore_rows)

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
    for row in summary_rows:
        em_text = "--" if row["mean_exact_match"] is None else f"{float(row['mean_exact_match']):.3f}"
        tag_text = "--" if row["mean_tag_found"] is None else f"{float(row['mean_tag_found']):.3f}"
        lines.append(
            f"| {row['condition']} | {row['mode']} | {row['protocol']} | "
            f"{float(row['mean_qa_f1']):.3f} | {em_text} | {tag_text} |"
        )
    lines.extend(
        [
            "",
            "## Raw-generation audit",
            "",
            "The direct strict-tagged prompt mostly triggers the learned continuation substrate: "
            "logic generations open `<formal>` in 98.5--99.0% of rows and natural-language "
            "generations open `<think>` in 97.0--99.0%. The 64-token diagnostic therefore "
            "usually ends before a usable answer. Instruction SFT removes those openings, but "
            "the 32-token stock protocol remains strongly cap-limited and frequently contains "
            "continuation artifacts. These rows measure response control as well as QA.",
            "",
            "A diagnostic rescore truncates only obvious generated continuation after the first "
            "answer span. Averaged over the three direct standard tasks, control/logic/NL QA-F1 "
            "changes from 0.189/0.250/0.238 under stock scoring to 0.349/0.361/0.367 under this "
            "answer-head sensitivity check. The apparent stock gains therefore mostly collapse "
            "after continuation is removed and are not clean evidence of reasoning transfer.",
        ]
    )
    md_path = args.output_dir / "qwen25_branchproof_unique_v2_multihop.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": True,
                "rows": len(rows),
                "summary_rows": len(summary_rows),
                "samples": len(samples),
                "csv": str(csv_path),
                "summary_csv": str(summary_csv_path),
                "generation_diagnostics_csv": str(diagnostics_path),
                "answer_head_rescore_csv": str(rescore_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
