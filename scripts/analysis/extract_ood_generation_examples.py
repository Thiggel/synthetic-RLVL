#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "lm_eval_tasks" / "synthrlvl_ood"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from utils import extract_answer, process_gsm8k_tagged  # noqa: E402


TASK_FILES = {
    "gsm8k": "samples_synthrlvl_gsm8k_tagged_*.jsonl",
    "hotpotqa": "samples_synthrlvl_longbench_hotpotqa_tagged_*.jsonl",
    "2wikimqa": "samples_synthrlvl_longbench_2wikimqa_tagged_*.jsonl",
    "musique": "samples_synthrlvl_longbench_musique_tagged_*.jsonl",
}


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No sample file matching {pattern} under {root}")
    return matches[-1]


def _load_samples(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["doc_id"])] = row
    return rows


def _raw_response(row: dict[str, Any]) -> str:
    resps = row.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0]:
        return str(resps[0][0])
    if resps:
        return str(resps[0])
    return ""


def _question(row: dict[str, Any]) -> str:
    doc = row.get("doc") or {}
    return " ".join(str(doc.get("question", "")).split())


def _gold(row: dict[str, Any]) -> str:
    doc = row.get("doc") or {}
    if "answers" in doc:
        return ", ".join(str(x) for x in doc.get("answers", []))
    answer = str(doc.get("answer", row.get("target", "")))
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()
    return " ".join(answer.split())


def _metric(row: dict[str, Any], task: str) -> float:
    if task == "gsm8k":
        return float(process_gsm8k_tagged(row.get("doc") or {}, [_raw_response(row)])["exact_match"])
    return float(row.get("qa_f1_score", row.get("score", 0.0)))


def _metric_label(task: str) -> str:
    return "EM" if task == "gsm8k" else "F1"


def _short(text: str, limit: int) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2].rstrip()
    tail = text[-limit // 2 :].lstrip()
    return f"{head}\n...[truncated]...\n{tail}"


def _pick_doc_ids(logic_rows: dict[int, dict[str, Any]], nl_rows: dict[int, dict[str, Any]], task: str, n: int) -> list[int]:
    common = sorted(set(logic_rows) & set(nl_rows))
    logic_better = [i for i in common if _metric(logic_rows[i], task) > _metric(nl_rows[i], task)]
    nl_better = [i for i in common if _metric(nl_rows[i], task) > _metric(logic_rows[i], task)]
    both_wrong = [i for i in common if _metric(logic_rows[i], task) == 0.0 and _metric(nl_rows[i], task) == 0.0]
    selected: list[int] = []
    buckets = [logic_better, nl_better, both_wrong, common]
    positions = [0 for _ in buckets]
    while len(selected) < n:
        advanced = False
        for bucket_idx, bucket in enumerate(buckets):
            while positions[bucket_idx] < len(bucket):
                doc_id = bucket[positions[bucket_idx]]
                positions[bucket_idx] += 1
                advanced = True
                if doc_id in selected:
                    continue
                selected.append(doc_id)
                break
            if len(selected) >= n:
                break
        if not advanced:
            break
    for doc_id in common:
        if len(selected) >= n:
            break
        if doc_id not in selected:
            selected.append(doc_id)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paired logic/NL OOD generation examples from lm-eval samples.")
    parser.add_argument("--logic-root", required=True)
    parser.add_argument("--nl-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples-per-task", type=int, default=3)
    parser.add_argument("--response-chars", type=int, default=1400)
    args = parser.parse_args()

    logic_root = Path(args.logic_root)
    nl_root = Path(args.nl_root)
    lines: list[str] = [
        "# OOD Generation Examples",
        "",
        f"Logic root: `{logic_root}`",
        f"NL root: `{nl_root}`",
        "",
        "Examples are paired by `doc_id`. GSM8K metric is exact match after tag/numeric extraction; HotpotQA, 2WikiMultiHopQA, and MuSiQue metric shown here is token F1 after strict answer-tag extraction.",
        "",
    ]
    for task, pattern in TASK_FILES.items():
        logic_rows = _load_samples(_find_one(logic_root, pattern))
        nl_rows = _load_samples(_find_one(nl_root, pattern))
        selected = _pick_doc_ids(logic_rows, nl_rows, task, int(args.examples_per_task))
        lines += [f"## {task}", ""]
        for doc_id in selected:
            logic = logic_rows[doc_id]
            nl = nl_rows[doc_id]
            logic_raw = _raw_response(logic)
            nl_raw = _raw_response(nl)
            logic_extracted = extract_answer(logic_raw, allow_raw_fallback=False)
            nl_extracted = extract_answer(nl_raw, allow_raw_fallback=False)
            lines += [
                f"### doc_id {doc_id}",
                "",
                f"Question: {_short(_question(logic), 1200)}",
                "",
                f"Gold: {_gold(logic)}",
                "",
                f"Logic explicit extraction: `{logic_extracted}`; {_metric_label(task)}={_metric(logic, task):.3f}; tag_found={float(logic.get('tag_found', 0.0)):.3f}",
                "",
                "Logic generation:",
                "",
                "```text",
                _short(logic_raw, int(args.response_chars)),
                "```",
                "",
                f"NL explicit extraction: `{nl_extracted}`; {_metric_label(task)}={_metric(nl, task):.3f}; tag_found={float(nl.get('tag_found', 0.0)):.3f}",
                "",
                "NL generation:",
                "",
                "```text",
                _short(nl_raw, int(args.response_chars)),
                "```",
                "",
            ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
