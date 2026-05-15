#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from logic_engine import LogicEngine
from synthrlvl.config import task_config_from_cfg
from synthrlvl.metrics import OutputEvaluator, extract_tag, _is_answer_match
from synthrlvl.task import TaskBuilder
from synthrlvl.types import StepRange


def _state_from_formula(formula: str, pred_to_value: dict[str, str]) -> str | None:
    m = re.match(r"\s*([A-Z])([a-r])\s*$", formula or "")
    if not m:
        return None
    return pred_to_value.get(m.group(1))


def _line_formula(text: str) -> str:
    raw = (text or "").strip()
    if ". " in raw:
        raw = raw.split(". ", 1)[1]
    if " ; " in raw:
        raw = raw.split(" ; ", 1)[0]
    return raw.strip()


def classify_generation(sample, generation: str, evaluator: OutputEvaluator, engine: LogicEngine) -> dict[str, Any]:
    meta = dict(sample.metadata)
    pred_to_value: dict[str, str] = {}
    for line in sample.gold_first_modality_lines:
        if "x: x is " in line:
            body = line.split(". ", 1)[1] if ". " in line else line
            pred, value = body.split("x: x is ", 1)
            pred_to_value[pred.strip()] = value.strip()

    answer = extract_tag(generation, "answer")
    premises, proof, conclusion = evaluator._extract_logic_components(generation, evaluator._logic_block_tag(sample_cfg.template))
    score = evaluator.evaluate(
        generation,
        template=sample_cfg.template,
        gold_answer=sample.answer,
        gold_logic_premises=sample.logic_premises,
        gold_logic_conclusion=sample.logic_conclusion,
        prefill=sample_cfg.prefill,
        gold_first_modality_lines=sample.gold_first_modality_lines,
    )
    report = engine.analyze_proof_citation_free(premises, conclusion, proof) if premises and proof and conclusion else None
    valid_prefix = 0
    first_invalid = None
    last_valid_formula = None
    if report is not None:
        for line in report.lines:
            if line.valid:
                valid_prefix += 1
                last_valid_formula = _line_formula(line.text)
            elif first_invalid is None:
                first_invalid = {"line_number": line.line_number, "text": line.text, "error": line.error or line.syntax_error}
                break
    conclusion_state = _state_from_formula(conclusion, pred_to_value)
    last_valid_state = _state_from_formula(last_valid_formula or "", pred_to_value)
    shortcut = meta.get("shortcut_branch_answer")
    gold = meta.get("gold_answer", sample.answer)

    if score.citation_free_valid and score.correct:
        category = "correct_supported_by_valid_proof"
    elif score.correct:
        category = "correct_but_invalid_proof"
    elif shortcut and _is_answer_match(answer, str(shortcut)):
        category = "shortcut_answer"
    elif conclusion_state and shortcut and conclusion_state == shortcut:
        category = "shortcut_conclusion"
    elif last_valid_state and last_valid_state == gold:
        category = "gold_reached_then_broken"
    elif last_valid_state:
        category = "partial_valid_prefix"
    elif answer:
        category = "wrong_nonshortcut_answer"
    else:
        category = "format_or_empty_failure"

    return {
        "depth": sample.depth,
        "gold_answer": gold,
        "shortcut_answer": shortcut,
        "predicted_answer": answer,
        "conclusion": conclusion,
        "conclusion_state": conclusion_state,
        "last_valid_formula": last_valid_formula,
        "last_valid_state": last_valid_state,
        "valid_prefix_lines": valid_prefix,
        "first_invalid": first_invalid,
        "correct": score.correct,
        "format_ok": score.format_ok,
        "syntactic": score.syntactic,
        "citation_free_valid": score.citation_free_valid,
        "category": category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="conf/posttrain_grpo_hard_v5_500.yaml")
    parser.add_argument("--step-min", type=int, default=4)
    parser.add_argument("--step-max", type=int, default=20)
    parser.add_argument("--samples-per-step", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Import lazily because vLLM startup is expensive and optional for tests.
    from synthrlvl.eval_loop import _VLLMTextGenerator

    cfg = OmegaConf.load(args.config)
    global sample_cfg
    sample_cfg = task_config_from_cfg(cfg)
    evaluator = OutputEvaluator()
    engine = LogicEngine()
    gen = _VLLMTextGenerator(
        model_path=args.checkpoint,
        tokenizer_path=args.checkpoint,
        batch_size=1024,
        gpu_memory_utilization=0.80,
        lora_adapter_path=None,
        lora_base_model_name=None,
    )

    rows = []
    for step in range(args.step_min, args.step_max + 1):
        step_cfg = replace(sample_cfg, val_steps=StepRange(step, step))
        samples = TaskBuilder(step_cfg).build_samples(args.samples_per_step, train=False)
        outs = gen.generate([s.prompt for s in samples], max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0)
        for sample, out in zip(samples, outs, strict=True):
            row = classify_generation(sample, out, evaluator, engine)
            row["generation"] = out
            rows.append(row)

    counts = Counter(r["category"] for r in rows)
    by_depth = defaultdict(Counter)
    prefix_by_depth = defaultdict(list)
    for r in rows:
        by_depth[r["depth"]][r["category"]] += 1
        prefix_by_depth[r["depth"]].append(r["valid_prefix_lines"])
    summary = {
        "checkpoint": args.checkpoint,
        "total": len(rows),
        "category_counts": dict(counts),
        "category_rates": {k: v / max(1, len(rows)) for k, v in counts.items()},
        "by_depth": {str(k): dict(v) for k, v in sorted(by_depth.items())},
        "valid_prefix_mean_by_depth": {str(k): sum(v) / max(1, len(v)) for k, v in sorted(prefix_by_depth.items())},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
