#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from omegaconf import OmegaConf

from hfsa_validity_analysis_lib import classify_generation
from synthrlvl.config import task_config_from_cfg
from synthrlvl.eval_loop import _VLLMTextGenerator
from synthrlvl.task import TaskBuilder
from synthrlvl.types import StepRange


def parse_steps(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def replace_initial_marker(prompt: str, *, first_const: str, old_marker: str, new_marker: str) -> str:
    lines = prompt.splitlines()
    target_prefix = f"2. {first_const} is "
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(target_prefix) and stripped.endswith("."):
            lines[idx] = line.replace(f"{first_const} is {old_marker}.", f"{first_const} is {new_marker}.")
            return "\n".join(lines)
    # Fallback: replace the first exact marker fact after the initial state.
    return prompt.replace(f"{first_const} is {old_marker}.", f"{first_const} is {new_marker}.", 1)


def make_records(*, config_path: Path, steps: list[int], prompts_per_step: int, seed: int, start_index: int, probe_types: list[str]) -> list[dict[str, Any]]:
    cfg = OmegaConf.load(config_path)
    cfg.seed = int(seed)
    task_cfg = task_config_from_cfg(cfg)
    records: list[dict[str, Any]] = []
    for step in steps:
        step_cfg = replace(task_cfg, val_steps=StepRange(step, step))
        samples = TaskBuilder(step_cfg).build_samples(prompts_per_step, train=False, start_index=start_index + step * 100_000)
        for sample_idx, sample in enumerate(samples):
            meta = dict(sample.metadata)
            normal = {
                "probe_type": "normal",
                "step": step,
                "sample_index": sample_idx,
                "prompt": sample.prompt,
                "gold_answer": sample.answer,
                "shortcut_answer": meta.get("shortcut_branch_answer"),
                "metadata": meta,
                "expected_branch": 0,
            }
            if "normal" in probe_types:
                records.append(normal)
            if "swap_marker" in probe_types:
                branch_markers = meta.get("branch_markers") or []
                branch_states = meta.get("branch_states") or []
                path_constants = meta.get("path_constants") or []
                if len(branch_markers) >= 2 and len(branch_states) >= 2 and branch_markers[0] and branch_markers[1] and path_constants:
                    first_const = str(path_constants[0])
                    old_marker = str(branch_markers[0][0])
                    new_marker = str(branch_markers[1][0])
                    alt_prompt = replace_initial_marker(
                        sample.prompt,
                        first_const=first_const,
                        old_marker=old_marker,
                        new_marker=new_marker,
                    )
                    alt_meta = dict(meta)
                    alt_meta["probe_intervention"] = "swap_initial_marker_to_branch1"
                    alt_meta["original_gold_answer"] = sample.answer
                    alt_meta["original_shortcut_answer"] = meta.get("shortcut_branch_answer")
                    records.append(
                        {
                            "probe_type": "swap_marker",
                            "step": step,
                            "sample_index": sample_idx,
                            "prompt": alt_prompt,
                            "gold_answer": str(branch_states[1][-1]),
                            "shortcut_answer": sample.answer,
                            "metadata": alt_meta,
                            "expected_branch": 1,
                        }
                    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled HFSA easy diagnostic samples with vLLM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="conf/posttrain_grpo_hard_fsa_schema_easy_500.yaml")
    parser.add_argument("--steps", default="5,10,15,20")
    parser.add_argument("--prompts-per-step", type=int, default=12)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--start-index", type=int, default=500_000)
    parser.add_argument("--probe-types", default="normal,swap_marker")
    args = parser.parse_args()

    records = make_records(
        config_path=Path(args.config),
        steps=parse_steps(args.steps),
        prompts_per_step=int(args.prompts_per_step),
        seed=int(args.seed),
        start_index=int(args.start_index),
        probe_types=[p.strip() for p in args.probe_types.split(",") if p.strip()],
    )
    print(f"[probe] built {len(records)} prompts for {args.condition}", flush=True)

    generator = _VLLMTextGenerator(
        model_path=str(args.checkpoint),
        tokenizer_path=str(args.checkpoint),
        batch_size=int(args.batch_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        lora_adapter_path=None,
        lora_base_model_name=None,
    )
    grouped = generator.generate_many(
        [r["prompt"] for r in records],
        max_new_tokens=int(args.max_new_tokens),
        num_samples=int(args.num_generations),
        temperature=float(args.temperature),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for rec, generations in zip(records, grouped, strict=True):
            for gen_idx, generation in enumerate(generations):
                cls = classify_generation(
                    generation=generation,
                    prompt=rec["prompt"],
                    step=int(rec["step"]),
                    gold_answer=str(rec["gold_answer"]),
                    shortcut_answer=rec.get("shortcut_answer"),
                    metadata=rec.get("metadata"),
                    source="targeted_probe",
                )
                row = {
                    "condition": args.condition,
                    "checkpoint": args.checkpoint,
                    "probe_type": rec["probe_type"],
                    "step": rec["step"],
                    "sample_index": rec["sample_index"],
                    "generation_index": gen_idx,
                    "prompt": rec["prompt"],
                    "generation": generation,
                    "metadata": rec.get("metadata", {}),
                    "expected_branch": rec.get("expected_branch"),
                    **{k: v for k, v in cls.items() if k not in {"proof_formulas", "atom_texts"}},
                    "proof_formulas": cls.get("proof_formulas", []),
                    "atom_texts": cls.get("atom_texts", []),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"[probe] wrote {count} generations to {out}", flush=True)

    del generator
    gc.collect()


if __name__ == "__main__":
    main()
