#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthrlvl.config import eval_loop_config_from_grpo, eval_loop_config_from_sft, task_config_from_cfg
from synthrlvl.eval_loop import UnifiedEvaluator


def _parse_k_values(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--k-values must contain at least one integer")
    return values


def _detect_profile(cfg) -> str:
    if "eval" in cfg and "synthetic_step_min" in cfg.eval:
        return "grpo"
    if "validation" in cfg and "step_min" in cfg.validation:
        return "sft"
    raise ValueError("Could not detect config profile. Use --profile {sft,grpo}.")


def _apply_eval_overrides(cfg, profile: str, args: argparse.Namespace) -> None:
    section = cfg.eval if profile == "grpo" else cfg.validation
    if args.backend:
        section.generation_backend = args.backend
    if args.step_min is not None:
        key = "synthetic_step_min" if profile == "grpo" else "step_min"
        section[key] = int(args.step_min)
    if args.step_max is not None:
        key = "synthetic_step_max" if profile == "grpo" else "step_max"
        section[key] = int(args.step_max)
    if args.samples_per_step is not None:
        key = "synthetic_samples_per_step" if profile == "grpo" else "samples_per_step"
        section[key] = int(args.samples_per_step)
    if args.max_new_tokens is not None:
        section.max_new_tokens = int(args.max_new_tokens)
    if args.batch_size is not None:
        section.generation_batch_size = int(args.batch_size)
    if args.gpu_memory_utilization is not None:
        section.vllm_gpu_memory_utilization = float(args.gpu_memory_utilization)
    if args.num_generations is not None:
        section.sampled_num_generations = int(args.num_generations)
    if args.k_values is not None:
        section.sampled_k_values = args.k_values
    if args.temperature is not None:
        section.sampled_temperature = float(args.temperature)
    section.sampled_enabled = True
    if args.constrained_enabled:
        if profile == "grpo":
            section.constrained_enabled = True
            section.constrained_samples_per_step = int(args.constrained_samples_per_step)
            section.constrained_num_generations = int(args.constrained_num_generations)
            section.constrained_candidates_per_line = int(args.constrained_candidates_per_line)
            section.constrained_max_lines = int(args.constrained_max_lines)
            section.constrained_max_line_tokens = int(args.constrained_max_line_tokens)
            section.constrained_suffix_max_tokens = int(args.constrained_suffix_max_tokens)
            section.constrained_temperature = float(args.constrained_temperature)
            section.constrained_k_values = args.constrained_k_values
        else:
            cfg.constrained_eval.enabled = True
            cfg.constrained_eval.samples_per_step = int(args.constrained_samples_per_step)
            cfg.constrained_eval.num_generations = int(args.constrained_num_generations)
            cfg.constrained_eval.candidates_per_line = int(args.constrained_candidates_per_line)
            cfg.constrained_eval.max_lines = int(args.constrained_max_lines)
            cfg.constrained_eval.max_line_tokens = int(args.constrained_max_line_tokens)
            cfg.constrained_eval.suffix_max_tokens = int(args.constrained_suffix_max_tokens)
            cfg.constrained_eval.temperature = float(args.constrained_temperature)
            cfg.constrained_eval.k_values = args.constrained_k_values

    if profile == "grpo" and args.disable_external:
        section.external_enabled = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc greedy and sampled pass@k checkpoint evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or HF model id.")
    parser.add_argument("--config", default="conf/posttrain_grpo.yaml", help="Path to an SFT or GRPO config YAML.")
    parser.add_argument("--profile", choices=["auto", "sft", "grpo"], default="auto")
    parser.add_argument("--output", default=None, help="Metrics JSON output path.")
    parser.add_argument("--samples-output", default=None, help="Optional JSONL output path for logged generations.")
    parser.add_argument("--collect-samples", type=int, default=8)
    parser.add_argument("--backend", choices=["auto", "hf", "vllm"], default=None)
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--samples-per-step", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--k-values", type=_parse_k_values, default=None, help="Comma-separated pass@k values.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--disable-external", action="store_true", help="Skip external benchmarks for GRPO configs.")
    parser.add_argument("--constrained-enabled", action="store_true", help="Also run line-level constrained proof pass@k.")
    parser.add_argument("--constrained-samples-per-step", type=int, default=4)
    parser.add_argument("--constrained-num-generations", type=int, default=8)
    parser.add_argument("--constrained-candidates-per-line", type=int, default=8)
    parser.add_argument("--constrained-k-values", type=_parse_k_values, default=[1, 2, 4, 8])
    parser.add_argument("--constrained-max-lines", type=int, default=32)
    parser.add_argument("--constrained-max-line-tokens", type=int, default=48)
    parser.add_argument("--constrained-suffix-max-tokens", type=int, default=256)
    parser.add_argument("--constrained-temperature", type=float, default=1.0)
    parser.add_argument("--wandb-project", default=None, help="Optional W&B project for metric upload.")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    profile = _detect_profile(cfg) if args.profile == "auto" else args.profile
    _apply_eval_overrides(cfg, profile, args)

    task_cfg = task_config_from_cfg(cfg)
    eval_cfg = eval_loop_config_from_grpo(cfg) if profile == "grpo" else eval_loop_config_from_sft(cfg)

    start = time.perf_counter()
    metrics, samples = UnifiedEvaluator().evaluate_checkpoint(
        args.checkpoint,
        collect_samples=max(0, int(args.collect_samples)),
        task_cfg=task_cfg,
        eval_cfg=eval_cfg,
    )
    elapsed = time.perf_counter() - start
    metrics["posthoc/elapsed_seconds"] = float(elapsed)
    metrics["posthoc/prompts"] = float(
        (int(eval_cfg.synthetic_step_max) - int(eval_cfg.synthetic_step_min) + 1) * int(eval_cfg.synthetic_samples_per_step)
    )
    metrics["posthoc/sampled_generations_per_prompt"] = float(eval_cfg.sampled_num_generations)

    payload = {"checkpoint": args.checkpoint, "profile": profile, "elapsed_seconds": elapsed, "metrics": metrics}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.samples_output:
        out = Path(args.samples_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.wandb_project:
        import wandb

        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name or f"passk_{Path(args.checkpoint).name}",
            "group": args.wandb_group,
            "job_type": "posthoc_passk_eval",
            "config": {
                "checkpoint": args.checkpoint,
                "profile": profile,
                "config_path": args.config,
                "step_min": eval_cfg.synthetic_step_min,
                "step_max": eval_cfg.synthetic_step_max,
                "samples_per_step": eval_cfg.synthetic_samples_per_step,
                "sampled_num_generations": eval_cfg.sampled_num_generations,
                "sampled_k_values": eval_cfg.sampled_k_values,
                "sampled_temperature": eval_cfg.sampled_temperature,
                "max_new_tokens": eval_cfg.max_new_tokens,
                "constrained_enabled": eval_cfg.constrained_enabled,
                "constrained_num_generations": eval_cfg.constrained_num_generations,
                "constrained_candidates_per_line": eval_cfg.constrained_candidates_per_line,
            },
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        run = wandb.init(**init_kwargs)
        wandb.log(metrics, step=0)
        if args.output:
            wandb.save(args.output, policy="now")
        run.finish()


if __name__ == "__main__":
    main()
