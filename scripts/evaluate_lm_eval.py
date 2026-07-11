#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


TASK_SUITES = {
    "reasoning_core": ["gsm8k", "fld_default", "logiqa"],
    "logic_core": ["fld_default", "fld_logical_formula_default", "logiqa"],
    "math_core": ["gsm8k"],
    "synthrlvl_ood": [
        "synthrlvl_gsm8k_tagged",
        "synthrlvl_longbench_hotpotqa_tagged",
        "synthrlvl_longbench_2wikimqa_tagged",
        "synthrlvl_longbench_musique_tagged",
    ],
    "synthrlvl_ood_cot_bare": [
        "synthrlvl_gsm8k_cot_bare",
        "synthrlvl_longbench_hotpotqa_cot_bare",
        "synthrlvl_longbench_2wikimqa_cot_bare",
        "synthrlvl_longbench_musique_cot_bare",
    ],
    "synthrlvl_ood_cot_prompted": [
        "synthrlvl_gsm8k_cot_prompted",
        "synthrlvl_longbench_hotpotqa_cot_prompted",
        "synthrlvl_longbench_2wikimqa_cot_prompted",
        "synthrlvl_longbench_musique_cot_prompted",
    ],
}


def _split_csv_or_space(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            if part.strip():
                out.append(part.strip())
    return out


def _expand_tasks(tasks: list[str], suites: list[str]) -> list[str]:
    expanded: list[str] = []
    for suite in suites:
        if suite not in TASK_SUITES:
            raise ValueError(f"Unknown task suite {suite!r}. Known suites: {sorted(TASK_SUITES)}")
        expanded.extend(TASK_SUITES[suite])
    expanded.extend(tasks)
    return list(dict.fromkeys(expanded))


def _kv_args(raw: list[str] | None) -> list[str]:
    args: list[str] = []
    for item in raw or []:
        if not item:
            continue
        args.append(str(item))
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Thin wrapper around lm-evaluation-harness for this project.")
    parser.add_argument("--checkpoint", required=True, help="HF model id or local merged checkpoint directory.")
    parser.add_argument("--model", default="vllm", choices=["hf", "vllm"], help="lm-eval model backend.")
    parser.add_argument("--tasks", nargs="*", default=None, help="lm-eval task names, comma- or space-separated.")
    parser.add_argument("--suite", action="append", default=[], choices=sorted(TASK_SUITES), help="Named task suite shortcut.")
    parser.add_argument("--output-path", required=True, help="lm-eval output directory or JSON path.")
    parser.add_argument("--batch-size", default="auto", help="lm-eval batch size.")
    parser.add_argument("--limit", default=None, help="Optional lm-eval --limit.")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--apply-chat-template", nargs="?", const="", default=None)
    parser.add_argument("--system-instruction", default=None)
    parser.add_argument("--gen-kwargs", nargs="*", default=None, help="Passed through to lm-eval --gen_kwargs.")
    parser.add_argument("--model-arg", action="append", default=[], help="Extra model arg key=value. Can be repeated.")
    parser.add_argument("--include-path", action="append", default=[], help="Additional lm-eval task directory. Can be repeated.")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--use-cache", default=None)
    parser.add_argument("--seed", default="0,1234,1234,1234")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--confirm-run-unsafe-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = _expand_tasks(_split_csv_or_space(args.tasks), args.suite)
    if not tasks:
        raise ValueError("Provide --tasks or --suite.")

    model_args = [f"pretrained={args.checkpoint}", f"dtype={args.dtype}", *args.model_arg]
    if args.model == "vllm":
        model_args.append(f"gpu_memory_utilization={args.gpu_memory_utilization}")

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        args.model,
        "--model_args",
        *model_args,
        "--tasks",
        *tasks,
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        str(args.output_path),
        "--seed",
        str(args.seed),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.num_fewshot is not None:
        cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.device:
        cmd += ["--device", str(args.device)]
    if args.apply_chat_template is not None:
        cmd.append("--apply_chat_template")
        if args.apply_chat_template:
            cmd.append(str(args.apply_chat_template))
    if args.system_instruction:
        cmd += ["--system_instruction", args.system_instruction]
    gen_kwargs = _kv_args(args.gen_kwargs)
    if gen_kwargs:
        cmd += ["--gen_kwargs", *gen_kwargs]
    for include_path in args.include_path:
        cmd += ["--include_path", include_path]
    if args.log_samples:
        cmd.append("--log_samples")
    if args.use_cache:
        cmd += ["--use_cache", args.use_cache]
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if args.confirm_run_unsafe_code:
        cmd.append("--confirm_run_unsafe_code")
    wandb_args: list[str] = []
    if args.wandb_project:
        wandb_args.append(f"project={args.wandb_project}")
    if args.wandb_run_name:
        wandb_args.append(f"name={args.wandb_run_name}")
    if args.wandb_group:
        wandb_args.append(f"group={args.wandb_group}")
    if wandb_args:
        cmd += ["--wandb_args", *wandb_args]

    metadata = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "tasks": tasks,
        "output_path": str(args.output_path),
        "cmd": cmd,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(args.output_path)).with_suffix(".command.json") if Path(str(args.output_path)).suffix else Path(args.output_path) / "command.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("[lm-eval-wrapper]", " ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
