#!/usr/bin/env python3
"""Verify a converted Qwen2 checkpoint before deleting its Nanotron source."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_local_files(checkpoint: Path) -> dict[str, Any]:
    errors: list[str] = []
    required = ("config.json", "tokenizer_config.json")
    for name in required:
        if not (checkpoint / name).is_file():
            errors.append(f"missing required file: {name}")

    index_path = checkpoint / "model.safetensors.index.json"
    shards: set[str] = set()
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append("empty or invalid safetensors weight_map")
            else:
                shards = {str(name) for name in weight_map.values()}
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid safetensors index: {exc}")
    elif (checkpoint / "model.safetensors").is_file():
        shards = {"model.safetensors"}
    else:
        errors.append("missing model.safetensors or model.safetensors.index.json")

    for shard in sorted(shards):
        path = checkpoint / shard
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"missing or empty weight shard: {shard}")

    local_files = {
        str(path.relative_to(checkpoint))
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    return {
        "accepted": not errors,
        "errors": errors,
        "local_files": sorted(local_files),
        "weight_shards": sorted(shards),
    }


def verify_cuda_forward(checkpoint: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the converted 7B checkpoint load smoke")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda")
    encoded = tokenizer("The answer to two plus two is", return_tensors="pt")
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1, :]
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != model.config.vocab_size:
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("non-finite logits from converted checkpoint")
    return {
        "model_type": model.config.model_type,
        "vocab_size": int(model.config.vocab_size),
        "logits_shape": list(logits.shape),
        "logits_finite": True,
    }


def audit_remote_files(repo_id: str, local_files: set[str], token_env: str) -> dict[str, Any]:
    token = os.environ.get(token_env) or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGINGFACE_HUB_TOKEN"
    )
    if not token:
        raise RuntimeError(
            f"No Hugging Face token found in ${token_env}, $HF_TOKEN, or $HUGGINGFACE_HUB_TOKEN"
        )
    remote_files = set(HfApi(token=token).list_repo_files(repo_id=repo_id, repo_type="model"))
    missing = sorted(local_files - remote_files)
    if missing:
        raise RuntimeError("uploaded repository is missing files: " + ", ".join(missing))
    return {
        "remote_file_count": len(remote_files),
        "missing_remote_files": missing,
    }


def main() -> None:
    args = parse_args()
    local = audit_local_files(args.checkpoint)
    if not local["accepted"]:
        raise SystemExit("; ".join(local["errors"]))
    forward = verify_cuda_forward(args.checkpoint)
    remote = audit_remote_files(args.repo_id, set(local["local_files"]), args.token_env)
    report = {
        "accepted": True,
        "checkpoint": str(args.checkpoint),
        "repo_id": args.repo_id,
        "local": local,
        "forward": forward,
        "remote": remote,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
