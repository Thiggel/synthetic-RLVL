#!/usr/bin/env python3
"""Fail unless the active Transformers environment resolves Qwen2 RoPE correctly."""

from __future__ import annotations

import argparse
import json

from transformers import AutoConfig


def resolved_rope_theta(checkpoint: str) -> float:
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=False)
    return float(config.rope_theta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected", type=float, default=1_000_000.0)
    args = parser.parse_args()
    actual = resolved_rope_theta(args.checkpoint)
    if actual != args.expected:
        raise SystemExit(
            f"Qwen2 RoPE compatibility failure: resolved rope_theta={actual}, expected {args.expected}"
        )
    print(json.dumps({"checkpoint": args.checkpoint, "rope_theta": actual}, sort_keys=True))


if __name__ == "__main__":
    main()
