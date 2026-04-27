#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_adapter_dir(checkpoint: Path) -> Path:
    if (checkpoint / "lora_adapter" / "adapter_config.json").is_file():
        return checkpoint / "lora_adapter"
    if (checkpoint / "adapter_config.json").is_file():
        return checkpoint
    raise FileNotFoundError(f"No LoRA adapter found under {checkpoint}")


def _resolve_tokenizer_dir(checkpoint: Path, adapter_dir: Path, base_model: str) -> str:
    actor_hf = checkpoint / "huggingface"
    if actor_hf.is_dir():
        return str(actor_hf)
    if checkpoint != adapter_dir and checkpoint.is_dir():
        return str(checkpoint)
    return base_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA checkpoint into a standalone HF model directory.")
    parser.add_argument("--checkpoint", required=True, help="Actor checkpoint or adapter directory.")
    parser.add_argument("--output", required=True, help="Output directory for merged HF model.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    adapter_dir = _resolve_adapter_dir(checkpoint)
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = adapter_cfg.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError(f"{adapter_dir}/adapter_config.json is missing base_model_name_or_path")
    base_model = base_model.strip()

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"[merge] base={base_model}", flush=True)
    print(f"[merge] adapter={adapter_dir}", flush=True)
    print(f"[merge] output={output}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output), safe_serialization=True, max_shard_size="5GB")

    tokenizer_dir = _resolve_tokenizer_dir(checkpoint, adapter_dir, base_model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.save_pretrained(str(output))
    print(f"[merge] wrote {output}", flush=True)


if __name__ == "__main__":
    main()

