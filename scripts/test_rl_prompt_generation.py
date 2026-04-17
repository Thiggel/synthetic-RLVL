#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT = """<question>
1. Uma is dry.
2. All things that are dry are swift.
3. Uma is swift.
What tempo does Uma have?
</question>

"""


def _load_model_and_tokenizer(checkpoint_path: str, device: str):
    ckpt = Path(checkpoint_path)
    adapter_cfg = ckpt / "adapter_config.json"
    adapter_weights = ckpt / "adapter_model.safetensors"

    if adapter_cfg.exists() and adapter_weights.exists():
        payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base = str(payload["base_model_name_or_path"]).strip()
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt))
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug generation on an exact RL prompt.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (or LoRA adapter checkpoint).")
    parser.add_argument("--prompt-file", default=None, help="Optional file containing the exact prompt text.")
    parser.add_argument("--prompt", default=None, help="Optional inline prompt string.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = DEFAULT_PROMPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = _load_model_and_tokenizer(args.checkpoint, device)

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.temperature > 0.0),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    completion_ids = out[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    print("=== PROMPT (exact) ===")
    print(prompt)
    print("=== COMPLETION (exact decoded) ===")
    print(completion)
    print("=== COMPLETION (repr) ===")
    print(repr(completion))


if __name__ == "__main__":
    main()

