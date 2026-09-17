#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and write a plain checkpoint.

The evaluation jobs load a checkpoint directory with vLLM and expect ordinary
weights, so a LoRA run is merged once on the CPU after training. The merged
directory carries the tokenizer and, when the base shipped none, the chat
template the run trained with, so evaluation sees the same format.
"""
import argparse
import json
import pathlib
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()

    cfg = json.loads((a.adapter / "adapter_config.json").read_text())
    print("adapter r=%s alpha=%s targets=%s" % (cfg.get("r"), cfg.get("lora_alpha"), cfg.get("target_modules")))
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(a.adapter))
    model = model.merge_and_unload()
    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(str(a.adapter))
    tok.save_pretrained(str(tmp))
    for extra in ("chat_template.jinja",):
        src = a.adapter / extra
        if src.exists():
            shutil.copy(src, tmp / extra)
    (tmp / "lora_merge.json").write_text(json.dumps({"base": a.base, "adapter": str(a.adapter), **cfg}, indent=2, default=str))
    if a.out.exists():
        shutil.rmtree(a.out)
    tmp.rename(a.out)
    print("merged into", a.out)


if __name__ == "__main__":
    main()
