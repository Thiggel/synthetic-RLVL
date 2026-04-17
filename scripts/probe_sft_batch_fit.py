from __future__ import annotations

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

MODEL = "allenai/Olmo-3-1025-7B"
SEQ_LEN = 2048
BSZ_VALUES = [1, 4, 8, 16, 32, 64]
TARGET_GLOBAL_BSZ = 64


def measure_for_bsz(model, bsz: int, seq_len: int) -> dict:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    inp = torch.randint(10, 1000, (bsz, seq_len), device=device)
    attn = torch.ones_like(inp)
    labels = inp.clone()

    ok = True
    err = None
    t0 = time.time()
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            loss = out.loss
        loss.backward()
    except RuntimeError as exc:
        ok = False
        err = str(exc)
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    return {
        "bsz": bsz,
        "ok": ok,
        "seconds": round(dt, 3),
        "peak_gb": round(peak_gb, 3),
        "error": err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    device = torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    results = []
    for bsz in BSZ_VALUES:
        res = measure_for_bsz(model, bsz=bsz, seq_len=int(args.seq_len))
        results.append(res)
        print(json.dumps(res))
        if not res["ok"]:
            torch.cuda.empty_cache()

    feasible = [r["bsz"] for r in results if r["ok"]]
    max_per_device = max(feasible) if feasible else 1
    rec = []
    for bsz in BSZ_VALUES:
        per_device = min(max_per_device, bsz)
        grad_accum = (bsz + per_device - 1) // per_device
        rec.append({"global_bsz": bsz, "per_device": per_device, "grad_accum": grad_accum})

    out = {
        "model": MODEL,
        "seq_len": int(args.seq_len),
        "results": results,
        "max_feasible_per_device": max_per_device,
        "recommended": rec,
    }
    print("SUMMARY")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
