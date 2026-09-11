#!/usr/bin/env python3
"""Qualitative probe: does the native <question> document format elicit a
derivation on material that is not BranchProof? Wraps GSM8K, ProofWriter and
FOLIO items in the exact midtraining wrapper (numbered lines, a final
question, </question>) and stores the greedy continuation for reading.
"""
import argparse
import json
import os
import re
from pathlib import Path


def wrap(lines, question):
    numbered = "\n".join(f"{i + 1}. {l.strip()}" for i, l in enumerate(lines) if l.strip())
    return f"<question>\n{numbered}\n{question.strip()}\n</question>\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-each", type=int, default=6)
    a = ap.parse_args()
    D = ("/vol/tmp2/laitenbf/rlvl_data/datasets")
    items = []
    from datasets import load_dataset
    g = load_dataset("gsm8k", "main", split="test")
    for r in list(g)[: a.n_each]:
        sents = re.split(r"(?<=[.?!])\s+", r["question"].strip())
        items.append(dict(source="gsm8k", prompt=wrap(sents[:-1], sents[-1]), gold=r["answer"].split("####")[-1].strip()))
    for line in list(open(f"{D}/graded_deduction_eval_20260826/proofwriter_owa_d3.jsonl"))[: a.n_each]:
        r = json.loads(line)
        sents = re.split(r"(?<=\.)\s+", r["context"].strip())
        items.append(dict(source="proofwriter_d3", prompt=wrap(sents, f"Is it true that {r['question'].rstrip('.')}?"), gold=r["answer"]))
    for line in list(open(f"{D}/folio_eval_20260910/folio_validation.jsonl"))[: a.n_each]:
        r = json.loads(line)
        items.append(dict(source="folio", prompt=wrap(r["context"].strip().splitlines(), f"Is it true that {r['question'].rstrip('.')}?"), gold=r["answer"]))
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.checkpoint, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.6, trust_remote_code=True)
    outs = llm.generate([it["prompt"] for it in items], SamplingParams(temperature=0.0, max_tokens=3000, stop=["</answer>"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for it, o in zip(items, outs):
            it["response"] = o.outputs[0].text
            f.write(json.dumps(it) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
