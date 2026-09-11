#!/usr/bin/env python3
"""Sample k derivations per BranchProof item in the native document format.

For each depth file, prompts the model exactly as its midtraining documents
("<question>\n1. ...\nWhich state applies to cN?\n</question>\n\n"), draws n
samples at temperature T plus one greedy sample, stops at </answer>, and
stores everything as jsonl for `native_passk_analysis.py` (pass@k, maj@k,
parse@k = first checker-valid derivation's answer, and the checker's own
validity rate).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lm_eval_tasks", "synthrlvl_ood"))
import utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=("/vol/tmp2/laitenbf/rlvl_data/datasets/graded_deduction_eval_20260826"))
    ap.add_argument("--depths", nargs="+", type=int, default=[5, 10, 15, 20, 25])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.checkpoint, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, seed=a.seed, trust_remote_code=True)
    stop = ["</answer>"]
    sampled = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, stop=stop, seed=a.seed)
    greedy = SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens, stop=stop)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for d in a.depths:
        items = [json.loads(l) for l in open(f"{a.data_root}/branchproof_nl_d{d}.jsonl")]
        if a.limit:
            items = items[: a.limit]
        prompts = [utils.doc_to_text_deduction_bp_native(it) for it in items]
        gs = llm.generate(prompts, greedy)
        ss = llm.generate(prompts, sampled)
        with open(out / f"native_d{d}.jsonl", "w") as f:
            for it, g, s in zip(items, gs, ss):
                f.write(json.dumps({"doc": it, "depth": d, "greedy": g.outputs[0].text,
                                    "samples": [o.text for o in s.outputs],
                                    "sample_tokens": [len(o.token_ids) for o in s.outputs]}) + "\n")
        print(f"depth {d}: {len(items)} items x {a.n} samples written", flush=True)
    (out / "manifest.json").write_text(json.dumps(vars(a), indent=2))


if __name__ == "__main__":
    main()
