#!/usr/bin/env python3
"""Full-set evaluation of GSM8K, ProofWriter and FOLIO in the midtraining
document format, greedy, with answer scoring. Stores every generation so the
derivations can be read and, for ProofWriter, checked.
"""
import argparse
import json
import os
import re
from pathlib import Path


def wrap(lines, question):
    numbered = "\n".join(f"{i + 1}. {l.strip()}" for i, l in enumerate(lines) if l.strip())
    return f"<question>\n{numbered}\n{question.strip()}\n</question>\n\n"


ANS = re.compile(r"<answer>\s*(.*?)\s*(?:</answer>|$)", re.DOTALL)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm_pred(text):
    m = ANS.search(text)
    src = m.group(1) if m else text[-300:]
    nums = NUM.findall(src.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def label_pred(text, labels):
    m = ANS.search(text)
    src = (m.group(1) if m else text[-200:]).lower()
    for l in labels:
        if re.search(r"\b%s\b" % l, src):
            return l
    if re.search(r"\byes\b", src):
        return "true"
    if re.search(r"\bno\b", src):
        return "false"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    D = os.path.expanduser("~/rlvl_data/datasets")
    items = []
    from datasets import load_dataset
    for r in load_dataset("gsm8k", "main", split="test"):
        sents = re.split(r"(?<=[.?!])\s+", r["question"].strip())
        items.append(dict(source="gsm8k", prompt=wrap(sents[:-1], sents[-1]), gold=r["answer"].split("####")[-1].strip().replace(",", "")))
    for d in (0, 1, 2, 3, 5):
        for line in open(f"{D}/graded_deduction_eval_20260826/proofwriter_owa_d{d}.jsonl"):
            r = json.loads(line)
            sents = re.split(r"(?<=\.)\s+", r["context"].strip())
            q = f"Is the following claim true, false, or unknown: {r['question'].strip()}"
            items.append(dict(source=f"proofwriter_d{d}", prompt=wrap(sents, q), gold=r["answer"].lower(), question=r["question"], context=r["context"]))
    for line in open(f"{D}/folio_eval_20260910/folio_validation.jsonl"):
        r = json.loads(line)
        q = f"Is the following claim true, false, or uncertain: {r['question'].strip()}"
        items.append(dict(source="folio", prompt=wrap(r["context"].strip().splitlines(), q), gold=r["answer"].lower()))
    if a.limit:
        items = [it for src in sorted({i["source"] for i in items}) for it in [i for i in items if i["source"] == src][: a.limit]]
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.checkpoint, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.6")),
              swap_space=2, enforce_eager=True, trust_remote_code=True)
    outs = llm.generate([it["prompt"] for it in items], SamplingParams(temperature=0.0, max_tokens=3000, stop=["</answer>"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    agg = {}
    with open(Path(a.out) / "samples.jsonl", "w") as f:
        for it, o in zip(items, outs):
            t = o.outputs[0].text
            if it["source"] == "gsm8k":
                pred = gsm_pred(t)
                ok = pred == it["gold"]
            else:
                labels = ["true", "false", "unknown", "uncertain"]
                pred = label_pred(t, labels)
                ok = pred == it["gold"]
            it.update(response=t, pred=pred, correct=float(ok), has_derivation=float("<proof>" in t), has_answer=float("<answer>" in t))
            s = agg.setdefault(it["source"], dict(n=0, correct=0.0, deriv=0.0, ans=0.0))
            s["n"] += 1; s["correct"] += ok; s["deriv"] += it["has_derivation"]; s["ans"] += it["has_answer"]
            f.write(json.dumps(it) + "\n")
    summ = {k: dict(n=v["n"], acc=v["correct"] / v["n"], derivation=v["deriv"] / v["n"], answer_tag=v["ans"] / v["n"]) for k, v in agg.items()}
    (Path(a.out) / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
