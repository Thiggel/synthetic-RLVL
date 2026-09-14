#!/usr/bin/env python3
"""Sampled ProofWriter with checker selection: pass@k, maj@k and parse@k.

The prompt is the midtraining DOCUMENT format, not a chat chain-of-thought
request: only the document format makes these models emit a <proof> block,
and a derivation has to exist in a fixed shape before a checker can accept
it. A first version used the chain-of-thought prompt and measured validity
near zero, which was the prompt's fault, not the model's.

parse@k answers with the first of k samples whose written derivation the
forward-chaining checker accepts against the item's own theory. No gold is
used in the selection, so this measures what a deployed system could do.
Prompts use the derive-then-answer form, since a derivation has to exist
before it can be checked.
"""
import argparse, json, os, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lm_eval_tasks", "synthrlvl_ood"))
from checkers import check_proofwriter  # noqa: E402
import utils  # noqa: E402

LAB = re.compile(r"answer\s*[:\-]?\s*\**\s*(true|false|unknown)", re.I)
ANY = re.compile(r"\b(true|false|unknown)\b", re.I)


def answer_of(t):
    m = LAB.findall(t)
    if m:
        return m[-1].lower()
    m = ANY.findall(t)
    return m[-1].lower() if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", nargs="+", type=int, default=[2, 3, 5])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()
    root = os.environ.get("PW_ROOT", "/home/vault/c107fa/c107fa12/synthetic-RLVL/datasets/graded_deduction_eval_20260826")
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.checkpoint, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.85")), trust_remote_code=True)
    tok = llm.get_tokenizer()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for d in a.depths:
        items = [json.loads(l) for l in open(f"{root}/proofwriter_owa_d{d}.jsonl")][: a.limit]
        def doc_prompt(it):
            sents = re.split(r"(?<=\.)\s+", it["context"].strip())
            numbered = "\n".join(f"{i + 1}. {l.strip()}" for i, l in enumerate(sents) if l.strip())
            q = "Is the following claim true, false, or unknown: " + it["question"].strip()
            return f"<question>\n{numbered}\n{q}\n</question>\n\n"
        prompts = [doc_prompt(it) for it in items]
        sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95,
                            max_tokens=a.max_tokens, seed=a.seed, stop=["</answer>"])
        gen = llm.generate(prompts, sp)
        rows, agg = [], Counter()
        for it, g in zip(items, gen):
            texts = [o.text for o in g.outputs]
            ans = [answer_of(t) for t in texts]
            val = [check_proofwriter(it["context"], t)["all_valid"] > 0 for t in texts]
            gold = it["answer"].lower()
            rows.append(dict(doc=it, answers=ans, valid=val, gold=gold,
                             texts=[t[:2000] for t in texts[:4]]))
            for k in (1, 4, 16):
                agg[f"pass@{k}"] += any(x == gold for x in ans[:k])
                c = Counter(x for x in ans[:k] if x)
                agg[f"maj@{k}"] += bool(c) and c.most_common(1)[0][0] == gold
                first = next((x for v, x in zip(val[:k], ans[:k]) if v), "")
                agg[f"parse@{k}"] += first == gold
            agg["valid_frac"] += sum(val) / len(val)
            agg["acc_given_valid_num"] += sum(v and x == gold for v, x in zip(val, ans))
            agg["acc_given_valid_den"] += sum(val)
        n = len(items)
        s = {k: (v / n if not k.startswith("acc_given") else v) for k, v in agg.items()}
        s["acc_given_valid"] = (agg["acc_given_valid_num"] / agg["acc_given_valid_den"]) if agg["acc_given_valid_den"] else float("nan")
        summary[f"d{d}"] = s
        with open(out / f"samples_d{d}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"d{d}: " + " ".join(f"{k}={s[k]:.3f}" for k in ("pass@1", "pass@16", "maj@16", "parse@1", "parse@16", "valid_frac", "acc_given_valid")), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
