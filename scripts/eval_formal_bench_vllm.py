#!/usr/bin/env python
"""Format-tagged downstream benchmarks: can the model reason fully in the formal system?

Every item of the fixed benchmark file (scripts/build_formal_bench_tagged.py) is
prompted as "<formal>\\n{prompt}" and generated greedily with the same vLLM loop
as the in-domain eval (eval_formal_vllm.generate). The lm-eval suite
(gruenau_formal_mix_bench_2026-09-26.slurm) measures transfer without the tag;
this script measures what happens when the model is asked to use the system.

Per item (the proof is the text between "<proof>\\n" and "</proof>"):
  has_proof     the output contains a <proof> block
  grammatical   rlvl.check(prompt, proof, strict=False) has no fatal parse error
  valid         rlvl.check(prompt, proof, strict=True)["ok"]: every step checks,
                every `given` quote occurs in the prompt, and the `ans` line
                answers the proof's own goal (no reference answer is passed)
  grounded      the proof has >= 1 `given` line and no quote error; the only
                faithfulness signal available without gold formalizations
  uses_know     the proof cites background knowledge (`know`) lines
  correct       the final answer matches the reference: the Answer: line, else
                the proof's `ans` value, else <answer>/\\boxed{}; yes/no is
                matched against true/false, letters with or without
                parentheses, numbers numerically; span items (multi-hop) score
                token F1 of the Answer: line (as in LongBench) and count as
                correct at F1 >= 0.5
  ans_correct   the proof's own `ans` value equals the reference (only possible
                when system_answerable: yes/no or number references)
  valid_correct valid and correct
  in_system     valid and ans_correct: a checked proof that itself derives the
                right answer (the "fully in the system" rate)
  valid_wrong   valid but not correct: a checked proof of the wrong answer
                means the premises were formalized unfaithfully (the checker
                cannot see this); on ProofWriter/FOLIO "unknown" items every
                valid yes/no proof is of this kind

summary.json aggregates per bench, per group and overall, each over all items
and over the system_answerable subset.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lm_eval_tasks" / "synthrlvl_ood"))

import rlvl  # noqa: E402  (PYTHONPATH must include RLVL-next/rlvl/python and RLVL-next/gen)
from eval_formal_vllm import ANSWER_TAG_RE, BOXED_RE, extract, generate  # noqa: E402
from utils import normalize_answer, qa_f1_score  # noqa: E402

YN = {"yes": "yes", "no": "no", "true": "yes", "false": "no", "valid": "yes", "invalid": "no"}
METRICS = ["has_proof", "grammatical", "valid", "grounded", "uses_know", "leak", "correct", "ans_correct",
           "valid_correct", "in_system", "valid_wrong"]


def _clean(a: str) -> str:
    a = a.strip().replace("**", "").replace("`", "")
    a = re.sub(r"(?<=\d),(?=\d{3}\b)", "", a)
    return a.rstrip(" .!")


def as_number(a: str):
    m = re.search(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", a)
    if not m:
        return None
    try:
        return Fraction(m.group(0).replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


def as_letter(a: str):
    m = re.match(r"^(?:the answer is|answer|option)?\s*[:\s]*\(?([A-Z])\)?(?:$|[\s.,:)])", a.strip(), re.I)
    return m.group(1).upper() if m else None


def match(cand, rec) -> float:
    if cand is None:
        return 0.0
    c, t, g = _clean(str(cand)), rec["answer_type"], rec["gold"]
    if t == "span":
        return max((qa_f1_score(c, x) for x in g), default=0.0)
    if t == "yesno":
        first = re.split(r"[\s,.;:!]+", c.lower(), maxsplit=1)[0]
        return float(YN.get(first, first) == g)
    if t == "letter":
        return float(as_letter(c) == g)
    if t == "number":
        n = as_number(c)
        return float(n is not None and n == Fraction(g))
    return float(normalize_answer(c) == normalize_answer(g))


def score(rec: dict, generation: str) -> dict:
    ex = extract(generation)
    proof = ex["proof"]
    row = {k: False for k in METRICS}
    row.update(has_proof=proof is not None, proof_closed=ex["proof_closed"], pred_answer=ex["answer"],
               sys_answer=None, error=None, n_givens=0)
    if proof is not None:
        loose = rlvl.check(rec["prompt"], proof, strict=False)
        strict = rlvl.check(rec["prompt"], proof, strict=True)
        trust = strict.get("trust") or {}
        quote_err = any((ln.get("code") == "quote") for ln in strict.get("lines") or [])
        row.update(
            grammatical=(loose.get("fatal") or {}).get("code") != "parse",
            valid=bool(strict["ok"]),
            sys_answer=strict.get("answer"),
            n_givens=trust.get("given", 0),
            grounded=trust.get("given", 0) > 0 and not quote_err,
            uses_know=trust.get("know", 0) > 0,
            leak=bool(strict.get("leak")),
            error=strict.get("first_error") or strict.get("fatal"),
            n_steps=strict.get("n_steps"),
        )
    cand = ex["answer"]
    if cand is None and row["sys_answer"] is not None:
        cand = str(row["sys_answer"])
    if cand is None:
        tags, boxed = ANSWER_TAG_RE.findall(generation), BOXED_RE.findall(generation)
        cand = tags[-1] if tags else (boxed[-1] if boxed else None)
    s = match(cand, rec)
    row["f1" if rec["answer_type"] == "span" else "score"] = s
    row["correct"] = s >= 0.5
    row["ans_correct"] = bool(rec["system_answerable"] and row["sys_answer"] is not None
                              and match(str(row["sys_answer"]), rec) == 1.0)
    row["valid_correct"] = row["valid"] and row["correct"]
    row["in_system"] = row["valid"] and row["ans_correct"]
    row["valid_wrong"] = row["valid"] and not row["correct"]
    return row


def aggregate(rows: list[dict]) -> dict:
    def agg(rs):
        if not rs:
            return {"n": 0}
        out = {"n": len(rs)}
        for k in METRICS:
            out[k] = sum(bool(r[k]) for r in rs) / len(rs)
        f1 = [r["f1"] for r in rs if "f1" in r]
        if f1:
            out["f1"] = sum(f1) / len(f1)
        return out

    def two(rs):
        return {"all": agg(rs), "answerable": agg([r for r in rs if r["system_answerable"]])}

    by_bench, by_group = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by_bench[r["bench"]].append(r)
        by_group[r["group"]].append(r)
        if r["bench"].startswith("gpqa"):
            by_bench["gpqa_diamond"].append(r)
        if r["bench"].startswith("pw_d"):
            by_bench["pw_all"].append(r)
    return {"overall": two(rows), "per_group": {g: two(v) for g, v in sorted(by_group.items())},
            "per_bench": {b: two(v) for b, v in sorted(by_bench.items())}}


def fit_prompts(records, args):
    """Cut the middle of prompts that would not leave room for generation (multi-hop contexts)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    budget = args.max_model_len - args.max_new_tokens - 256
    n_cut = 0
    for rec in records:
        ids = tok(rec["prompt"], add_special_tokens=False)["input_ids"]
        if len(ids) > budget:
            half = budget // 2
            rec["prompt"] = tok.decode(ids[:half]) + "\n...\n" + tok.decode(ids[-half:])
            rec["truncated"] = True
            n_cut += 1
    print(f"truncated {n_cut} prompts to {budget} tokens", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-jsonl", default="/vol/tmp2/laitenbf/rlvl_data/datasets/formal_bench_tagged_20260926/test.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-bench-limit", type=int, default=None, help="first N items of every bench (smoke)")
    ap.add_argument("--benches", default=None, help="comma-separated bench or group names to keep")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-tool-calls", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--no-mm", action="store_true", help="disable image/video inputs (VLM checkpoints)")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.test_jsonl)]
    if args.benches:
        keep = set(args.benches.split(","))
        records = [r for r in records if r["bench"] in keep or r["group"] in keep]
    if args.per_bench_limit:
        seen = collections.Counter()
        records = [r for r in records if (seen.update([r["bench"]]) or seen[r["bench"]] <= args.per_bench_limit)]
    fit_prompts(records, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    states = generate(records, args)
    elapsed = time.time() - t0
    flat = []
    with open(out_dir / "generations.jsonl", "w") as f:
        for rec, s in zip(records, states):
            r = {k: rec[k] for k in ("bench", "group", "id", "gold", "answer_type", "gold_label", "system_answerable")}
            r.update(score(rec, s["text"]), generation=s["text"], gen_tokens=s["gen_tokens"],
                     finish_reason=s["finish"], truncated=rec.get("truncated", False))
            flat.append(r)
            f.write(json.dumps(r, default=str) + "\n")
    summary = {"model": args.model, "test_jsonl": args.test_jsonl, "n": len(flat), "elapsed_s": elapsed,
               "max_new_tokens": args.max_new_tokens, "per_bench_limit": args.per_bench_limit, **aggregate(flat)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"n={len(flat)} elapsed={elapsed:.0f}s")
    hdr = "  ".join(f"{k[:9]:>9s}" for k in METRICS)
    print(f"{'bench':42s} {'n':>5s}  {hdr}")
    for b, v in summary["per_bench"].items():
        a = v["all"]
        print(f"{b:42s} {a['n']:5d}  " + "  ".join(f"{a[k]:9.3f}" for k in METRICS))


if __name__ == "__main__":
    main()
