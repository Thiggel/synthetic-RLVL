#!/usr/bin/env python
"""Materialize the downstream benchmarks as one fixed jsonl for the format-tagged eval.

scripts/eval_formal_bench_vllm.py prompts every item as "<formal>\\n{prompt}", the
format the formal-mixture checkpoints were trained on, and scores the proof with
rlvl.check. The builder fixes the items once so that every checkpoint sees the
same prompts. Run it on a login node with the offline HF cache:

  HF_HOME=/vol/tmp2/laitenbf HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
    .venv_rlvl_vllm/bin/python scripts/build_formal_bench_tagged.py

Fields: bench, group, id, prompt, gold, answer_type, gold_label, system_answerable.
  answer_type        yesno | number | letter | text | span (span = multi-hop, F1)
  gold_label         the source label (true/false/unknown, A, 42, ...)
  system_answerable  the formal system can state the gold answer as `ans`
                     (yes/no or a number), i.e. a fully in-system proof can be
                     correct; letters, free text and "unknown" cannot.

Prompts mirror the training prompts, which state the facts and end in a question:
  ProofWriter d0/1/2/3/5 (no d4 file), FOLIO: "<context> Is it true that <claim>?";
    gold true -> yes, false -> no; unknown/uncertain kept, not system-answerable.
  BBH: all 27 tasks, zero-shot, the first --bbh-cap items per task; binary targets
    (True/False, Yes/No, valid/invalid, yes/no) become yesno; (X) targets letter.
  GPQA-Diamond (options in the question, letter; is_quant as in the paper subset),
  ARC-Challenge, LogiQA, MMLU (seeded --mmlu-n over all subjects): the options
    listed as "(A) ..." and a one-line answer instruction.
  GSM8K (first --gsm8k-n): number.
  LongBench HotpotQA / 2Wiki / MuSiQue: passages + question, span, F1.
HellaSwag, PIQA, WinoGrande (completion ranking) and HumanEval/MBPP (code) have
no natural in-system formulation and are left out.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from formal_mix_bench_table import is_quant  # noqa: E402

DATA = Path("/vol/tmp2/laitenbf/rlvl_data/datasets")
PW = DATA / "graded_deduction_eval_20260826"
FOLIO = DATA / "folio_eval_20260910/folio_validation.jsonl"
GPQA = DATA / "gpqa_diamond_20260910/gpqa_diamond.jsonl"
HF_DATASETS = Path("/vol/tmp2/laitenbf/datasets")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MC_INSTR = "Answer with the letter of the correct option."
BIN = {"true": "yes", "false": "no", "yes": "yes", "no": "no", "valid": "yes", "invalid": "no"}


def bbh_binary_prompt(task: str, prompt: str) -> str:
    """Turn a BBH binary item into a yes/no question (web of lies, navigate, ... already are)."""
    if task == "boolean_expressions":
        return f"Is the expression {prompt.removesuffix(' is').strip()} True?"
    if task == "formal_fallacies":
        return prompt[: prompt.index("Is the argument, given")] + \
            "Is the argument, given the explicitly stated premises, deductively valid?"
    return re.sub(r"\s*Options:\s*- Yes\s*- No\s*$", "", prompt)
DETERMINERS = ("The ", "A ", "An ", "All ", "Some ", "No ", "Every ", "Each ")


def claim_question(context: str, claim: str) -> str:
    c = claim.strip().rstrip(".")
    if c.startswith(DETERMINERS):
        c = c[0].lower() + c[1:]
    return f"{' '.join(context.split())} Is it true that {c}?"


def jl(path):
    return [json.loads(l) for l in open(path)]


def tf(label: str):
    lab = label.strip().lower()
    return (BIN[lab], "yesno", True) if lab in ("true", "false") else (lab, "yesno", False)


def mc_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))
    return f"{question.strip()}\n{opts}\n{MC_INSTR}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA / "formal_bench_tagged_20260926/test.jsonl")
    ap.add_argument("--bbh-cap", type=int, default=100)
    ap.add_argument("--mmlu-n", type=int, default=1000)
    ap.add_argument("--gsm8k-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    from datasets import load_dataset

    def configs(repo: str) -> list[str]:  # offline mode cannot list configs; read the cache
        d = HF_DATASETS / repo.replace("/", "___")
        return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name not in ("default", "all", "auxiliary_train"))

    rows = []

    def add(bench, group, i, prompt, gold, atype, label, answerable):
        rows.append({"bench": bench, "group": group, "id": f"{bench}/{i}", "prompt": prompt, "gold": gold,
                     "answer_type": atype, "gold_label": label, "system_answerable": answerable})

    for d in (0, 1, 2, 3, 5):
        for i, r in enumerate(jl(PW / f"proofwriter_owa_d{d}.jsonl")):
            g, t, a = tf(r["answer"])
            add(f"pw_d{d}", "deduction", i, claim_question(r["context"], r["question"]), g, t, r["answer"], a)
    for i, r in enumerate(jl(FOLIO)):
        g, t, a = tf(r["answer"])
        add("folio", "deduction", i, claim_question(r["context"], r["question"]), g, t, r["answer"], a)

    for task in sorted(configs("SaylorTwift/bbh")):
        ds = load_dataset("SaylorTwift/bbh", task, split="test")
        for i, r in enumerate(ds.select(range(min(args.bbh_cap, len(ds))))):
            tgt = r["target"].strip()
            prompt = r["input"].strip()
            if tgt.lower() in BIN:
                g, t = BIN[tgt.lower()], "yesno"
                prompt = bbh_binary_prompt(task, prompt)
            elif re.fullmatch(r"\([A-Z]\)", tgt):
                g, t = tgt[1], "letter"
                prompt += f"\n{MC_INSTR}"
            elif re.fullmatch(r"-?\d+", tgt):
                g, t = tgt, "number"
            else:
                g, t = tgt, "text"
            add(f"bbh_{task}", "bbh", i, prompt, g, t, tgt, t in ("yesno", "number"))

    for i, r in enumerate(jl(GPQA)):
        q = r["question"].strip() + f"\n{MC_INSTR}"
        add("gpqa_quant" if is_quant(r["question"]) else "gpqa_rest", "gpqa", i, q, r["answer"].strip(), "letter",
            r["answer"].strip(), False)

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for i, r in enumerate(arc):
        labels = r["choices"]["label"]
        key = r["answerKey"]
        idx = labels.index(key)
        add("arc_challenge", "standard", i, mc_prompt(r["question"], r["choices"]["text"]), LETTERS[idx], "letter",
            key, False)

    lq = load_dataset("hails/agieval-logiqa-en", split="test")
    for i, r in enumerate(lq):
        q = r["query"]
        q = re.sub(r"\s*Answer Choices:.*$", "", q, flags=re.S).strip()
        q = re.sub(r"^Q:\s*", "", q)
        opts = [re.sub(r"^\([A-Z]\)\s*", "", c) for c in r["choices"]]
        g = LETTERS[r["gold"][0]]
        add("logiqa", "standard", i, mc_prompt(q.replace("Q: ", "\n"), opts), g, "letter", g, False)

    subjects = configs("cais/mmlu")
    pool = []
    for s in sorted(subjects):
        for i, r in enumerate(load_dataset("cais/mmlu", s, split="test")):
            pool.append((s, i, r))
    for s, i, r in sorted(random.Random(args.seed).sample(pool, args.mmlu_n), key=lambda t: (t[0], t[1])):
        g = LETTERS[r["answer"]]
        add("mmlu", "standard", f"{s}/{i}", mc_prompt(r["question"], r["choices"]), g, "letter", g, False)

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i, r in enumerate(gsm.select(range(args.gsm8k_n))):
        g = r["answer"].split("####")[-1].strip().replace(",", "")
        add("gsm8k", "standard", i, r["question"].strip(), g, "number", g, True)

    for task, name in (("hotpotqa", "hotpotqa"), ("2wikimqa", "2wikimqa"), ("musique", "musique")):
        for i, r in enumerate(load_dataset("Xnhyacinth/LongBench", task, split="test")):
            prompt = (f"Passages:\n{r['context'].strip()}\n\nAnswer the question based on the passages. "
                      f"Give only the answer, a few words.\nQuestion: {r['question'].strip()}")
            add(name, "multihop", i, prompt, r["answers"], "span", r["answers"][0], False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    import collections
    c = collections.Counter(r["bench"] for r in rows)
    print(f"{len(rows)} items -> {args.out}")
    for b, n in c.items():
        na = sum(r["system_answerable"] for r in rows if r["bench"] == b)
        print(f"  {b:45s} {n:5d}  system_answerable {na}")


if __name__ == "__main__":
    main()
