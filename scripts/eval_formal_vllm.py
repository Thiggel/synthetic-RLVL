#!/usr/bin/env python
"""Evaluate formal-CoT proofs from a model on the held-out rlvlgen test split.

Generation (vLLM, greedy): the user turn is "<formal>\\n{prompt}", rendered
with the model's chat template through scripts/formal_chat_format.py, the same
module the trainer uses. vLLM stops on "<result>" and on <|im_end|>. On a
"<result>" stop the harness takes the call after "do " on the last line, appends
rlvlgen.toolsim.run(call, record["tool_docs"]) + "</result>\\n" and resumes, up to
--max-tool-calls calls and --max-new-tokens generated tokens in total. The
continuation prompt is re-encoded with the train-time tool-segment cut.

Parsing: the proof is the text between "<proof>\\n" and "</proof>" (to the end
if unclosed), the answer is the last "Answer:" line.

Metrics, per example and aggregated overall and per family:
  faithful   every line matching ^(\\S+) (.*) ; given "(.*)"$ has its quote inside
             one sentence's text and its formula, normalized with rlvl.normalize,
             equal to one of that same sentence's normalized forms; and at least
             one given exists.
             given_precision = faithful givens / givens (micro),
             given_recall    = gold-proof given formulas matched by a faithful
                               predicted given / gold given formulas (micro).
             faithful_on_given_refs restricts `faithful` to examples whose
             reference proof has a given (most `tools` proofs use only obs lines).
  grammatical  rlvl.check(prompt, proof, strict=False) has no fatal parse error.
  valid        rlvl.check(prompt, proof, expected=answer, strict=True, tools=...)["ok"].
  answer_acc   the Answer: line equals the reference answer (after strip).
  answer_acc_lenient  format-agnostic secondary readout, so conditions that never
             saw the formal format (X=0) are comparable: the candidate is the
             Answer: line, else the last <answer>...</answer>, else the last
             \boxed{...}; compared lowercased, without thousands commas and
             trailing punctuation, and for yes/no references on the first word.

--source gold|corrupt scores reference outputs instead of a model (evaluator
sanity checks; no vLLM needed).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rlvl  # noqa: E402  (PYTHONPATH must include RLVL-next/rlvl/python and RLVL-next/gen)
from rlvlgen import toolsim  # noqa: E402

GIVEN_RE = re.compile(r"^(\S+) (.*) ; given \"(.*)\"$")
ANSWER_RE = re.compile(r"(?m)^Answer:[ \t]*(.*?)[ \t]*$")
TAG_USER = "<formal>"
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _norm_ans(a: str) -> str:
    a = a.strip().lower().replace("**", "")
    a = re.sub(r"(?<=\d),(?=\d{3}\b)", "", a)
    return a.rstrip(" .!")


def lenient_correct(generation: str, strict_answer, gold) -> bool:
    cand = strict_answer
    if cand is None:
        tags = ANSWER_TAG_RE.findall(generation)
        boxed = BOXED_RE.findall(generation)
        cand = tags[-1] if tags else (boxed[-1] if boxed else None)
    if cand is None:
        return False
    c, g = _norm_ans(cand), _norm_ans(str(gold))
    if c == g:
        return True
    if g in ("yes", "no"):
        first = re.split(r"[\s,.;:!]+", c, maxsplit=1)[0]
        return first == g
    return False


# ---------------------------------------------------------------- parsing / metrics
@lru_cache(maxsize=None)
def norm_formula(f: str) -> str:
    try:
        line = rlvl.normalize(f"1 {f.strip()} ; given \"q\"\n")
        m = GIVEN_RE.match(line.rstrip("\n"))
        if m:
            return m.group(2)
    except Exception:
        pass
    return "\0unparsed:" + " ".join(f.split())


def extract(generation: str) -> dict:
    start = generation.find("<proof>\n")
    proof, closed = None, False
    if start >= 0:
        body = generation[start + len("<proof>\n"):]
        end = body.find("</proof>")
        closed = end >= 0
        proof = body[:end] if closed else body
    answers = ANSWER_RE.findall(generation)
    return {"proof": proof, "proof_closed": closed, "answer": answers[-1] if answers else None}


def givens(proof: str) -> list[tuple[str, str, str]]:
    out = []
    for line in proof.split("\n"):
        m = GIVEN_RE.match(line)
        if m:
            out.append(m.groups())
    return out


def faithful_given(formula: str, quote: str, sentences: list[dict]) -> bool:
    nf = norm_formula(formula)
    for s in sentences:
        if quote and quote in s["text"] and any(nf == norm_formula(f) for f in s["forms"]):
            return True
    return False


def score(rec: dict, generation: str) -> dict:
    ex = extract(generation)
    proof = ex["proof"]
    cfg = {"tools": rec["tools"]} if rec.get("tools") else {}
    gold_forms = {norm_formula(f) for _, f, _ in givens(rec["proof"])}
    row = {"has_proof": proof is not None, "proof_closed": ex["proof_closed"], "pred_answer": ex["answer"],
           "answer_correct": ex["answer"] is not None and ex["answer"].strip() == str(rec["answer"]).strip(),
           "answer_correct_lenient": lenient_correct(generation, ex["answer"], rec["answer"]),
           "n_gold_givens": len(gold_forms)}
    if proof is None:
        row.update(n_givens=0, n_faithful=0, faithful=False, gold_recalled=0, grammatical=False, valid=False,
                   error=None)
        return row
    gs = givens(proof)
    ok = [faithful_given(f, q, rec["sentences"]) for _, f, q in gs]
    faithful_forms = {norm_formula(f) for (_, f, _), o in zip(gs, ok) if o}
    loose = rlvl.check(rec["prompt"], proof, strict=False, **cfg)
    strict = rlvl.check(rec["prompt"], proof, expected=rec["answer"], strict=True, **cfg)
    fatal = loose.get("fatal") or {}
    row.update(
        n_givens=len(gs), n_faithful=sum(ok), faithful=bool(gs) and all(ok),
        gold_recalled=len(gold_forms & faithful_forms),
        grammatical=fatal.get("code") != "parse",
        valid=bool(strict["ok"]),
        error=strict.get("first_error"), loose_fatal=loose.get("fatal"),
        n_steps=strict.get("n_steps"),
    )
    return row


def aggregate(rows: list[dict]) -> dict:
    def agg(rs):
        n = len(rs)
        g = sum(r["n_givens"] for r in rs)
        gold = sum(r["n_gold_givens"] for r in rs)
        with_ref = [r for r in rs if r["n_gold_givens"] > 0]
        return {
            "n": n,
            "faithful": sum(r["faithful"] for r in rs) / n,
            "faithful_on_given_refs": (sum(r["faithful"] for r in with_ref) / len(with_ref)) if with_ref else None,
            "n_given_refs": len(with_ref),
            "given_precision": sum(r["n_faithful"] for r in rs) / g if g else None,
            "given_recall": sum(r["gold_recalled"] for r in rs) / gold if gold else None,
            "grammatical": sum(r["grammatical"] for r in rs) / n,
            "valid": sum(r["valid"] for r in rs) / n,
            "answer_acc": sum(r["answer_correct"] for r in rs) / n,
            "answer_acc_lenient": sum(r["answer_correct_lenient"] for r in rs) / n,
            "has_proof": sum(r["has_proof"] for r in rs) / n,
            "proof_closed": sum(r["proof_closed"] for r in rs) / n,
            "mean_gen_tokens": (sum(r.get("gen_tokens", 0) for r in rs) / n),
        }
    fams = collections.defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    return {"overall": agg(rows), "per_family": {f: agg(v) for f, v in sorted(fams.items())}}


# ---------------------------------------------------------------- reference sources
def gold_generation(rec: dict) -> str:
    return rec["messages"][1]["content"]


def corrupt(rec: dict, kind: str, rng: random.Random) -> str | None:
    text = gold_generation(rec)
    lines = text.split("\n")
    gi = [i for i, l in enumerate(lines) if GIVEN_RE.match(l)]
    if kind == "quote":          # quote not in the problem text
        if not gi:
            return None
        i = rng.choice(gi)
        lines[i] = lines[i][:-1] + ' and more"'
    elif kind == "formula":      # formula no longer matches its sentence
        if not gi:
            return None
        i = rng.choice(gi)
        lab, f, q = GIVEN_RE.match(lines[i]).groups()
        lines[i] = f'{lab} ~({f}) ; given "{q}"'
    elif kind == "syntax":       # a malformed line
        body = [i for i, l in enumerate(lines) if re.match(r"^\d", l)]
        i = rng.choice(body)
        lines[i] = lines[i].replace(" ; ", " ;; ", 1) + " ((("
    elif kind == "rule":         # wrong but well-formed justification
        rl = [i for i, l in enumerate(lines) if re.match(r"^\d\S* .* ; (mp|and_e|and_i|subst|calc|mt|or_e)\b", l)]
        if not rl:
            return None
        i = rng.choice(rl)
        lines[i] = re.sub(r" ; (\w+)", " ; refl", lines[i], count=1)
    elif kind == "answer":       # only the Answer: line is wrong; the proof is untouched
        return re.sub(r"(?m)^Answer: .*$", "Answer: WRONG", text)
    elif kind == "ans":          # the proof's own ans line names a different, well-formed value
        m = re.search(r"(?m)^ans (\S+) ; (.*)$", text)
        if not m:
            return None
        v = m.group(1)
        alt = {"yes": "no", "no": "yes", "true": "false", "false": "true"}.get(v)
        if alt is None:
            alt = str(int(v) + 1) if re.fullmatch(r"-?\d+", v) else None
        if alt is None:
            return None
        return text[: m.start(1)] + alt + text[m.end(1):]
    elif kind == "cite":         # a derived line cites the wrong earlier line
        cands = []
        for i, l in enumerate(lines):
            m = re.match(r"^(\d+) .* ; [a-z_.:]+ (\d+)\b", l)
            if m and not re.search(r"given|obs|assume", l) and int(m.group(2)) > 1:
                cands.append(i)
        if not cands:
            return None
        i = rng.choice(cands)
        lines[i] = re.sub(r"( ; [a-z_.:]+ )(\d+)\b", lambda m: m.group(1) + str(int(m.group(2)) - 1), lines[i], count=1)
    elif kind == "truncate":     # proof cut in the middle, no </proof>
        body = text.split("</proof>")[0]
        return body[: len(body) // 2]
    else:
        raise ValueError(kind)
    return "\n".join(lines)


# ---------------------------------------------------------------- generation
def last_call(text: str) -> str:
    before = text[: -len("<result>")] if text.endswith("<result>") else text
    line = before.rstrip("\n").split("\n")[-1]
    return line.split(" do ", 1)[1] if " do " in line else line


def generate(records, args) -> list[dict]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from formal_chat_format import encode_continuation, render_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")]
    if tok.eos_token_id is not None:
        stop_ids.append(tok.eos_token_id)
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=0,
              tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 0, "video": 0} if args.no_mm else None)
    states = []
    for rec in records:
        user = f"{TAG_USER}\n{rec['prompt']}"
        pids = tok(render_prompt(tok, user), add_special_tokens=False)["input_ids"]
        states.append({"prompt_ids": pids, "text": "", "gen_tokens": 0, "tool_calls": 0, "done": False,
                       "finish": None, "tool_log": []})
    rounds = 0
    while True:
        active = [i for i, s in enumerate(states) if not s["done"]]
        if not active:
            break
        rounds += 1
        prompts, params = [], []
        for i in active:
            s = states[i]
            ids = encode_continuation(tok, s["prompt_ids"], s["text"]) if s["text"] else s["prompt_ids"]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(temperature=0.0, max_tokens=max(1, args.max_new_tokens - s["gen_tokens"]),
                                         stop=["<result>"], include_stop_str_in_output=True,
                                         stop_token_ids=stop_ids, skip_special_tokens=True))
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        print(f"[round {rounds}] {len(active)} active, {time.time()-t0:.1f}s", flush=True)
        for i, out in zip(active, outs):
            s, o = states[i], out.outputs[0]
            s["text"] += o.text
            s["gen_tokens"] += len(o.token_ids)
            s["finish"] = o.finish_reason
            hit_tool = o.finish_reason == "stop" and o.stop_reason == "<result>"
            if hit_tool and s["tool_calls"] < args.max_tool_calls and s["gen_tokens"] < args.max_new_tokens:
                rec = records[i]
                call = last_call(s["text"])
                result = toolsim.run(call, rec.get("tool_docs") or [])
                s["tool_log"].append({"call": call, "result": result})
                s["text"] += result + "</result>\n"
                s["tool_calls"] += 1
            else:
                s["done"] = True
    return states


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default="model", help="model | gold | corrupt:<quote,formula,syntax,rule,cite,ans,answer,truncate>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-tool-calls", type=int, default=12)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--no-mm", action="store_true", help="disable image/video inputs (VLM checkpoints)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.test_jsonl)]
    if args.limit:
        records = records[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    if args.source == "model":
        states = generate(records, args)
        for rec, s in zip(records, states):
            r = score(rec, s["text"])
            r.update(generation=s["text"], gen_tokens=s["gen_tokens"], tool_calls=s["tool_calls"],
                     finish_reason=s["finish"], tool_log=s["tool_log"])
            rows.append((rec, r))
    elif args.source == "gold":
        rows = [(rec, {**score(rec, gold_generation(rec)), "generation": gold_generation(rec)}) for rec in records]
    elif args.source.startswith("corrupt:"):
        rng = random.Random(args.seed)
        for kind in args.source.split(":", 1)[1].split(","):
            for rec in records:
                g = corrupt(rec, kind, rng)
                if g is not None:
                    rows.append(({**rec, "family": f"{kind}"}, {**score(rec, g), "generation": g,
                                                               "true_family": rec["family"]}))
    else:
        raise SystemExit(f"unknown --source {args.source}")
    elapsed = time.time() - t0

    flat = []
    with open(out_dir / "generations.jsonl", "w") as f:
        for rec, r in rows:
            r = {"id": rec["id"], "family": rec["family"], "prompt": rec["prompt"], "gold_answer": rec["answer"], **r}
            flat.append(r)
            f.write(json.dumps(r) + "\n")
    summary = {"model": args.model, "source": args.source, "test_jsonl": args.test_jsonl, "n": len(flat),
               "max_new_tokens": args.max_new_tokens, "elapsed_s": elapsed, **aggregate(flat)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    o = summary["overall"]
    print(json.dumps({"n": summary["n"], "elapsed_s": round(elapsed, 1),
                      **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}}))
    for fam, v in summary["per_family"].items():
        print(f"  {fam:10s} n={v['n']:4d} faithful={v['faithful']:.3f} gram={v['grammatical']:.3f} "
              f"valid={v['valid']:.3f} acc={v['answer_acc']:.3f} acc_len={v['answer_acc_lenient']:.3f}")


if __name__ == "__main__":
    main()
