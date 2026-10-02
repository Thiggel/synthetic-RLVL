#!/usr/bin/env python
"""Checker-guided proof decoding: every proof that comes out is valid by construction.

Answer to the user's question (2026-10-02): can constrained decoding *guarantee* that a proof
which comes out is valid, with backtracking at dead ends? Token masks (rlvl.Guide) only
guarantee grammar, because a line states its formula before its rule (`3 x = 5 ; subst 1 2`),
so whether the formula follows is known only at the end of the line. This decoder therefore
works at the level of proof lines:

  expand    for the current verified prefix, sample candidate next lines from the policy
            (vLLM, prompt + "<proof>\\n" + prefix, stop at "\\n"; one greedy line plus K lines
            at temperature T; prefix caching makes each request cost only the new line)
  verify    rlvl.check(prompt, prefix + line, strict=True). A line is accepted iff the
            report has no parse error and every line is ok, except block claims whose block
            is still open (`3 ~p ; contra` until 3.x closes). An `ans` line is accepted only
            if the whole proof is ok, which includes the leak rule (the answer must rest on
            given/obs lines, not on `know`). `back`, `do` and `</proof>` before `ans` are rejected.
  repair    (--repair) a candidate whose formula is fine but whose forward rule or citations
            are wrong (code `rule`) is retried with every forward rule and up to two cited
            labels from the last --repair-window lines; the formula is the model's own
  DFS       the accepted candidates of a node are tried in order (greedy line first, then by
            log-probability). A node whose children all fail is re-expanded up to
            --node-expansions times with new samples, then dropped (backtrack to the parent).
  finish    after an accepted `ans t ; N` the decoder appends "</proof>\\nAnswer: t"
            (forced text: the Answer line must agree with `ans`)

Every proof the decoder returns therefore passes rlvl.check(strict=True). There is no
completeness guarantee (the search space is infinite), so each item has a budget
(--max-expansions vLLM expansions, --max-lines, --max-proof-tokens); an item that exhausts it
returns its deepest verified prefix, which scores as an unfinished proof. Validity is not
faithfulness: `given` lines are only checked to quote the prompt, so a valid proof can still
answer wrongly (valid_wrong).

Output: <out-dir>/generations.jsonl (the eval_formal_bench_vllm schema plus search
statistics), summary.json (aggregate() of eval_formal_bench_vllm plus decoder stats).
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rlvl  # noqa: E402
from eval_formal_bench_vllm import aggregate, fit_prompts, score  # noqa: E402
from eval_formal_vllm import TAG_USER  # noqa: E402
from formal_rewards import TRUST, UNIT_CONSTANTS, components, premise_numbers_ok, tautology  # noqa: E402

OPEN = "<proof>\n"
BLOCK_RULES = {"imp", "contra", "cases", "all", "pick"}
FORWARD = {0: ["lem", "refl", "calc"], 1: ["split", "or", "absurd", "inst", "some", "calc", "closed", "subst"],
           2: ["mp", "and", "not", "subst"]}
_LINE = re.compile(r"^(\d+(?:\.\d+)*) (.*?) ; ([a-z]+)\b(.*)$")
_ANS = re.compile(r"^\s*ans\s+(.+?)\s*;")
_STEP = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*?)\s*;\s*(.*)$")


def open_claims(lines: list[dict]) -> set[int]:
    """Indices of block-claim lines whose block is still open at the end of the proof."""
    if not lines:
        return set()
    last = lines[-1]["label"]
    out = set()
    for i, ln in enumerate(lines):
        if ln["kind"] == "step" and ln.get("rule") in BLOCK_RULES and ln["label"]:
            if i == len(lines) - 1 or last.startswith(ln["label"] + "."):
                out.add(i)
    return out


def verdict(rep: dict, line: str) -> str:
    """'ok', 'done' (accepted ans line) or the failure reason."""
    if rep.get("fatal"):
        return (rep["fatal"] or {}).get("code") or "fatal"
    s = line.strip()
    if s.startswith("ans "):
        if rep.get("ok"):
            return "done"
        err = rep.get("first_error") or {}
        return "ans_" + (err.get("code") or "incomplete")
    lines = rep.get("lines") or []
    pend = open_claims(lines)
    for i, ln in enumerate(lines):
        if not ln["ok"] and i not in pend:
            return ln.get("code") or "invalid"
    if rep.get("n_back"):
        return "back"
    return "ok"


def _parse(line: str):
    """(label, formula, rule, cites, quote) of a step line, else None."""
    m = _STEP.match(line)
    if not m:
        return None
    lab, f, just = m.groups()
    q = re.search(r'"([^"]*)"', just)
    toks = just.split('"', 1)[0].split()
    return lab, f, toks[0] if toks else "", [t for t in toks[1:] if re.fullmatch(r"\d+(?:\.\d+)*", t)], q.group(1) if q else ""


def harden(prompt: str, prior: list[str], line: str, max_know: int | None) -> str | None:
    """The Stage-2 hacks (formal_rewards.line_stats) rejected per line, so the search cannot exploit them:
    a premise whose numbers its quote does not state (`chips = grid_size ; given "5x5 grid"` is fine,
    `x = 42 ; given "<sentence>"` is not), a `know`/`def` with numbers the prompt does not contain, a repeated
    premise, and derived lines that derive nothing (tautology `x = x`, premise-free arithmetic, restating a
    cited formula). --max-know caps background-knowledge lines."""
    p = _parse(line)
    if p is None:
        return None
    lab, f, rule, cites, quote = p
    norm = lambda x: "".join(x.split())
    prev = {q[0]: q for q in map(_parse, prior) if q}
    if rule in TRUST - {"assume"}:
        src = quote if rule in ("given", "obs") else prompt + UNIT_CONSTANTS
        if not premise_numbers_ok(f, src):
            return "h_prem_numbers"
        if any(q[2] in TRUST and norm(q[1]) == norm(f) for q in prev.values()):
            return "h_dup_premise"
        if rule == "know" and max_know is not None and sum(q[2] == "know" for q in prev.values()) >= max_know:
            return "h_know_cap"
        return None
    if rule in BLOCK_RULES:
        return None
    if tautology(f):
        return "h_tautology"
    if rule == "calc" and not cites and not re.search(r"[A-Za-z_]", f):
        return "h_ground_calc"
    if any(c in prev and norm(prev[c][1]) == norm(f) for c in cites):
        return "h_restate"
    return None


def precheck(line: str) -> str | None:
    s = line.strip()
    if not s:
        return "empty"
    if s.startswith("back ") or re.match(r"^\d+(?:\.\d+)* do ", s):
        return "back_or_do"
    if s.startswith("</proof>"):
        return "early_close"
    return None


def repairs(line: str, labels: list[str]) -> list[str]:
    """Same label and formula, every forward rule with 0-2 cited labels from `labels`."""
    m = _LINE.match(line.strip())
    if not m or m.group(3) in {"given", "obs", "know", "lib", "def", "assume"} | BLOCK_RULES:
        return []
    lab, formula = m.group(1), m.group(2)
    out = []
    for r in FORWARD[0]:
        out.append(f"{lab} {formula} ; {r}")
    for a in labels:
        out += [f"{lab} {formula} ; {r} {a}" for r in FORWARD[1]]
    for a, b in itertools.permutations(labels, 2):
        out += [f"{lab} {formula} ; {r} {a} {b}" for r in FORWARD[2]]
    orig = line.strip()
    return [c for c in dict.fromkeys(out) if c != orig]


class Node:
    __slots__ = ("lines", "cands", "seen", "expansions")

    def __init__(self, lines):
        self.lines = lines          # accepted proof lines (each ending in "\n")
        self.cands = []             # verified, untried children (line text)
        self.seen = set()           # every candidate text already proposed here
        self.expansions = 0


class Item:
    def __init__(self, rec, prompt_ids):
        self.rec = rec
        self.prompt_ids = prompt_ids
        self.stack = [Node([])]
        self.done = False
        self.result = None          # final assistant text
        self.status = None          # found / budget / exhausted / max_lines / max_tokens
        self.expansions = 0
        self.backtracks = 0
        self.gen_tokens = 0         # every sampled token, accepted or not
        self.n_cands = 0
        self.n_repaired_used = 0
        self.fail = collections.Counter()
        self.deepest = []
        self.repaired = set()

    def body(self, node=None):
        return "".join((node or self.stack[-1]).lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-jsonl", default="/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N items (smoke)")
    ap.add_argument("--k", type=int, default=4, help="sampled candidate lines per expansion (plus one greedy line)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-greedy", action="store_true", help="no greedy candidate in the first expansion of a node")
    ap.add_argument("--node-expansions", type=int, default=2, help="expansions of one node before backtracking")
    ap.add_argument("--max-expansions", type=int, default=64, help="per-item budget of expansions")
    ap.add_argument("--max-lines", type=int, default=48)
    ap.add_argument("--max-line-tokens", type=int, default=192)
    ap.add_argument("--max-proof-tokens", type=int, default=2048)
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--repair-window", type=int, default=8)
    ap.add_argument("--no-harden", action="store_true", help="accept any rlvl-ok line (the 2026-10-02 smoke setting)")
    ap.add_argument("--max-know", type=int, default=None, help="cap on `know` lines per proof (default: no cap)")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.max_new_tokens = args.max_proof_tokens

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from formal_chat_format import render_prompt

    records = [json.loads(l) for l in open(args.test_jsonl)]
    if args.limit:
        records = records[:args.limit]
    fit_prompts(records, args)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0})
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    items = [Item(r, tok(render_prompt(tok, f"{TAG_USER}\n{r['prompt']}"), add_special_tokens=False)["input_ids"])
             for r in records]

    def sp(n, temp):
        return SamplingParams(n=n, temperature=temp, max_tokens=args.max_line_tokens, stop=["\n"],
                              include_stop_str_in_output=True, stop_token_ids=stop_ids, logprobs=0,
                              skip_special_tokens=True, seed=None)

    def stage2(it, line):
        """An `ans` line is accepted only if the finished proof is Stage-2 valid with faithful premises
        (formal_rewards.components: grounded, ans agrees, >= 1 checked derived ancestor, not circular)."""
        m = _ANS.match(line)
        if not m:  # rlvl accepted an `ans` line without the `<value> ;` shape (crashed job 10249)
            return "s2_invalid"
        c = components(it.rec, OPEN + it.body() + line + "</proof>\nAnswer: " + m.group(1))
        if not c["valid"]:
            return "s2_circular" if c["circular"] else ("s2_no_derived" if c["n_ok"] < 1 else "s2_invalid")
        return "done" if c["prem_ok"] else "s2_prem"

    def finish(it, status, text=None):
        it.done, it.status = True, status
        it.result = text if text is not None else OPEN + "".join(it.deepest)

    t0 = time.time()
    rounds = 0
    while True:
        # 1) advance every item's DFS until it needs an expansion (or finishes)
        need = []
        for it in items:
            while not it.done:
                node = it.stack[-1]
                if len(node.lines) > len(it.deepest):
                    it.deepest = node.lines
                if node.cands:
                    line = node.cands.pop(0)
                    if line in it.repaired:
                        it.n_repaired_used += 1
                    if line.strip().startswith("ans ") and _ANS.match(line):
                        ans = _ANS.match(line).group(1)
                        finish(it, "found", OPEN + it.body() + line + "</proof>\nAnswer: " + ans)
                        break
                    child = Node(node.lines + [line])
                    if len(child.lines) >= args.max_lines:
                        it.deepest = child.lines
                        finish(it, "max_lines")
                        break
                    it.stack.append(child)
                    continue
                if node.expansions < args.node_expansions and it.expansions < args.max_expansions:
                    need.append(it)
                    break
                if it.expansions >= args.max_expansions:
                    finish(it, "budget")
                    break
                it.stack.pop()
                it.backtracks += 1
                if not it.stack:
                    finish(it, "exhausted")
        if not need:
            break
        rounds += 1
        # 2) one batched vLLM call: per needy item one greedy line (first expansion) + K samples
        prompts, params, owners = [], [], []
        for it in need:
            node = it.stack[-1]
            text = OPEN + it.body()
            ids = it.prompt_ids + tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) - len(it.prompt_ids) >= args.max_proof_tokens:
                finish(it, "max_tokens")
                continue
            if node.expansions == 0 and not args.no_greedy:
                prompts.append(TokensPrompt(prompt_token_ids=ids)); params.append(sp(1, 0.0)); owners.append(it)
            prompts.append(TokensPrompt(prompt_token_ids=ids)); params.append(sp(args.k, args.temperature)); owners.append(it)
            node.expansions += 1
            it.expansions += 1
        tg = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False) if prompts else []
        tg = time.time() - tg
        # 3) collect candidates per item, ordered greedy first then by log-probability
        cands = collections.defaultdict(list)
        for it, out, p in zip(owners, outs, params):
            for o in out.outputs:
                it.gen_tokens += len(o.token_ids)
                cands[id(it)].append((o.text, o.cumulative_logprob if o.cumulative_logprob is not None else 0.0,
                                      p.temperature == 0.0))
        jobs = []  # (item, line, is_repair)
        for it in need:
            if it.done:
                continue
            node = it.stack[-1]
            got = cands.get(id(it), [])
            # the greedy line first, then the samples by log-probability
            for text, _, _ in sorted(got, key=lambda c: (not c[2], -c[1])):
                line = text if text.endswith("\n") else text + "\n"
                if line in node.seen:
                    continue
                node.seen.add(line)
                it.n_cands += 1
                why = precheck(line) or (None if args.no_harden else
                                         harden(it.rec["prompt"], node.lines, line, args.max_know))
                if why:
                    it.fail[why] += 1
                    continue
                jobs.append((it, line, False))
        tc = time.time()
        reps = rlvl.check_batch([j[0].rec["prompt"] for j in jobs], [j[0].body() + j[1] for j in jobs], strict=True)
        rule_fail = collections.defaultdict(list)
        for (it, line, _), rep in zip(jobs, reps):
            v = verdict(rep, line)
            if v == "done" and not args.no_harden:
                v = stage2(it, line)
            if v in ("ok", "done"):
                it.stack[-1].cands.append(line)
            else:
                it.fail[v] += 1
                if v == "rule":
                    rule_fail[id(it)].append((it, line))
        # 4) repair: items whose expansion produced no verified line
        if args.repair:
            rjobs = []
            for key, fails in rule_fail.items():
                it = fails[0][0]
                node = it.stack[-1]
                if node.cands:
                    continue
                labels = [m.group(1) for m in (re.match(r"^(\d+(?:\.\d+)*) ", l) for l in node.lines) if m]
                labels = labels[-args.repair_window:]
                for _, line in fails[:2]:
                    for c in repairs(line, labels):
                        c += "\n"
                        if c not in node.seen:
                            node.seen.add(c)
                            rjobs.append((it, c))
            if rjobs:
                rreps = rlvl.check_batch([j[0].rec["prompt"] for j in rjobs], [j[0].body() + j[1] for j in rjobs],
                                         strict=True)
                for (it, line), rep in zip(rjobs, rreps):
                    if not args.no_harden and harden(it.rec["prompt"], it.stack[-1].lines, line, args.max_know):
                        continue
                    if verdict(rep, line) == "ok" and len(it.stack[-1].cands) < 2:
                        it.stack[-1].cands.append(line)
                        it.repaired.add(line)
        tc = time.time() - tc
        n_done = sum(it.done for it in items)
        n_found = sum(it.status == "found" for it in items)
        print(f"[round {rounds}] expanded {len(need)} gen {tg:.1f}s check {tc:.1f}s ({len(jobs)} lines) "
              f"done {n_done}/{len(items)} found {n_found}", flush=True)

    elapsed = time.time() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    with open(out_dir / "generations.jsonl", "w") as f:
        for it in items:
            rec = it.rec
            r = {k: rec[k] for k in ("bench", "group", "id", "gold", "answer_type", "gold_label", "system_answerable")}
            r["sample"] = 0
            text = it.result
            c = components(rec, text)
            r.update(score(rec, text), valid_s2=c["valid"], prem_ok=c["prem_ok"], cvf=c["correct"] * c["valid"] * c["prem_ok"],
                     n_taut=c["n_taut"], circular=c["circular"], generation=text, gen_tokens=it.gen_tokens,
                     proof_tokens=len(tok(text, add_special_tokens=False)["input_ids"]),
                     finish_reason=it.status, truncated=rec.get("truncated", False),
                     found=it.status == "found", expansions=it.expansions, backtracks=it.backtracks,
                     n_cands=it.n_cands, n_lines=text.count("\n"), repaired_lines=it.n_repaired_used,
                     fail=dict(it.fail))
            flat.append(r)
            f.write(json.dumps(r, default=str) + "\n")
    found = [r for r in flat if r["found"]]
    assert all(r["valid"] for r in found), "a found proof failed the checker"
    if not args.no_harden:
        assert all(r["valid_s2"] and r["prem_ok"] for r in found), "a found proof failed the Stage-2 check"
    fails = collections.Counter()
    for it in items:
        fails.update(it.fail)
    summary = {"model": args.model, "test_jsonl": args.test_jsonl, "n": len(flat), "elapsed_s": elapsed,
               "rounds": rounds, "args": vars(args),
               "decoder": {"found": len(found) / len(flat),
                           "status": dict(collections.Counter(r["finish_reason"] for r in flat)),
                           "mean_expansions": sum(r["expansions"] for r in flat) / len(flat),
                           "mean_backtracks": sum(r["backtracks"] for r in flat) / len(flat),
                           "mean_gen_tokens": sum(r["gen_tokens"] for r in flat) / len(flat),
                           "mean_proof_tokens_found": (sum(r["proof_tokens"] for r in found) / len(found)) if found else None,
                           "mean_gen_tokens_found": (sum(r["gen_tokens"] for r in found) / len(found)) if found else None,
                           "repaired_lines": sum(r["repaired_lines"] for r in flat),
                           "valid_s2": sum(r["valid_s2"] for r in flat) / len(flat),
                           "cvf": sum(r["cvf"] for r in flat) / len(flat),
                           "candidate_failures": dict(fails.most_common())},
               **aggregate(flat)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n", "elapsed_s", "rounds", "decoder")}, indent=1))
    o = summary["overall"]["all"] if "overall" in summary else {}
    print({k: round(o[k], 3) for k in ("valid", "correct", "valid_correct", "in_system", "uses_know") if k in o})


if __name__ == "__main__":
    main()
