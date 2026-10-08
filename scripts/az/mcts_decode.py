#!/usr/bin/env python
"""Stage 3: line-level AlphaZero search (PUCT) over formal proofs, with checker pruning and a value head.

docs/research_plan.md, Stage 3. The state is prompt + verified proof prefix and an action is one proof line.
This is sampled MuZero/AlphaZero (Hubert et al. 2021) at the line level:

  expand    at a leaf, sample K candidate lines from the policy (vLLM, stop at "\\n", prefix caching). The prior of
            a distinct line is its empirical frequency among the K samples, the sampled-AZ estimate of the policy.
  prune     every candidate goes through the checks of scripts/guided_decode.py (precheck, Stage-2 hardening,
            rlvl strict, an `ans` line must leave a Stage-2-valid proof). Rejected lines are illegal moves; the
            prior is renormalized over the legal ones (--no-prune keeps every line: the plan's ablation).
  evaluate  the value of the new leaf is the probe of scripts/az/value_probe.py on the policy's last hidden
            state at the end of the prefix (--value probe), or 0.5 everywhere (--value none). A leaf with no legal
            child is re-expanded once with fresh samples, then counts as a loss (value 0).
  terminal  an accepted `ans` line ends the proof. Its value is the probe at the end of the proof (--terminal
            probe: the honest test-time setting, no gold) or the gold reward cvf (--terminal gold: self-play on
            training prompts, where the reward is known).
  select    PUCT: argmax Q + c * P * sqrt(N_parent) / (1 + N), unvisited children use the parent's mean value.
  move      after --sims simulations from the current root, commit the most-visited child (test) or sample it
            by visits^(1/tau) (--move-temp > 0, self-play), record the visit distribution as the policy target,
            and re-root. A committed root whose subtree is all dead is abandoned for its parent.

Budget per item: --max-expansions vLLM expansions (the guided DFS default is 64), --max-lines, --max-proof-tokens.
An item that exhausts it returns its committed prefix (an unfinished proof).

Outputs: <out-dir>/generations.jsonl (guided_decode schema), summary.json, search.jsonl (per move: prefix, the
children with prior / visits / Q; the AZ policy and value targets).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import rlvl  # noqa: E402
from eval_formal_bench_vllm import aggregate, fit_prompts, score  # noqa: E402
from eval_formal_vllm import TAG_USER  # noqa: E402
from formal_rewards import components  # noqa: E402
from guided_decode import _ANS, OPEN, harden, precheck, verdict  # noqa: E402


# --reward c_cvf (2026-10-08, the user's "correctness + validity" for AlphaZero): the tree keeps checker-invalid
# lines, since a pruned tree only ever reaches valid proofs and the validity term would be constant. Dropped are only
# lines that do not parse (or back/do/early close, a fatal checker error) and the Stage-2 hacks that add no proof
# content; a premise with unstated numbers (h_prem_numbers), every line checker error and every `ans` line stay, and
# are paid for at the terminal through valid * prem_ok.
HARD = {"empty", "back_or_do", "early_close", "parse", "fatal", "back",
        "h_dup_premise", "h_know_cap", "h_tautology", "h_ground_calc", "h_restate"}


def c_cvf(c: dict) -> float:
    """correct * (0.5 + 0.5 * valid * premise numbers stated): the additive reward of GRPO run G20."""
    return float(c["correct"]) * (0.5 + 0.5 * float(c["valid"]) * float(c["prem_ok"]))


def relaxed(prompt, prior_lines, line, rep, rec):
    """--reward c_cvf: (keep, is_terminal, gold c_cvf, soft failure code or None) for a line that passed precheck."""
    v = verdict(rep, line)
    is_ans = line.strip().startswith("ans ")
    m = _ANS.match(line) if is_ans else None
    if v in HARD or rep.get("fatal") or (is_ans and not m):
        return False, False, 0.0, v
    if not m:
        return True, False, 0.0, (None if v == "ok" else v)
    c = components(rec, OPEN + "".join(prior_lines) + line + "</proof>\nAnswer: " + m.group(1))
    ok = v == "done" and c["valid"] and c["prem_ok"]
    return True, True, c_cvf(c), (None if ok else ("s2_invalid" if v == "done" else v))


class TNode:
    __slots__ = ("lines", "parent", "children", "prior", "N", "W", "v", "expanded", "expansions", "terminal",
                 "term_value", "dead", "line")

    def __init__(self, lines, parent=None, prior=1.0, line=""):
        self.lines, self.parent, self.prior, self.line = lines, parent, prior, line
        self.children: list[TNode] = []
        self.N, self.W, self.v = 0, 0.0, None
        self.expanded = False
        self.expansions = 0
        self.terminal = False
        self.term_value = 0.0
        self.dead = False

    def q(self):
        return self.W / self.N if self.N else None


class Item:
    def __init__(self, rec, prompt_ids):
        self.rec, self.prompt_ids = rec, prompt_ids
        self.root = TNode([])
        self.move_start = 0         # root.N when the current move began
        self.done = False
        self.result = None
        self.status = None
        self.expansions = 0
        self.gen_tokens = 0
        self.n_cands = 0
        self.fail = collections.Counter()
        self.wrong_terminals = 0
        self.moves = []             # per committed move: the policy target
        self.leaf = None            # leaf awaiting expansion this round
        self.path = None


class Value:
    """Probe on the policy's last hidden state (scripts/az/value_probe.py writes probe_<target>.pt)."""

    def __init__(self, model_dir, probe_path, batch):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
        p = torch.load(probe_path)
        self.w, self.b = p["w"].cuda(), p["b"].cuda()
        self.batch = batch

    def __call__(self, seqs: list[list[int]]) -> list[float]:
        torch = self.torch
        out = [0.0] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        with torch.no_grad():
            for s in range(0, len(order), self.batch):
                idx = order[s:s + self.batch]
                L = max(len(seqs[i]) for i in idx)
                x = torch.zeros((len(idx), L), dtype=torch.long)
                att = torch.zeros_like(x)
                for j, i in enumerate(idx):  # right padding, as in value_probe.extract
                    x[j, :len(seqs[i])] = torch.tensor(seqs[i])
                    att[j, :len(seqs[i])] = 1
                hs = self.model(input_ids=x.cuda(), attention_mask=att.cuda(), output_hidden_states=True,
                                logits_to_keep=1).hidden_states[-1]
                last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=hs.device)
                h = hs[torch.arange(len(idx), device=hs.device), last].float()
                v = torch.sigmoid(h @ self.w + self.b).tolist()
                for j, i in enumerate(idx):
                    out[i] = v[j]
        return out


def puct_child(node: TNode, c: float) -> TNode | None:
    live = [ch for ch in node.children if not ch.dead]
    if not live:
        return None
    fpu = node.q() if node.N else (node.v if node.v is not None else 0.5)
    sq = math.sqrt(max(1, node.N))
    return max(live, key=lambda ch: (ch.q() if ch.N else fpu) + c * ch.prior * sq / (1 + ch.N))


def backup(path: list[TNode], value: float):
    for n in path:
        n.N += 1
        n.W += value


def mark_dead(node: TNode):
    """A node all of whose children are dead is dead too (propagates up to the first live ancestor)."""
    while node is not None and node.expanded and not node.terminal and all(ch.dead for ch in node.children) \
            and node.expansions >= 2:
        node.dead = True
        node = node.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-jsonl", default="/vol/tmp2/laitenbf/rlvl_data/datasets/rl_gate_dolci_instruct_20260928/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="file with one item id per line (subset)")
    ap.add_argument("--k", type=int, default=8, help="sampled lines per expansion")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sims", type=int, default=8, help="simulations per committed move")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--move-temp", type=float, default=0.0, help="0: most visited child; >0: sample by visits")
    ap.add_argument("--value", choices=("probe", "none"), default="probe")
    ap.add_argument("--probe", default="/vol/tmp2/laitenbf/rlvl_data/az/value_probe_e4/probe_cvf.pt")
    ap.add_argument("--terminal", choices=("probe", "gold", "one"), default="probe",
                    help="value of a finished Stage-2-valid proof: probe (test time), gold cvf (self-play), 1 (search for any proof)")
    ap.add_argument("--solve", action="store_true",
                    help="data-generation mode (needs --terminal gold): no move commitment; a gold-wrong terminal is "
                         "dead and backs up 0, the search continues from the root until a gold-correct terminal "
                         "(found) or the expansion budget (budget)")
    ap.add_argument("--no-prune", action="store_true", help="ablation: no checker pruning before the terminal check")
    ap.add_argument("--reward", choices=("cvf", "c_cvf"), default="cvf",
                    help="cvf: checker-pruned tree, terminal = correct*valid*prem_ok; c_cvf: relaxed tree (see HARD), "
                         "gold terminal = correct*(0.5+0.5*valid*prem_ok)")
    ap.add_argument("--max-expansions", type=int, default=64)
    ap.add_argument("--max-lines", type=int, default=48)
    ap.add_argument("--max-line-tokens", type=int, default=192)
    ap.add_argument("--max-proof-tokens", type=int, default=2048)
    ap.add_argument("--max-know", type=int, default=None)
    ap.add_argument("--value-batch", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.25)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.solve and args.terminal != "gold":
        ap.error("--solve needs --terminal gold")
    args.max_new_tokens = args.max_proof_tokens
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from formal_chat_format import render_prompt

    records = [json.loads(l) for l in open(args.test_jsonl)]
    if args.ids:
        keep = set(open(args.ids).read().split())
        records = [r for r in records if r["id"] in keep]
    if args.limit:
        records = records[:args.limit]
    fit_prompts(records, args)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0})
    value = Value(args.model, args.probe, args.value_batch) if (args.value == "probe" or args.terminal == "probe") else None
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    items = [Item(r, tok(render_prompt(tok, f"{TAG_USER}\n{r['prompt']}"), add_special_tokens=False)["input_ids"])
             for r in records]
    sp = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=args.max_line_tokens, stop=["\n"],
                        include_stop_str_in_output=True, stop_token_ids=stop_ids, skip_special_tokens=True)

    def ids_of(it, lines):
        return it.prompt_ids + tok(OPEN + "".join(lines), add_special_tokens=False)["input_ids"]

    def finish(it, status, text=None):
        it.done, it.status = True, status
        it.result = text if text is not None else OPEN + "".join(it.root.lines)

    def commit(it):
        """End of a move: record the visit distribution and re-root at the chosen child."""
        root = it.root
        live = [ch for ch in root.children if ch.N and not ch.dead]
        if not live:
            it.move_start = root.N  # nothing visited and alive yet: keep simulating
            return
        if args.move_temp > 0:
            wts = [ch.N ** (1 / args.move_temp) for ch in live]
            ch = rng.choices(live, weights=wts)[0]
        else:
            ch = max(live, key=lambda c: (c.N, c.q()))
        it.moves.append({"prefix": "".join(root.lines), "root_N": root.N, "root_q": root.q(),
                         "children": [{"line": c.line, "prior": c.prior, "N": c.N, "q": c.q(), "terminal": c.terminal,
                                       "dead": c.dead} for c in root.children], "chosen": ch.line})
        if ch.terminal:
            ans = _ANS.match(ch.line).group(1)
            finish(it, "found", OPEN + "".join(ch.lines) + "</proof>\nAnswer: " + ans)
            return
        it.root = ch
        it.move_start = ch.N
        if len(ch.lines) >= args.max_lines:
            finish(it, "max_lines")

    t0 = time.time()
    rounds = 0
    while True:
        # 1) one simulation per active item: descend by PUCT until an unexpanded leaf (terminal leaves back up at once)
        need = []
        for it in items:
            guard = 0
            while not it.done and it.leaf is None and guard < 64:
                guard += 1
                while it.root.dead and it.root.parent is not None:  # abandoned subtree: back to the parent
                    it.root = it.root.parent
                    it.move_start = it.root.N
                if it.root.dead or (it.root.expanded and not it.root.children and it.root.expansions >= 2):
                    finish(it, "exhausted")
                    break
                if not args.solve and it.root.N - it.move_start >= args.sims and it.root.children:
                    commit(it)
                    continue
                node, path = it.root, [it.root]
                while node.expanded and node.children:
                    nxt = puct_child(node, args.c_puct)
                    if nxt is None:
                        break
                    node = nxt
                    path.append(node)
                if node.terminal:
                    backup(path, node.term_value)
                    continue
                if it.expansions >= args.max_expansions:
                    # out of budget: commit the best line found so far, if any
                    best = None if args.solve else max((c for c in it.root.children if c.terminal),
                                                       key=lambda c: c.term_value, default=None)
                    if best is not None:
                        finish(it, "found", OPEN + "".join(best.lines) + "</proof>\nAnswer: " + _ANS.match(best.line).group(1))
                    else:
                        finish(it, "budget")
                    break
                if node.expanded and node.expansions >= 2:  # dead leaf (no legal child twice)
                    node.dead = True
                    backup(path, 0.0)
                    mark_dead(node.parent)
                    continue
                ids = ids_of(it, node.lines)
                if len(ids) - len(it.prompt_ids) >= args.max_proof_tokens:
                    node.dead = True
                    backup(path, 0.0)
                    mark_dead(node.parent)
                    continue
                it.leaf, it.path = node, path
                need.append((it, ids))
        if not need:
            break
        rounds += 1
        # 2) expand: K sampled lines per leaf in one vLLM call
        tg = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids in need], [sp] * len(need), use_tqdm=False)
        tg = time.time() - tg
        jobs, leaf_cands = [], {}
        for (it, _), out in zip(need, outs):
            node = it.leaf
            node.expansions += 1
            it.expansions += 1
            counts = collections.Counter()
            for o in out.outputs:
                it.gen_tokens += len(o.token_ids)
                line = o.text if o.text.endswith("\n") else o.text + "\n"
                counts[line] += 1
            known = {ch.line for ch in node.children}
            leaf_cands[id(it)] = counts
            for line in counts:
                if line in known:
                    continue
                it.n_cands += 1
                why = precheck(line) or (None if args.no_prune else
                                         harden(it.rec["prompt"], node.lines, line, args.max_know))
                if why and (args.reward == "cvf" or why in HARD):
                    it.fail[why] += 1
                    continue
                if why:
                    it.fail["soft_" + why] += 1
                jobs.append((it, line))
        tc = time.time()
        reps = rlvl.check_batch([j[0].rec["prompt"] for j in jobs], ["".join(j[0].leaf.lines) + j[1] for j in jobs],
                                strict=True)
        legal = collections.defaultdict(list)  # id(item) -> [(line, terminal, gold cvf)]
        for (it, line), rep in zip(jobs, reps):
            if args.reward == "c_cvf":
                keep, term, gold, why = relaxed(it.rec["prompt"], it.leaf.lines, line, rep, it.rec)
                if why:
                    it.fail[why if not keep else "soft_" + why] += 1
                if keep:
                    legal[id(it)].append((line, term, 1.0 if args.terminal == "one" else gold if args.terminal == "gold" else 0.0))
                continue
            v = verdict(rep, line)
            text = None
            if v == "done":
                m = _ANS.match(line)
                text = OPEN + "".join(it.leaf.lines) + line + "</proof>\nAnswer: " + (m.group(1) if m else "")
                c = components(it.rec, text)
                v = "done" if (m and c["valid"] and c["prem_ok"]) else "s2_invalid"
            if v in ("ok", "done") or (args.no_prune and not line.strip().startswith("ans ")):
                gold = 1.0 if args.terminal == "one" else 0.0
                if v == "done" and args.terminal == "gold":
                    c = components(it.rec, text)
                    gold = float(c["correct"] * c["valid"] * c["prem_ok"])
                legal[id(it)].append((line, v == "done", gold))
            else:
                it.fail[v] += 1
        tc = time.time() - tc
        # 3) children with renormalized empirical priors; values for the new leaf and for terminal children
        tv = time.time()
        vq, vowners = [], []
        for it, ids in need:
            node = it.leaf
            counts = leaf_cands[id(it)]
            new = legal.get(id(it), [])
            tot = sum(counts[l] for l, _, _ in new) + sum(counts.get(ch.line, 0) for ch in node.children)
            for ch in node.children:  # re-expansion: refresh the priors with the new counts
                ch.prior = (ch.prior + counts.get(ch.line, 0) / max(1, tot)) / 2
            for line, term, gold in new:
                ch = TNode(node.lines + [line], node, counts[line] / max(1, tot), line)
                if term:
                    ch.terminal = True
                    ch.term_value = gold
                    if args.solve and not it.done:
                        if gold >= 1:
                            finish(it, "found", OPEN + "".join(ch.lines) + "</proof>\nAnswer: " + _ANS.match(line).group(1))
                        else:
                            ch.dead = True
                            it.wrong_terminals += 1
                    if args.terminal == "probe":
                        vq.append(ids_of(it, ch.lines)); vowners.append(ch)
                node.children.append(ch)
            node.expanded = True
            if node.v is None:
                if args.value == "probe":
                    vq.append(ids); vowners.append(node)
                else:
                    node.v = 0.5
        if vq:
            for n, v in zip(vowners, value(vq)):
                if n.terminal:
                    n.term_value = v
                else:
                    n.v = v
        tv = time.time() - tv
        for it, _ in need:
            node, path = it.leaf, it.path
            it.leaf = it.path = None
            if not any(not ch.dead for ch in node.children):
                if node.expansions >= 2:
                    node.dead = True
                    backup(path, 0.0)
                    mark_dead(node.parent)
                else:
                    backup(path, 0.0)  # counts as a visit; the leaf is re-expanded on its next selection
                continue
            backup(path, 0.0 if args.solve and any(ch.terminal and ch.dead for ch in node.children) else node.v)
        n_done = sum(it.done for it in items)
        n_found = sum(it.status == "found" for it in items)
        print(f"[round {rounds}] leaves {len(need)} gen {tg:.1f}s check {tc:.1f}s ({len(jobs)} lines) value {tv:.1f}s "
              f"done {n_done}/{len(items)} found {n_found}", flush=True)

    for it in items:  # the loop can end with items whose selection only revisited terminal leaves: close them as is
        if not it.done:
            finish(it, "stalled")
    elapsed = time.time() - t0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    with open(out_dir / "generations.jsonl", "w") as f, open(out_dir / "search.jsonl", "w") as fs:
        for it in items:
            rec = it.rec
            r = {k: rec[k] for k in ("bench", "group", "id", "gold", "answer_type", "gold_label", "system_answerable")}
            r["sample"] = 0
            text = it.result
            c = components(rec, text)
            r.update(score(rec, text), valid_s2=c["valid"], prem_ok=c["prem_ok"],
                     cvf=c["correct"] * c["valid"] * c["prem_ok"], c_cvf=c_cvf(c), n_taut=c["n_taut"], circular=c["circular"],
                     generation=text, gen_tokens=it.gen_tokens,
                     proof_tokens=len(tok(text, add_special_tokens=False)["input_ids"]), finish_reason=it.status,
                     truncated=rec.get("truncated", False), found=it.status == "found", expansions=it.expansions,
                     n_cands=it.n_cands, n_moves=len(it.moves), wrong_terminals=it.wrong_terminals, n_lines=text.count("\n"), fail=dict(it.fail))
            flat.append(r)
            f.write(json.dumps(r, default=str) + "\n")
            fs.write(json.dumps({"id": rec["id"], "cvf": r["cvf"], "correct": r["correct"], "moves": it.moves}) + "\n")
    found = [r for r in flat if r["found"]]
    if not args.no_prune and args.reward == "cvf":
        assert all(r["valid"] and r["valid_s2"] and r["prem_ok"] for r in found), "a found proof failed the checker"
    fails = collections.Counter()
    for it in items:
        fails.update(it.fail)
    summary = {"model": args.model, "test_jsonl": args.test_jsonl, "n": len(flat), "elapsed_s": elapsed,
               "rounds": rounds, "args": vars(args),
               "decoder": {"found": len(found) / len(flat),
                           "status": dict(collections.Counter(r["finish_reason"] for r in flat)),
                           "mean_expansions": sum(r["expansions"] for r in flat) / len(flat),
                           "mean_gen_tokens": sum(r["gen_tokens"] for r in flat) / len(flat),
                           "valid_s2": sum(r["valid_s2"] for r in flat) / len(flat),
                           "cvf": sum(r["cvf"] for r in flat) / len(flat),
                           "c_cvf": sum(r["c_cvf"] for r in flat) / len(flat),
                           "correct": sum(r["correct"] for r in flat) / len(flat),
                           "candidate_failures": dict(fails.most_common())},
               **aggregate(flat)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n", "elapsed_s", "rounds", "decoder")}, indent=1))


if __name__ == "__main__":
    main()
