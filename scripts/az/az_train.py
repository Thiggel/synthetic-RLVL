#!/usr/bin/env python
"""Stage 3: online AlphaZero. MCTS self-play and joint policy (NTP) + value-head training, one model, one GPU.

docs/research_plan.md, Stage 3 (2026-10-05). The search of scripts/az/mcts_decode.py runs on training prompts and
the model learns from it online, AlphaZero style:

  model     the policy LM (init: an EI SFT checkpoint) plus a linear value head on its last hidden state, the same
            backbone for both (v(s) = sigmoid(w . h_last(s) + b)). bf16 on the GPU, fp32 master weights + AdamW on the CPU.
  search    line-level sampled PUCT (mcts_decode.py): K lines per expansion from vLLM (colocated, sleep mode, weights
            synced after every update), checker pruning (precheck / Stage-2 hardening / rlvl strict), leaf value =
            the live value head, terminal value = the gold reward cvf (correct * valid * premise numbers stated) as in
            AlphaZero where the game result is known in self-play, Dirichlet noise on root priors, moves sampled by
            visits^(1/move_temp). An episode ends with an accepted `ans` line (z = its gold cvf) or out of budget (z=0).
  targets   per committed move: the visit distribution over the legal sampled lines (policy target, cross-entropy on
            the line log-probability), value targets for the move's state (z, or (z + root Q)/2 with --value-target
            mix), for its children (a terminal: gold cvf; a dead line: 0; a child visited >= 2: its Q) - the
            correctness signal for partial proofs; plus NTP on every found correct proof (incl. "</proof>\\nAnswer").
  train     after every self-play batch: one pass over --train-moves moves sampled from the last --buffer-iters
            batches, AdamW (backbone --lr, head --head-lr), then the new weights go to vLLM.

Evaluation every --eval-every iterations on the 300-item gate subset (greedy, the same vLLM): valid_prem / cvf /
correct, and the value head's AUC for cvf at the end of the proof and at its middle line (held-out prompts).
Outputs in --out-dir: metrics.jsonl (per iteration), evals.jsonl, selfplay/iter_XXXX.jsonl (episodes), ckpt-XXXX/
(bf16 model + value_head.pt in the probe format of value_probe.py), latest/ (fp32 + optimizer, for --resume).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import rlvl  # noqa: E402
from eval_formal_bench_vllm import fit_prompts, score  # noqa: E402
from eval_formal_vllm import TAG_USER  # noqa: E402
from formal_rewards import components  # noqa: E402
from guided_decode import _ANS, OPEN, harden, precheck, verdict  # noqa: E402
from mcts_decode import TNode, backup, mark_dead, puct_child  # noqa: E402

LINE_NORM = 32.0  # token sums of line log-probs are divided by this (a typical proof line), for a loss of O(1)


class Node(TNode):
    __slots__ = ("ids", "lids")  # token ids of prompt + OPEN + lines; of this node's own line


class Episode:
    def __init__(self, rec, prompt_ids, open_ids):
        self.rec, self.prompt_ids = rec, prompt_ids
        self.root = Node([])
        self.root.ids, self.root.lids = prompt_ids + open_ids, []
        self.move_start = 0
        self.done, self.result, self.status, self.z = False, None, None, 0.0
        self.expansions = self.gen_tokens = self.n_cands = 0
        self.fail = collections.Counter()
        self.moves = []
        self.leaf = self.path = None
        self.final_node = None


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y >= 0.5]
    neg = [s for s, y in zip(scores, labels) if y < 0.5]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    r, rank_sum, i = 0, 0.0, 0
    while i < len(ranked):  # average ranks for ties
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompts", required=True, help="jsonl (gate test schema) of self-play training prompts")
    ap.add_argument("--eval-jsonl", default="/vol/tmp2/laitenbf/rlvl_data/az/gate_subset_300.jsonl")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--prompts-per-iter", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--move-temp", type=float, default=1.0)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet-frac", type=float, default=0.25)
    ap.add_argument("--max-expansions", type=int, default=64)
    ap.add_argument("--max-lines", type=int, default=48)
    ap.add_argument("--max-line-tokens", type=int, default=192)
    ap.add_argument("--max-proof-tokens", type=int, default=2048)
    ap.add_argument("--max-know", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--init-head", default="/vol/tmp2/laitenbf/rlvl_data/az/value_probe_e4/probe_cvf.pt",
                    help="probe_<target>.pt to start the value head from ('' = zero weights, bias logit(0.2))")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--policy-coef", type=float, default=1.0)
    ap.add_argument("--ntp-coef", type=float, default=0.25)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--value-target", choices=("z", "mix"), default="mix")
    ap.add_argument("--min-visits-q", type=int, default=2, help="children with >= this many visits get Q as value target")
    ap.add_argument("--train-moves", type=int, default=1024, help="moves per training pass (sampled from the buffer)")
    ap.add_argument("--moves-per-step", type=int, default=64, help="moves per optimizer step")
    ap.add_argument("--buffer-iters", type=int, default=2)
    ap.add_argument("--tok-budget", type=int, default=12288, help="padded tokens per micro-batch")
    ap.add_argument("--value-batch", type=int, default=16)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.28)
    ap.add_argument("--stop-at", type=float, default=None, help="unix time: save latest/ and exit before it")
    ap.add_argument("--iter-budget-s", type=float, default=3600, help="expected seconds per iteration (for --stop-at)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.max_new_tokens = args.max_proof_tokens
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    (out / "selfplay").mkdir(parents=True, exist_ok=True)

    # vLLM in this process (external_launcher, world size 1) so that weights can be pushed in place, as TRL colocate
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket() as s:
            s.bind(("", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from formal_chat_format import render_prompt

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    state = {"iter": 0, "pool_pos": 0}
    latest = out / "latest"
    src = str(latest) if args.resume and (latest / "state.json").exists() else args.model
    if src != args.model:
        state = json.loads((latest / "state.json").read_text())
        print(f"resuming from {latest} at iteration {state['iter']}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # bf16 model on the GPU (search values + forward/backward); fp32 master weights and AdamW on the CPU, so that
    # the GPU holds only bf16 weights + grads + activations next to vLLM (gruenau L40s are shared)
    def retie(m):  # older `latest` saves wrote a bf16 lm_head next to the fp32 embedding, which transformers then unties
        if m.config.get_text_config().tie_word_embeddings and m.lm_head.weight is not m.get_input_embeddings().weight:
            m.lm_head.weight = m.get_input_embeddings().weight
        return m

    model = retie(AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gpu_params = [p for p in model.parameters() if p.requires_grad]
    cpu_model = retie(AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32))
    master = [torch.nn.Parameter(q.detach().clone()) for q in cpu_model.parameters() if q.requires_grad]
    del cpu_model
    assert len(master) == len(gpu_params) and all(a.shape == b.shape for a, b in zip(master, gpu_params))
    head = torch.nn.Linear(model.config.hidden_size, 1).cuda()
    with torch.no_grad():
        if src != args.model:
            p = torch.load(latest / "value_head.pt")
            head.weight.copy_(p["w"][None]); head.bias.copy_(p["b"])
        elif args.init_head:
            p = torch.load(args.init_head)
            head.weight.copy_(p["w"][None]); head.bias.copy_(p["b"])
        else:
            head.weight.zero_(); head.bias.fill_(math.log(0.2 / 0.8))
    opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, foreach=True)
    opt_head = torch.optim.AdamW(head.parameters(), lr=args.head_lr, betas=(0.9, 0.99), weight_decay=0.0)
    if src != args.model and (latest / "optim.pt").exists():
        o = torch.load(latest / "optim.pt", map_location="cpu")
        opt.load_state_dict(o["backbone"]); opt_head.load_state_dict(o["head"])
    llm = LLM(model=src, tokenizer=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, max_num_seqs=256,
              enable_sleep_mode=True, distributed_executor_backend="external_launcher", max_num_batched_tokens=8192)
    vmodel = llm.llm_engine.model_executor.driver_worker.model_runner.model

    def sync_weights():  # vLLM is asleep (level 2: weights and KV cache freed) since the training pass
        torch.cuda.empty_cache()
        llm.wake_up(tags=["weights"])
        with torch.no_grad():
            for name, p in model.named_parameters():
                vmodel.load_weights([(name, p.detach().to(torch.bfloat16))])
        llm.wake_up(tags=["kv_cache"])
        llm.reset_prefix_cache()

    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>")] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    open_ids = tok(OPEN, add_special_tokens=False)["input_ids"]
    sp_line = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=args.max_line_tokens, stop=["\n"],
                             include_stop_str_in_output=True, stop_token_ids=stop_ids, skip_special_tokens=True)
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model

    def hidden(x, att):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return base(input_ids=x, attention_mask=att).last_hidden_state

    def pad(seqs):
        L = max(len(s) for s in seqs)
        x = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros_like(x)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        return x.cuda(), att.cuda()

    @torch.no_grad()
    def values(seqs):
        model.eval()
        res = [0.0] * len(seqs)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        for s in range(0, len(order), args.value_batch):
            idx = order[s:s + args.value_batch]
            x, att = pad([seqs[i] for i in idx])
            h = hidden(x, att)
            last = torch.tensor([len(seqs[i]) - 1 for i in idx], device=h.device)
            v = torch.sigmoid(head(h[torch.arange(len(idx), device=h.device), last].float())).squeeze(-1).tolist()
            for j, i in enumerate(idx):
                res[i] = v[j]
        return res

    def load_records(path):
        recs = [json.loads(l) for l in open(path)]
        fit_prompts(recs, args)
        return recs

    pool = load_records(args.prompts)
    random.Random(args.seed).shuffle(pool)
    eval_recs = load_records(args.eval_jsonl) if args.eval_jsonl else []

    def prompt_ids(rec):
        return tok(render_prompt(tok, f"{TAG_USER}\n{rec['prompt']}"), add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------ evaluation (greedy, held-out gate subset)
    nl_cache = {}

    def is_nl(t_):
        if t_ not in nl_cache:
            nl_cache[t_] = "\n" in tok.decode([t_])
        return nl_cache[t_]

    def evaluate(it_no):
        t = time.time()
        pids = [prompt_ids(r) for r in eval_recs]
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_proof_tokens, stop_token_ids=stop_ids,
                            skip_special_tokens=True)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp, use_tqdm=False)
        rows, vq_end, vq_mid = [], [], []
        for rec, p, o in zip(eval_recs, pids, outs):
            text = o.outputs[0].text
            c = components(rec, text)
            vp = float(c["valid"]) * float(c["prem_ok"])
            rows.append({"id": rec["id"], "bench": rec["bench"], "correct": float(c["correct"]), "valid": float(c["valid"]),
                         "valid_prem": vp, "cvf": vp * float(c["correct"]),
                         "score_correct": float(score(rec, text).get("correct", 0))})
            gen = list(o.outputs[0].token_ids)
            vq_end.append((p + gen)[:args.max_model_len])
            nl = [i for i, t_ in enumerate(gen) if is_nl(t_)]
            vq_mid.append(p + gen[:nl[len(nl) // 2] + 1] if nl else p + gen[:1])
        v_end, v_mid = values(vq_end), values(vq_mid)
        lab = [r["cvf"] for r in rows]
        res = {"iter": it_no, "n": len(rows), **{m: sum(r[m] for r in rows) / len(rows)
                                                 for m in ("valid", "valid_prem", "cvf", "correct", "score_correct")},
               "value_auc_end": auc(v_end, lab), "value_auc_mid": auc(v_mid, lab),
               "value_mean_end": sum(v_end) / len(v_end), "eval_s": time.time() - t}
        by = collections.defaultdict(list)
        for r in rows:
            by[r["bench"]].append(r)
        res["bench"] = {b: {m: sum(r[m] for r in rr) / len(rr) for m in ("valid_prem", "cvf", "correct")}
                        for b, rr in by.items()}
        with open(out / "evals.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print("[eval]", json.dumps({k: v for k, v in res.items() if k != "bench"}), flush=True)
        return res

    # ------------------------------------------------------------------ self-play (batched PUCT over episodes)
    def noise_root(node):
        live = [c for c in node.children if not c.dead]
        if len(live) > 1 and args.dirichlet_frac > 0:
            d = np_rng.dirichlet([args.dirichlet_alpha] * len(live))
            for c, n in zip(live, d):
                c.prior = (1 - args.dirichlet_frac) * c.prior + args.dirichlet_frac * float(n)

    def finish(ep, status, node=None):
        ep.done, ep.status, ep.final_node = True, status, node
        if node is not None:
            ep.result = OPEN + "".join(node.lines) + "</proof>\nAnswer: " + _ANS.match(node.line).group(1)
            ep.z = node.term_value
        else:
            ep.result = OPEN + "".join(ep.root.lines)

    def record_move(ep, root, chosen):
        ep.moves.append({"pre": root.ids, "root_q": root.q(), "root_N": root.N, "v": root.v, "chosen": chosen.line,
                         "children": [{"lids": c.lids, "line": c.line, "N": c.N, "q": c.q(), "terminal": c.terminal,
                                       "term": c.term_value, "dead": c.dead, "prior": c.prior} for c in root.children]})

    def commit(ep):
        root = ep.root
        live = [ch for ch in root.children if ch.N and not ch.dead]
        if not live:
            ep.move_start = root.N
            return
        wts = [ch.N ** (1 / args.move_temp) for ch in live] if args.move_temp > 0 else None
        ch = rng.choices(live, weights=wts)[0] if wts else max(live, key=lambda c: (c.N, c.q()))
        record_move(ep, root, ch)
        if ch.terminal:
            finish(ep, "found", ch)
            return
        ep.root = ch
        ep.move_start = ch.N
        if ch.expanded:
            noise_root(ch)
        if len(ch.lines) >= args.max_lines:
            finish(ep, "max_lines")

    def self_play(recs):
        eps = [Episode(r, prompt_ids(r), open_ids) for r in recs]
        t0, rounds, t_gen, t_chk, t_val = time.time(), 0, 0.0, 0.0, 0.0
        while True:
            need = []
            for ep in eps:
                guard = 0
                while not ep.done and ep.leaf is None and guard < 64:
                    guard += 1
                    while ep.root.dead and ep.root.parent is not None:
                        ep.root = ep.root.parent
                        ep.move_start = ep.root.N
                    if ep.root.dead or (ep.root.expanded and not ep.root.children and ep.root.expansions >= 2):
                        finish(ep, "exhausted")
                        break
                    if ep.root.N - ep.move_start >= args.sims and ep.root.children:
                        commit(ep)
                        continue
                    node, path = ep.root, [ep.root]
                    while node.expanded and node.children:
                        nxt = puct_child(node, args.c_puct)
                        if nxt is None:
                            break
                        node = nxt
                        path.append(node)
                    if node.terminal:
                        backup(path, node.term_value)
                        continue
                    if ep.expansions >= args.max_expansions:
                        best = max((c for c in ep.root.children if c.terminal and not c.dead),
                                   key=lambda c: c.term_value, default=None)
                        if best is not None:
                            record_move(ep, ep.root, best)
                        finish(ep, "found" if best is not None else "budget", best)
                        break
                    if (node.expanded and node.expansions >= 2) or \
                            len(node.ids) - len(ep.prompt_ids) >= args.max_proof_tokens:
                        node.dead = True
                        backup(path, 0.0)
                        mark_dead(node.parent)
                        continue
                    ep.leaf, ep.path = node, path
                    need.append(ep)
            if not need:
                break
            rounds += 1
            tg = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=ep.leaf.ids) for ep in need], [sp_line] * len(need),
                                use_tqdm=False)
            t_gen += time.time() - tg
            tc = time.time()
            jobs, cands = [], {}
            for ep, o in zip(need, outs):
                node = ep.leaf
                node.expansions += 1
                ep.expansions += 1
                counts, lids = collections.Counter(), {}
                for s in o.outputs:
                    ep.gen_tokens += len(s.token_ids)
                    line = s.text if s.text.endswith("\n") else s.text + "\n"
                    counts[line] += 1
                    if line not in lids:
                        lids[line] = tok(line, add_special_tokens=False)["input_ids"]
                cands[id(ep)] = (counts, lids)
                known = {ch.line for ch in node.children}
                for line in counts:
                    if line in known:
                        continue
                    ep.n_cands += 1
                    why = precheck(line) or harden(ep.rec["prompt"], node.lines, line, args.max_know)
                    if why:
                        ep.fail[why] += 1
                        continue
                    jobs.append((ep, line))
            reps = rlvl.check_batch([j[0].rec["prompt"] for j in jobs], ["".join(j[0].leaf.lines) + j[1] for j in jobs],
                                    strict=True)
            legal = collections.defaultdict(list)
            for (ep, line), rep in zip(jobs, reps):
                v = verdict(rep, line)
                gold = 0.0
                if v == "done":
                    m = _ANS.match(line)
                    text = OPEN + "".join(ep.leaf.lines) + line + "</proof>\nAnswer: " + (m.group(1) if m else "")
                    c = components(ep.rec, text)
                    v = "done" if (m and c["valid"] and c["prem_ok"]) else "s2_invalid"
                    gold = float(c["correct"] * c["valid"] * c["prem_ok"])
                if v in ("ok", "done"):
                    legal[id(ep)].append((line, v == "done", gold))
                else:
                    ep.fail[v] += 1
            t_chk += time.time() - tc
            tv = time.time()
            vq, vowners = [], []
            for ep in need:
                node = ep.leaf
                counts, lids = cands[id(ep)]
                new = legal.get(id(ep), [])
                tot = sum(counts[l] for l, _, _ in new) + sum(counts.get(ch.line, 0) for ch in node.children)
                for ch in node.children:
                    ch.prior = (ch.prior + counts.get(ch.line, 0) / max(1, tot)) / 2
                for line, term, gold in new:
                    ch = Node(node.lines + [line], node, counts[line] / max(1, tot), line)
                    ch.lids = lids[line]
                    ch.ids = node.ids + ch.lids
                    if term:
                        ch.terminal, ch.term_value = True, gold
                    node.children.append(ch)
                first = not node.expanded
                node.expanded = True
                if first and node is ep.root:
                    noise_root(node)
                if node.v is None:
                    vq.append(node.ids); vowners.append(node)
            if vq:
                for n, v in zip(vowners, values(vq)):
                    n.v = v
            t_val += time.time() - tv
            for ep in need:
                node, path = ep.leaf, ep.path
                ep.leaf = ep.path = None
                if not any(not ch.dead for ch in node.children):
                    if node.expansions >= 2:
                        node.dead = True
                        backup(path, 0.0)
                        mark_dead(node.parent)
                    else:
                        backup(path, 0.0)
                    continue
                backup(path, node.v)
            if rounds % 8 == 0:
                print(f"  [round {rounds}] leaves {len(need)} done {sum(e.done for e in eps)}/{len(eps)} "
                      f"found {sum(e.status == 'found' for e in eps)} gen {t_gen:.0f}s chk {t_chk:.0f}s val {t_val:.0f}s",
                      flush=True)
        return eps, {"rounds": rounds, "search_s": time.time() - t0, "gen_s": t_gen, "check_s": t_chk, "value_s": t_val}

    # ------------------------------------------------------------------ training examples from moves / episodes
    def examples(moves, eps_found):
        """Each example: token ids, policy ranges [(start, end, weight)] (log-prob of ids[start:end] given the rest),
        value points [(position, target)]. Weights are already normalized per move / per found episode."""
        ex = []
        for m in moves:
            pre = m["pre"]
            ch = m["children"]
            totN = sum(c["N"] for c in ch if not c["dead"])
            s_tgt = m["z"] if args.value_target == "z" or m["root_q"] is None else (m["z"] + m["root_q"]) / 2
            first = True
            for c in ch:
                w = c["N"] / totN if totN and c["N"] and not c["dead"] else 0.0
                if c["terminal"]:
                    vt = c["term"]
                elif c["dead"]:
                    vt = 0.0
                elif c["N"] >= args.min_visits_q and c["q"] is not None:
                    vt = c["q"]
                else:
                    vt = None
                if w == 0 and vt is None and not first:
                    continue
                ids = (pre + c["lids"])[:args.max_model_len]
                e = {"ids": ids, "pol": [], "val": []}
                if w > 0 and len(ids) > len(pre):
                    e["pol"].append((len(pre), len(ids), args.policy_coef * w / LINE_NORM))
                if first:
                    e["val"].append((len(pre) - 1, s_tgt))
                    first = False
                if vt is not None and len(ids) > len(pre):
                    e["val"].append((len(ids) - 1, vt))
                ex.append(e)
        for ep in eps_found:
            full = ep["ids"][:args.max_model_len]
            ex.append({"ids": full, "pol": [(ep["n_prompt"], len(full), args.ntp_coef / LINE_NORM)], "val": []})
        return ex

    def train_pass(moves, eps_found):
        model.train()
        rng.shuffle(moves)
        n_steps = max(1, math.ceil(len(moves) / args.moves_per_step))
        stats = collections.defaultdict(float)
        for s in range(n_steps):
            mv = moves[s * args.moves_per_step:(s + 1) * args.moves_per_step]
            ef = eps_found[s::n_steps]
            ex = examples(mv, ef)
            n_units = max(1, len(mv) + len(ef))
            n_val = max(1, sum(len(e["val"]) for e in ex))
            ex.sort(key=lambda e: len(e["ids"]))
            mbs, cur = [], []
            for e in ex:
                if cur and (len(cur) + 1) * len(e["ids"]) > args.tok_budget:
                    mbs.append(cur); cur = []
                cur.append(e)
            if cur:
                mbs.append(cur)
            for mb in mbs:
                x, att = pad([e["ids"] for e in mb])
                h = hidden(x, att)
                bi, pos, tgt, wt = [], [], [], []
                for j, e in enumerate(mb):
                    for a, b, w in e["pol"]:
                        for t in range(a, b):
                            bi.append(j); pos.append(t - 1); tgt.append(e["ids"][t]); wt.append(w)
                loss = h.new_zeros((), dtype=torch.float32)
                if bi:
                    hs = h[torch.tensor(bi, device=h.device), torch.tensor(pos, device=h.device)]
                    tg = torch.tensor(tgt, device=h.device)
                    wv = torch.tensor(wt, device=h.device)
                    lp_parts = []
                    for c0 in range(0, hs.shape[0], 1024):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model.lm_head(hs[c0:c0 + 1024])
                        lp_parts.append(torch.log_softmax(logits.float(), -1).gather(-1, tg[c0:c0 + 1024, None])[:, 0])
                    lp = torch.cat(lp_parts)
                    lpol = -(wv * lp).sum() / n_units
                    loss = loss + lpol
                    stats["policy_loss"] += lpol.item()
                    stats["policy_tokens"] += len(bi)
                vb, vp, vt = [], [], []
                for j, e in enumerate(mb):
                    for p_, t_ in e["val"]:
                        vb.append(j); vp.append(p_); vt.append(t_)
                if vb:
                    hv = h[torch.tensor(vb, device=h.device), torch.tensor(vp, device=h.device)].float()
                    logit = head(hv).squeeze(-1)
                    tv = torch.tensor(vt, device=h.device, dtype=torch.float32)
                    lv = torch.nn.functional.binary_cross_entropy_with_logits(logit, tv, reduction="sum") / n_val
                    loss = loss + args.value_coef * lv
                    stats["value_loss"] += lv.item()
                    stats["value_points"] += len(vb)
                    stats["value_abs_err"] += (torch.sigmoid(logit) - tv).abs().sum().item()
                loss.backward()
                del h, loss
            gn = torch.nn.utils.clip_grad_norm_(gpu_params + list(head.parameters()), args.max_grad_norm)
            for mp, gp in zip(master, gpu_params):
                mp.grad = gp.grad.to("cpu").float() if gp.grad is not None else None
                gp.grad = None
            opt.step()
            opt_head.step()
            opt.zero_grad(set_to_none=True)
            opt_head.zero_grad(set_to_none=True)
            with torch.no_grad():
                for mp, gp in zip(master, gpu_params):
                    gp.copy_(mp.to(gp.dtype), non_blocking=True)
            stats["grad_norm"] += float(gn) / n_steps
            stats["opt_steps"] += 1
        torch.cuda.empty_cache()
        stats["value_abs_err"] /= max(1, stats["value_points"])
        for k in ("policy_loss", "value_loss"):  # mean over optimizer steps (logs before 2026-10-05 21:30 hold the sum)
            stats[k] /= max(1, stats["opt_steps"])
        return dict(stats)

    def save(d: Path, full: bool):
        d.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().to("cpu", torch.bfloat16) for k, v in model.state_dict().items()}
        if full:  # fp32 masters for --resume
            names = {}
            for n, q in model.named_parameters(remove_duplicate=False):
                names.setdefault(id(q), []).append(n)
            for gp, mp in zip(gpu_params, master):  # every tied name gets the same fp32 master
                sd.update({n: mp.detach().float() for n in names[id(gp)]})
        model.save_pretrained(d, state_dict=sd, safe_serialization=True)
        tok.save_pretrained(d)
        torch.save({"w": head.weight.detach()[0].cpu(), "b": head.bias.detach().cpu(), "feature": "last",
                    "layer": -1, "target": "cvf"}, d / "value_head.pt")
        if full:
            torch.save({"backbone": opt.state_dict(), "head": opt_head.state_dict()}, d / "optim.pt")
            (d / "state.json").write_text(json.dumps(state))

    # ------------------------------------------------------------------ main loop
    print(json.dumps({"args": vars(args), "pool": len(pool), "eval": len(eval_recs)}), flush=True)
    if state["iter"] == 0 and eval_recs:
        evaluate(0)
    buffer = collections.deque(maxlen=args.buffer_iters)
    while state["iter"] < args.iters:
        if args.stop_at and time.time() + args.iter_budget_s > args.stop_at:
            print("stopping before the walltime", flush=True)
            break
        it_no = state["iter"] + 1
        t_it = time.time()
        recs = [pool[(state["pool_pos"] + i) % len(pool)] for i in range(args.prompts_per_iter)]
        state["pool_pos"] = (state["pool_pos"] + args.prompts_per_iter) % len(pool)
        eps, sstats = self_play(recs)
        moves, found = [], []
        v_root, z_root = [], []
        with open(out / "selfplay" / f"iter_{it_no:04d}.jsonl", "w") as f:
            for ep in eps:
                for m in ep.moves:
                    m["z"] = ep.z
                    moves.append(m)
                    if m["v"] is not None:
                        v_root.append(m["v"]); z_root.append(ep.z)
                if ep.status == "found" and ep.z >= 1:
                    fn = ep.final_node
                    tail = tok("</proof>\nAnswer: " + _ANS.match(fn.line).group(1), add_special_tokens=False)["input_ids"]
                    found.append({"ids": fn.ids + tail + [im_end], "n_prompt": len(ep.prompt_ids)})
                f.write(json.dumps({"id": ep.rec["id"], "bench": ep.rec["bench"], "status": ep.status, "z": ep.z,
                                    "expansions": ep.expansions, "n_moves": len(ep.moves), "gen_tokens": ep.gen_tokens,
                                    "fail": dict(ep.fail), "text": ep.result,
                                    "moves": [{"prefix_tokens": len(m["pre"]), "v": m["v"], "root_q": m["root_q"],
                                               "chosen": m["chosen"],
                                               "children": [{k: c[k] for k in ("line", "N", "q", "terminal", "term",
                                                                                "dead", "prior")} for c in m["children"]]}
                                              for m in ep.moves]}) + "\n")
        buffer.append((moves, found))
        pool_moves = [m for mv, _ in buffer for m in mv]
        pool_found = [e for _, fd in buffer for e in fd]
        tr_moves = rng.sample(pool_moves, min(args.train_moves, len(pool_moves)))
        n_f = min(len(pool_found), max(1, int(args.train_moves * len(pool_found) / max(1, len(pool_moves)))))
        tr_found = rng.sample(pool_found, n_f) if pool_found else []
        llm.sleep(level=2)
        torch.cuda.empty_cache()
        tt = time.time()
        tstats = train_pass(tr_moves, tr_found)
        tt = time.time() - tt
        ts = time.time()
        sync_weights()
        ts = time.time() - ts
        state["iter"] = it_no
        st = collections.Counter(ep.status for ep in eps)
        by = collections.defaultdict(list)
        for ep in eps:
            by[ep.rec["bench"]].append(ep.z)
        met = {"iter": it_no, "n": len(eps), "z": sum(ep.z for ep in eps) / len(eps),
               "found_any": st.get("found", 0) / len(eps), "status": dict(st),
               "z_bench": {b: sum(v) / len(v) for b, v in by.items()},
               "moves": len(moves), "found_correct": len(found),
               "value_auc_root_vs_z": auc(v_root, z_root),
               "mean_expansions": sum(ep.expansions for ep in eps) / len(eps),
               "mean_gen_tokens": sum(ep.gen_tokens for ep in eps) / len(eps),
               **sstats, **tstats, "train_s": tt, "sync_s": ts, "iter_s": time.time() - t_it,
               "gpu_max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(met) + "\n")
        print("[iter]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()
                                    if k not in ("z_bench",)}), flush=True)
        if eval_recs and it_no % args.eval_every == 0:
            evaluate(it_no)
        if it_no % args.save_every == 0:
            save(out / f"ckpt-{it_no:04d}", full=False)
            save(latest, full=True)
    save(latest, full=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
