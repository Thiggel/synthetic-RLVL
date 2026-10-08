#!/usr/bin/env python
"""Stage 3: online AlphaZero. MCTS self-play and joint policy (NTP) + value-head training, one model, one GPU.

docs/research_plan.md, Stage 3 (2026-10-05). The search of scripts/az/mcts_decode.py runs on training prompts and
the model learns from it online, AlphaZero style:

  model     the policy LM (init: an EI SFT checkpoint) plus a linear value head on its last hidden state, the same
            backbone for both (v(s) = sigmoid(w . h_last(s) + b)). bf16 on the GPU, fp32 master weights + AdamW on the CPU.
  search    line-level sampled PUCT (mcts_decode.py): K lines per expansion from vLLM (colocated, sleep mode, weights
            synced after every update), checker pruning (precheck / Stage-2 hardening / rlvl strict), leaf value =
            the live value head, terminal value = the gold reward cvf (correct * valid * premise numbers stated) as in
            AlphaZero where the game result is known in self-play (--reward c_cvf: correct * (0.5 + 0.5 * valid *
            prem_ok) over a relaxed tree that keeps checker-invalid lines, mcts_decode.HARD), Dirichlet noise on root priors, moves sampled by
            visits^(1/move_temp). An episode ends with an accepted `ans` line (z = its gold cvf) or out of budget (z=0).
  targets   per committed move: the visit distribution over the legal sampled lines (policy target, cross-entropy on
            the line log-probability), value targets for the move's state (z, or (z + root Q)/2 with --value-target
            mix), for its children (a terminal: gold cvf; a dead line: 0; a child visited >= 2: its Q) - the
            correctness signal for partial proofs; plus NTP on every found correct proof (incl. "</proof>\\nAnswer").
  train     after every self-play batch: one pass over --train-moves moves sampled from the last --buffer-iters
            batches, AdamW (backbone --lr, head --head-lr, clipped separately), then the new weights go to vLLM. Only
            --value-backbone-scale of the value gradient reaches the backbone: at 1.0 (run r1) the backbone bent to fit
            the near-frozen head and the policy collapsed (gate valid .39 -> .15 in 12 iterations).

Evaluation every --eval-every iterations on the 300-item gate subset (greedy, the same vLLM): valid_prem / cvf /
correct, and the value head's AUC for cvf at the end of the proof and at its middle line (held-out prompts).
Data parallel: under torchrun (WORLD_SIZE > 1) every rank runs its own vLLM (external_launcher, tp 1) and searches
its slice of the iteration's prompts; each rank trains on moves from its own buffer (--train-moves and --moves-per-step
are global and split evenly), gradients are averaged over the ranks before the clip, and every rank takes the same
CPU AdamW step, so the weights stay identical. Rank 0 writes metrics/evals/checkpoints; self-play files are per rank.
Outputs in --out-dir: metrics.jsonl (per iteration), evals.jsonl, selfplay/iter_XXXX.jsonl (episodes), ckpt-XXXX/
(bf16 model + value_head.pt in the probe format of value_probe.py), latest/ (fp32 + optimizer, for --resume).
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import os
import random
import socket
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import rlvl  # noqa: E402
from eval_formal_bench_vllm import fit_prompts, score  # noqa: E402
from eval_formal_vllm import TAG_USER  # noqa: E402
from formal_rewards import components  # noqa: E402
from guided_decode import _ANS, OPEN, harden, precheck, verdict  # noqa: E402
from mcts_decode import HARD, TNode, backup, c_cvf, mark_dead, puct_child, relaxed  # noqa: E402

LINE_NORM = 32.0  # token sums of line log-probs are divided by this (a typical proof line), for a loss of O(1)


class Node(TNode):
    # token ids of prompt + OPEN + lines; of this node's own line; its sampled log-prob (vLLM, the search policy);
    # the illegal lines sampled at its expansions [(lids, lp)]; Gumbel noise as a root child; N at the start of the
    # current move (root children); a leaf awaiting expansion this round
    __slots__ = ("ids", "lids", "lp", "bad", "g", "n0", "pend")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lp, self.bad, self.g, self.n0, self.pend = None, [], None, 0, False


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
        self.pend = []  # [(leaf, path)] awaiting expansion this round (virtual loss applied)
        self.final_node = None
        self.full, self.cur_sims = True, 0  # playout cap randomization: this move's search size


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
    ap.add_argument("--reward", choices=("cvf", "c_cvf"), default="cvf",
                    help="terminal value: cvf over the checker-pruned tree, or c_cvf = correct*(0.5+0.5*valid*prem_ok) over "
                         "a relaxed tree (invalid lines stay; mcts_decode.HARD). NTP is on z >= 1 (valid correct) either way")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--search", choices=("puct", "gumbel"), default="puct",
                    help="gumbel: Gumbel AlphaZero (Danihelka et al. 2022): Gumbel-top-m + sequential halving at the root, "
                         "deterministic improved-policy selection inside, the move = argmax g + logits + sigma(completed Q); "
                         "no Dirichlet noise. Use with --policy-target cq")
    ap.add_argument("--gumbel-m", type=int, default=8, help="lines considered at the root (sequential halving)")
    ap.add_argument("--cheap-sims", type=int, default=None,
                    help="playout cap randomization (KataGo, Wu 2019): a move gets --sims with prob. --full-frac, else "
                         "--cheap-sims; only full moves give policy targets (all give value targets)")
    ap.add_argument("--full-frac", type=float, default=0.25)
    ap.add_argument("--leaves-per-round", type=int, default=1,
                    help="leaves expanded per episode per vLLM call (virtual loss 0 on the pending paths)")
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
    ap.add_argument("--policy-target", choices=("visits", "cq"), default="visits",
                    help="visits: N(a)/sum N (r1-r4: with 8 sims the visits are flatter than the prior, the policy lost "
                         "entropy control and correctness); cq: prior(a) * exp(sigma(completed Q(a))), Gumbel-AZ style")
    ap.add_argument("--cq-scale", type=float, default=0.1, help="sigma(q) = cq_scale * (cq_visit + max N) * q")
    ap.add_argument("--cq-visit", type=float, default=50.0)
    ap.add_argument("--policy-loss", choices=("full", "subset"), default="full",
                    help="full: -sum_a w_a log pi(a) (pushes all sampled lines up: self-distillation, r4/r5 lost entropy "
                         "and correctness); subset: Sampled-MuZero style cross-entropy against pi renormalized over the "
                         "move's sampled lines (legal children + up to --max-bad illegal ones, target 0), gradient "
                         "sum_a (pi~(a) - w_a) grad log pi(a): zero when the search adds no information")
    ap.add_argument("--max-bad", type=int, default=4, help="illegal sampled lines per move in the subset loss")
    ap.add_argument("--ntp-coef", type=float, default=0.25)
    ap.add_argument("--neg-scale", type=float, default=1.0,
                    help="subset loss: scale of the negative (w - pi) coefficients; 0 keeps only the push-ups")
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--value-target", choices=("z", "mix"), default="mix")
    ap.add_argument("--min-visits-q", type=int, default=2, help="children with >= this many visits get Q as value target")
    ap.add_argument("--train-moves", type=int, default=1024, help="moves per training pass (sampled from the buffer)")
    ap.add_argument("--moves-per-step", type=int, default=64, help="moves per optimizer step")
    ap.add_argument("--buffer-iters", type=int, default=2)
    ap.add_argument("--tok-budget", type=int, default=12288, help="padded tokens per micro-batch")
    ap.add_argument("--value-batch", type=int, default=16)
    ap.add_argument("--value-tok-budget", type=int, default=32768, help="padded tokens per value-head forward")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="backbone clip")
    ap.add_argument("--max-head-grad-norm", type=float, default=1.0, help="value head clip (separate from the backbone)")
    ap.add_argument("--value-backbone-scale", type=float, default=0.1,
                    help="fraction of the value-loss gradient that reaches the backbone (1.0 = r1, which collapsed the policy)")
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--gpu-mem", type=float, default=0.28)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--fused-adam", action="store_true", help="fused CPU AdamW (faster step)")
    ap.add_argument("--stop-at", type=float, default=None, help="unix time: save latest/ and exit before it")
    ap.add_argument("--iter-budget-s", type=float, default=3600, help="expected seconds per iteration (for --stop-at)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.max_new_tokens = args.max_proof_tokens
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    rng = random.Random(args.seed + 1000 * rank)
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

    import torch.distributed as dist
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")

    def agree_max(x):  # the same decision / count on every rank
        if world == 1:
            return x
        t = torch.tensor([float(x)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return type(x)(t.item())

    def gather(obj):
        if world == 1:
            return [obj]
        res = [None] * world
        dist.all_gather_object(res, obj)
        return res

    torch.manual_seed(args.seed + rank)
    np_rng = np.random.default_rng(args.seed + 1000 * rank)
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
    opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0,
                            **({"fused": True} if args.fused_adam else {"foreach": True}))
    opt_head = torch.optim.AdamW(head.parameters(), lr=args.head_lr, betas=(0.9, 0.99), weight_decay=0.0)
    if src != args.model and (latest / "optim.pt").exists():
        o = torch.load(latest / "optim.pt", map_location="cpu")
        opt.load_state_dict(o["backbone"]); opt_head.load_state_dict(o["head"])
    llm = LLM(model=src, tokenizer=args.model, dtype="bfloat16", seed=args.seed + rank, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, max_num_seqs=args.max_num_seqs,
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
    need_lp = args.search == "gumbel" or args.policy_loss == "subset"  # line log-probs from vLLM
    sp_line = SamplingParams(n=args.k, temperature=args.temperature, max_tokens=args.max_line_tokens, stop=["\n"],
                             include_stop_str_in_output=True, stop_token_ids=stop_ids, skip_special_tokens=True,
                             **({"logprobs": 0} if need_lp else {}))
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
        batches, cur = [], []
        for i in order:  # at most --value-batch sequences and --value-tok-budget padded tokens per forward
            if cur and (len(cur) >= args.value_batch or (len(cur) + 1) * len(seqs[i]) > args.value_tok_budget):
                batches.append(cur); cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        for idx in batches:
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
        my_recs = eval_recs[rank::world]
        pids = [prompt_ids(r) for r in my_recs]
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_proof_tokens, stop_token_ids=stop_ids,
                            skip_special_tokens=True)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sp, use_tqdm=False)
        rows, vq_end, vq_mid = [], [], []
        for rec, p, o in zip(my_recs, pids, outs):
            text = o.outputs[0].text
            c = components(rec, text)
            vp = float(c["valid"]) * float(c["prem_ok"])
            rows.append({"id": rec["id"], "bench": rec["bench"], "correct": float(c["correct"]), "valid": float(c["valid"]),
                         "valid_prem": vp, "cvf": vp * float(c["correct"]), "c_cvf": c_cvf(c),
                         "score_correct": float(score(rec, text).get("correct", 0))})
            gen = list(o.outputs[0].token_ids)
            vq_end.append((p + gen)[:args.max_model_len])
            nl = [i for i, t_ in enumerate(gen) if is_nl(t_)]
            vq_mid.append(p + gen[:nl[len(nl) // 2] + 1] if nl else p + gen[:1])
        v_end, v_mid = values(vq_end), values(vq_mid)
        parts = gather((rows, v_end, v_mid))
        if rank != 0:
            return None
        rows = [r for p_ in parts for r in p_[0]]
        v_end = [v for p_ in parts for v in p_[1]]
        v_mid = [v for p_ in parts for v in p_[2]]
        lab = [r["cvf"] if args.reward == "cvf" else r["c_cvf"] for r in rows]  # auc: label >= 0.5
        res = {"iter": it_no, "n": len(rows), **{m: sum(r[m] for r in rows) / len(rows)
                                                 for m in ("valid", "valid_prem", "cvf", "c_cvf", "correct", "score_correct")},
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
        if len(live) > 1 and args.dirichlet_frac > 0 and args.search == "puct":
            d = np_rng.dirichlet([args.dirichlet_alpha] * len(live))
            for c, n in zip(live, d):
                c.prior = (1 - args.dirichlet_frac) * c.prior + args.dirichlet_frac * float(n)

    # ---- Gumbel AlphaZero (Danihelka et al. 2022; mctx gumbel_muzero_policy), over the sampled legal lines
    def logits(live):  # log pi renormalized over the live children (exact line log-probs; else the sample frequencies)
        lg = [c.lp if c.lp is not None else math.log(max(c.prior, 1e-9)) for c in live]
        mx = max(lg)
        return [x - mx for x in lg]

    def sigma_cq(node, live):  # sigma(completed Q): visited -> Q, unvisited -> the mixed value of the node
        vis = [c for c in live if c.N]
        v0 = node.v if node.v is not None else (node.q() or 0.0)
        if vis:
            sp = sum(math.exp(x) for x, c in zip(logits(live), live) if c.N)
            wq = sum(math.exp(x) * c.q() for x, c in zip(logits(live), live) if c.N) / max(sp, 1e-12)
            sn = sum(c.N for c in vis)
            v0 = (v0 + sn * wq) / (1 + sn)
        beta = args.cq_scale * (args.cq_visit + max([c.N for c in node.children] + [0]))
        return [beta * (c.q() if c.N else v0) for c in live]

    @functools.lru_cache(maxsize=None)
    def halving_seq(m, n):  # mctx seq_halving.get_sequence_of_considered_visits
        if m <= 1:
            return tuple(range(n))
        log2m, seq, visits, k = math.ceil(math.log2(m)), [], [0] * m, m
        while len(seq) < n:
            for _ in range(max(1, n // (log2m * k))):
                seq.extend(visits[:k])
                for i in range(k):
                    visits[i] += 1
            k = max(2, k // 2)
        return tuple(seq[:n])

    def gumbel_scores(node, live):
        for c in live:
            if c.g is None:
                c.g = float(np_rng.gumbel())
        return [c.g + l + s for c, l, s in zip(live, logits(live), sigma_cq(node, live))]

    def sims_done(ep):
        if args.search == "gumbel":
            return sum(c.N - c.n0 for c in ep.root.children)
        return ep.root.N - ep.move_start

    def select_child(ep, node):
        if args.search == "puct":
            return puct_child(node, args.c_puct)
        live = [c for c in node.children if not c.dead]
        if not live:
            return None
        if node is ep.root:  # sequential halving: the considered lines are those with the scheduled visit count
            sc = gumbel_scores(node, live)
            seq = halving_seq(min(args.gumbel_m, len(live), max(1, ep.cur_sims)), max(1, ep.cur_sims))
            i = sims_done(ep)
            tgt = seq[i] if i < len(seq) else None
            idx = [j for j, c in enumerate(live) if c.N - c.n0 == tgt]
            if not idx:
                lo = min(c.N - c.n0 for c in live)
                idx = [j for j, c in enumerate(live) if c.N - c.n0 == lo]
            return live[max(idx, key=lambda j: sc[j])]
        lg = [l + s for l, s in zip(logits(live), sigma_cq(node, live))]  # improved policy - visit share
        mx = max(lg)
        pi = [math.exp(x - mx) for x in lg]
        z, tot = sum(pi), sum(c.N for c in live)
        return max(zip(live, pi), key=lambda t: t[1] / z - t[0].N / (1 + tot))[0]

    def new_move(ep):  # playout cap randomization: full search (policy target) or cheap search (value only)
        ep.full = args.cheap_sims is None or rng.random() < args.full_frac
        ep.cur_sims = args.sims if ep.full else args.cheap_sims
        for c in ep.root.children:
            c.n0 = c.N

    def finish(ep, status, node=None):
        ep.done, ep.status, ep.final_node = True, status, node
        if node is not None:
            ep.result = OPEN + "".join(node.lines) + "</proof>\nAnswer: " + _ANS.match(node.line).group(1)
            ep.z = node.term_value
        else:
            ep.result = OPEN + "".join(ep.root.lines)

    def record_move(ep, root, chosen):
        ep.moves.append({"pre": root.ids, "root_q": root.q(), "root_N": root.N, "v": root.v, "chosen": chosen.line,
                         "full": ep.full, "bad": root.bad[:args.max_bad] if args.policy_loss == "subset" else [],
                         "children": [{"lids": c.lids, "line": c.line, "N": c.N, "q": c.q(), "terminal": c.terminal,
                                       "term": c.term_value, "dead": c.dead, "prior": c.prior, "lp": c.lp}
                                      for c in root.children]})

    def commit(ep):
        root = ep.root
        if args.search == "gumbel":  # among the most visited lines of this move: argmax g + logits + sigma(cq)
            live = [c for c in root.children if not c.dead]
            sc = gumbel_scores(root, live) if live else []
            top = max([c.N - c.n0 for c in live] + [0])
            idx = [j for j, c in enumerate(live) if c.N - c.n0 == top and top > 0]
            ch = live[max(idx, key=lambda j: sc[j])] if idx else None
        else:
            live = [c for c in root.children if c.N and not c.dead]
            ch = None
            if live:
                wts = [c.N ** (1 / args.move_temp) for c in live] if args.move_temp > 0 else None
                ch = rng.choices(live, weights=wts)[0] if wts else max(live, key=lambda c: (c.N, c.q()))
        if ch is None:
            ep.move_start = root.N
            for c in root.children:
                c.n0 = c.N
            return
        record_move(ep, root, ch)
        if ch.terminal:
            finish(ep, "found", ch)
            return
        ep.root = ch
        ep.move_start = ch.N
        new_move(ep)
        if ch.expanded and ep.full:
            noise_root(ch)
        if len(ch.lines) >= args.max_lines:
            finish(ep, "max_lines")

    def self_play(recs):
        eps = [Episode(r, prompt_ids(r), open_ids) for r in recs]
        for ep in eps:
            new_move(ep)
        t0, rounds, t_gen, t_chk, t_val, n_leaves = time.time(), 0, 0.0, 0.0, 0.0, 0
        L = args.leaves_per_round
        while True:
            need = []
            for ep in eps:
                guard = 0
                while not ep.done and len(ep.pend) < L and guard < 64:
                    guard += 1
                    if ep.pend and ep.root.dead:
                        break
                    while ep.root.dead and ep.root.parent is not None:
                        ep.root = ep.root.parent
                        ep.move_start = ep.root.N
                        new_move(ep)
                    if ep.root.dead or (ep.root.expanded and not ep.root.children and ep.root.expansions >= 2):
                        finish(ep, "exhausted")
                        break
                    if sims_done(ep) >= ep.cur_sims and ep.root.children:
                        if ep.pend:  # commit once this round's leaves are backed up
                            break
                        commit(ep)
                        continue
                    node, path = ep.root, [ep.root]
                    while node.expanded and node.children:
                        nxt = select_child(ep, node)
                        if nxt is None:
                            break
                        node = nxt
                        path.append(node)
                    if node.terminal:
                        backup(path, node.term_value)
                        continue
                    if node.pend:
                        break
                    if ep.expansions + len(ep.pend) >= args.max_expansions:
                        if ep.pend:
                            break
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
                    node.pend = True
                    for n in path:  # virtual loss: a pending visit with value 0 steers the next leaf elsewhere
                        n.N += 1
                    ep.pend.append((node, path))
                if ep.pend:
                    need.append(ep)
            if not need:
                break
            rounds += 1
            leaves = [(ep, node, path) for ep in need for node, path in ep.pend]
            n_leaves += len(leaves)
            tg = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=node.ids) for _, node, _ in leaves],
                                [sp_line] * len(leaves), use_tqdm=False)
            t_gen += time.time() - tg
            tc = time.time()
            jobs, cands = [], {}
            for (ep, node, _), o in zip(leaves, outs):
                node.expansions += 1
                ep.expansions += 1
                counts, lids, lps = collections.Counter(), {}, {}
                for s in o.outputs:
                    ep.gen_tokens += len(s.token_ids)
                    line = s.text if s.text.endswith("\n") else s.text + "\n"
                    counts[line] += 1
                    if line not in lids:
                        lids[line] = tok(line, add_special_tokens=False)["input_ids"]
                        lps[line] = s.cumulative_logprob
                cands[id(node)] = (counts, lids, lps)
                known = {ch.line for ch in node.children}
                for line in counts:
                    if line in known:
                        continue
                    ep.n_cands += 1
                    why = precheck(line) or harden(ep.rec["prompt"], node.lines, line, args.max_know)
                    if why and args.reward == "c_cvf" and why not in HARD:
                        ep.fail["soft_" + why] += 1
                        why = None
                    if why:
                        ep.fail[why] += 1
                        if need_lp and lps[line] is not None:
                            node.bad.append((lids[line], lps[line]))
                        continue
                    jobs.append((ep, node, line))
            reps = rlvl.check_batch([j[0].rec["prompt"] for j in jobs], ["".join(j[1].lines) + j[2] for j in jobs],
                                    strict=True)
            legal = collections.defaultdict(list)
            for (ep, node, line), rep in zip(jobs, reps):
                if args.reward == "c_cvf":
                    keep, term, gold, why = relaxed(ep.rec["prompt"], node.lines, line, rep, ep.rec)
                    if keep:
                        legal[id(node)].append((line, term, gold))
                        if why:
                            ep.fail["soft_" + why] += 1
                        continue
                    v = why
                    ep.fail[v] += 1
                    lp_ = cands[id(node)][2][line]
                    if need_lp and lp_ is not None:
                        node.bad.append((cands[id(node)][1][line], lp_))
                    continue
                v = verdict(rep, line)
                gold = 0.0
                if v == "done":
                    m = _ANS.match(line)
                    text = OPEN + "".join(node.lines) + line + "</proof>\nAnswer: " + (m.group(1) if m else "")
                    c = components(ep.rec, text)
                    v = "done" if (m and c["valid"] and c["prem_ok"]) else "s2_invalid"
                    gold = float(c["correct"] * c["valid"] * c["prem_ok"])
                if v in ("ok", "done"):
                    legal[id(node)].append((line, v == "done", gold))
                else:
                    ep.fail[v] += 1
                    lp_ = cands[id(node)][2][line]
                    if need_lp and lp_ is not None:
                        node.bad.append((cands[id(node)][1][line], lp_))
            t_chk += time.time() - tc
            tv = time.time()
            vq, vowners = [], []
            for ep, node, _ in leaves:
                counts, lids, lps = cands[id(node)]
                new = legal.get(id(node), [])
                tot = sum(counts[l] for l, _, _ in new) + sum(counts.get(ch.line, 0) for ch in node.children)
                for ch in node.children:
                    ch.prior = (ch.prior + counts.get(ch.line, 0) / max(1, tot)) / 2
                for line, term, gold in new:
                    ch = Node(node.lines + [line], node, counts[line] / max(1, tot), line)
                    ch.lids = lids[line]
                    ch.lp = lps[line]
                    ch.ids = node.ids + ch.lids
                    if term:
                        ch.terminal, ch.term_value = True, gold
                    node.children.append(ch)
                first = not node.expanded
                node.expanded = True
                if first and node is ep.root and ep.full:
                    noise_root(node)
                if node.v is None:
                    vq.append(node.ids); vowners.append(node)
            if vq:
                for n, v in zip(vowners, values(vq)):
                    n.v = v
            t_val += time.time() - tv
            for ep in need:
                for node, path in ep.pend:
                    node.pend = False
                    for n in path:  # undo the virtual loss before the real backup
                        n.N -= 1
                    if not any(not ch.dead for ch in node.children):
                        if node.expansions >= 2:
                            node.dead = True
                            backup(path, 0.0)
                            mark_dead(node.parent)
                        else:
                            backup(path, 0.0)
                        continue
                    backup(path, node.v)
                ep.pend = []
            if rounds % 8 == 0:
                print(f"  [round {rounds}] leaves {len(leaves)} done {sum(e.done for e in eps)}/{len(eps)} "
                      f"found {sum(e.status == 'found' for e in eps)} gen {t_gen:.0f}s chk {t_chk:.0f}s val {t_val:.0f}s",
                      flush=True)
        return eps, {"rounds": rounds, "leaves": n_leaves, "search_s": time.time() - t0, "gen_s": t_gen,
                     "check_s": t_chk, "value_s": t_val}

    # ------------------------------------------------------------------ training examples from moves / episodes
    def examples(moves, eps_found):
        """Each example: token ids, policy ranges [(start, end, weight)] (log-prob of ids[start:end] given the rest),
        value points [(position, target)]. Weights are already normalized per move / per found episode."""
        ex = []
        for m in moves:
            pre = m["pre"]
            ch = m["children"]
            totN = sum(c["N"] for c in ch if not c["dead"])
            subset = args.policy_loss == "subset" and all(c.get("lp") is not None for c in ch)
            if args.policy_target == "cq":  # completed Q: unvisited lines get the root value; dead lines weight 0
                vmix = m["root_q"] if m["root_q"] is not None else m["v"]
                beta = args.cq_scale * (args.cq_visit + max([c["N"] for c in ch] + [0]))
                lg = [(c["lp"] if subset else math.log(c["prior"])) + beta * (c["q"] if c["N"] and c["q"] is not None else vmix)
                      if not c["dead"] and (subset or c["prior"] > 0) else None for c in ch]
                mx = max([x for x in lg if x is not None], default=0.0)
                ws = [math.exp(x - mx) if x is not None else 0.0 for x in lg]
                ws = [x / sum(ws) for x in ws] if sum(ws) else ws
            else:
                ws = [c["N"] / totN if totN and c["N"] and not c["dead"] else 0.0 for c in ch]
            if not m.get("full", True):  # playout cap: cheap searches give no policy target
                ws = [0.0] * len(ch)
            bad = m.get("bad", []) if subset and any(ws) else []
            if subset:  # coefficient on log pi(line): w - pi renormalized over the sampled lines (search-time pi)
                lps = [c["lp"] for c in ch] + [b[1] for b in bad]
                mx = max(lps)
                pt = [math.exp(x - mx) for x in lps]
                pt = [x / sum(pt) for x in pt]
                coef = [w - q for w, q in zip(ws, pt)] + [-q for q in pt[len(ch):]] if any(ws) else [0.0] * len(lps)
                coef = [x if x > 0 else args.neg_scale * x for x in coef]  # --neg-scale 0: positive part only
            else:
                coef = ws
            s_tgt = m["z"] if args.value_target == "z" or m["root_q"] is None else (m["z"] + m["root_q"]) / 2
            first = True
            for ci, c in enumerate(ch):
                w = coef[ci]
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
                if w != 0 and len(ids) > len(pre):
                    e["pol"].append((len(pre), len(ids), args.policy_coef * w / LINE_NORM))
                if first:
                    e["val"].append((len(pre) - 1, s_tgt))
                    first = False
                if vt is not None and len(ids) > len(pre):
                    e["val"].append((len(ids) - 1, vt))
                ex.append(e)
            for (lids, _), w in zip(bad, coef[len(ch):]):  # illegal sampled lines: target 0
                ids = (pre + lids)[:args.max_model_len]
                if w != 0 and len(ids) > len(pre):
                    ex.append({"ids": ids, "pol": [(len(pre), len(ids), args.policy_coef * w / LINE_NORM)], "val": []})
        for ep in eps_found:
            full = ep["ids"][:args.max_model_len]
            ex.append({"ids": full, "pol": [(ep["n_prompt"], len(full), args.ntp_coef / LINE_NORM)], "val": []})
        return ex

    def train_pass(moves, eps_found):
        model.train()
        rng.shuffle(moves)
        mps = max(1, args.moves_per_step // world)  # per rank
        n_steps = agree_max(max(1, math.ceil(len(moves) / mps)))
        stats = collections.defaultdict(float)
        for s in range(n_steps):
            mv = moves[s * mps:(s + 1) * mps]
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
                    if args.value_backbone_scale != 1.0:  # the head sees h, the backbone gets scale * the value gradient
                        hv = hv * args.value_backbone_scale + hv.detach() * (1 - args.value_backbone_scale)
                    logit = head(hv).squeeze(-1)
                    tv = torch.tensor(vt, device=h.device, dtype=torch.float32)
                    lv = torch.nn.functional.binary_cross_entropy_with_logits(logit, tv, reduction="sum") / n_val
                    loss = loss + args.value_coef * lv
                    stats["value_loss"] += lv.item()
                    stats["value_points"] += len(vb)
                    stats["value_abs_err"] += (torch.sigmoid(logit) - tv).abs().sum().item()
                loss.backward()
                del h, loss
            to = time.time()
            if world > 1:  # average over the ranks (a rank without examples this step contributes zeros)
                for q in gpu_params + list(head.parameters()):
                    if q.grad is None:
                        q.grad = torch.zeros_like(q)
                    dist.all_reduce(q.grad, op=dist.ReduceOp.AVG)
            gn = torch.nn.utils.clip_grad_norm_(gpu_params, args.max_grad_norm)
            gnh = torch.nn.utils.clip_grad_norm_(list(head.parameters()), args.max_head_grad_norm)
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
            stats["opt_s"] += time.time() - to
            stats["grad_norm"] += float(gn) / n_steps
            stats["head_grad_norm"] += float(gnh) / n_steps
            stats["opt_steps"] += 1
        torch.cuda.empty_cache()
        stats["value_abs_err"] /= max(1, stats["value_points"])
        for k in ("policy_loss", "value_loss"):  # mean over optimizer steps (logs before 2026-10-05 21:30 hold the sum)
            stats[k] /= max(1, stats["opt_steps"])
        stats["stats_v"] = 2
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
                    "layer": -1, "target": args.reward}, d / "value_head.pt")
        if full:
            torch.save({"backbone": opt.state_dict(), "head": opt_head.state_dict()}, d / "optim.pt")
            (d / "state.json").write_text(json.dumps(state))

    # ------------------------------------------------------------------ main loop
    if rank == 0:
        print(json.dumps({"args": vars(args), "pool": len(pool), "eval": len(eval_recs), "world": world}), flush=True)
    if state["iter"] == 0 and eval_recs:
        evaluate(0)
    buffer = collections.deque(maxlen=args.buffer_iters)
    while state["iter"] < args.iters:
        if agree_max(int(bool(args.stop_at) and time.time() + args.iter_budget_s > args.stop_at)):
            print("stopping before the walltime", flush=True)
            break
        it_no = state["iter"] + 1
        t_it = time.time()
        recs = [pool[(state["pool_pos"] + i) % len(pool)] for i in range(args.prompts_per_iter)]
        state["pool_pos"] = (state["pool_pos"] + args.prompts_per_iter) % len(pool)
        eps, sstats = self_play(recs[rank::world])
        moves, found = [], []
        v_root, z_root = [], []
        with open(out / "selfplay" / (f"iter_{it_no:04d}.jsonl" if world == 1 else f"iter_{it_no:04d}.r{rank}.jsonl"), "w") as f:
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
                                               "chosen": m["chosen"], "full": m["full"], "n_bad": len(m["bad"]),
                                               "children": [{k: c[k] for k in ("line", "N", "q", "terminal", "term",
                                                                                "dead", "prior", "lp")} for c in m["children"]]}
                                              for m in ep.moves]}) + "\n")
        buffer.append((moves, found))
        pool_moves = [m for mv, _ in buffer for m in mv]
        pool_found = [e for _, fd in buffer for e in fd]
        n_tr = max(1, args.train_moves // world)
        tr_moves = rng.sample(pool_moves, min(n_tr, len(pool_moves)))
        n_f = min(len(pool_found), max(1, int(n_tr * len(pool_found) / max(1, len(pool_moves)))))
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
        # metrics over all ranks' episodes
        parts = gather(([{"bench": ep.rec["bench"], "status": ep.status, "z": ep.z, "expansions": ep.expansions,
                          "gen_tokens": ep.gen_tokens} for ep in eps], v_root, z_root, len(moves), len(found), sstats))
        eps = [types.SimpleNamespace(rec={"bench": e["bench"]}, **e) for p_ in parts for e in p_[0]]
        v_root = [v for p_ in parts for v in p_[1]]
        z_root = [v for p_ in parts for v in p_[2]]
        n_moves, n_found = sum(p_[3] for p_ in parts), sum(p_[4] for p_ in parts)
        if world > 1:  # wall-clock: the slowest rank; counts: summed
            sstats = {k: (max if k.endswith("_s") else sum)(p_[5][k] for p_ in parts) for k in sstats}
        st = collections.Counter(ep.status for ep in eps)
        by = collections.defaultdict(list)
        for ep in eps:
            by[ep.rec["bench"]].append(ep.z)
        met = {"iter": it_no, "n": len(eps), "z": sum(ep.z for ep in eps) / len(eps),
               "z_valid_correct": sum(ep.z >= 1 for ep in eps) / len(eps), "z_correct": sum(ep.z >= .5 for ep in eps) / len(eps),
               "found_any": st.get("found", 0) / len(eps), "status": dict(st),
               "z_bench": {b: sum(v) / len(v) for b, v in by.items()},
               "moves": n_moves, "found_correct": n_found, "world": world,
               "value_auc_root_vs_z": auc(v_root, z_root),
               "mean_expansions": sum(ep.expansions for ep in eps) / len(eps),
               "mean_gen_tokens": sum(ep.gen_tokens for ep in eps) / len(eps),
               **sstats, **tstats, "train_s": tt, "sync_s": ts, "iter_s": time.time() - t_it,
               "gpu_max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30}
        if rank == 0:
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(met) + "\n")
            print("[iter]", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()
                                        if k not in ("z_bench",)}), flush=True)
        if eval_recs and it_no % args.eval_every == 0:
            evaluate(it_no)
        if it_no % args.save_every == 0 and rank == 0:
            save(out / f"ckpt-{it_no:04d}", full=False)
            save(latest, full=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        save(latest, full=True)
        print("done", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
