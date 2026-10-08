#!/usr/bin/env python3
"""Token-level branching of the formal proof policy under the rlvl.Guide legal-token mask, 2026-10-08.

Question: if MCTS actions are tokens (not lines), how many real decisions does a proof contain? The guide gives the
semantically legal next tokens after any prefix (labels, in-scope cites, prompt quotes, lib names, EOS only after
`ans`). Over the 248k Qwen vocabulary that set is large (new identifiers, numbers, quote pieces), so the effective
branching is set by the policy restricted to the mask. For self-play proofs (found and failed) of the r12 run, one
teacher-forced forward pass per proof gives, at every proof token:
  legal     = |mask|
  illegal   = policy mass outside the mask (what masking removes)
  top1      = largest masked-renormalised probability
  eff       = exp(entropy) of the masked policy (effective branching factor)
  n05       = tokens with masked prob >= 0.05
A "decision point" is a position with top1 < 0.9 (or eff >= 1.5). Positions are bucketed by where they are in the
line (line start = label/keyword, formula, rule name after ';', cites).
Writes analysis/token_branching.json and reports/figures/token_branching.{png,pdf}.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import rlvl  # noqa: E402
from eval_formal_vllm import TAG_USER  # noqa: E402
from formal_chat_format import render_prompt  # noqa: E402

AZ = "/vol/tmp2/laitenbf/rlvl_data/az"
E6 = "/vol/tmp2/laitenbf/rlvl_data/formal_mixture_sft_20260925/qwen35_2b_lc_libext_e6_lr5em6_seed3407/final"
MODELS = {"e6 SFT (AZ init)": E6, "r12 AZ iter 200": f"{AZ}/online_e6_r12_gumbel_exit_dp2/latest"}
OPEN = "<proof>\n"


def kind(line: str) -> str:
    if line == "":
        return "line start"
    if ";" not in line:
        return "formula"
    after = line.split(";", 1)[1]
    return "rule name" if after.strip() == "" or " " not in after.strip() else "cites / quote"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--selfplay", default=f"{AZ}/online_e6_r12_gumbel_exit_dp2/selfplay/iter_01[89]*.jsonl")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E6)
    voc = rlvl.vocab_of(tok)
    pool = {}
    for l in open(f"{AZ}/online_pool_gatematched/pool.jsonl"):
        r = json.loads(l)
        pool[r["id"]] = r
    recs = {"found": [], "failed": []}
    for f in sorted(glob.glob(args.selfplay)):
        for l in open(f):
            r = json.loads(l)
            if r["id"] in pool:
                recs["found" if r["status"] == "found" else "failed"].append(r)
    recs = [r for k in recs for r in recs[k][: args.n // 2]]

    # Guide masks per proof position (model independent)
    data = []
    for r in recs:
        body = r["text"][len(OPEN):].split("</proof>")[0]
        ids = tok.encode(body, add_special_tokens=False)
        g = rlvl.Guide(pool[r["id"]]["prompt"], voc)
        masks, kinds = [], []
        for i, x in enumerate(ids):
            masks.append(np.frombuffer(g.mask_bytes(), dtype="<u8").copy())
            kinds.append(kind(tok.decode(ids[:i]).split("\n")[-1]))
            if not g.advance(x):
                break
        n = len(masks)
        pre = tok(render_prompt(tok, f"{TAG_USER}\n{pool[r['id']]['prompt']}") + OPEN, add_special_tokens=False)["input_ids"]
        data.append({"id": r["id"], "found": r["status"] == "found", "pre": pre, "ids": ids[:n], "masks": masks,
                     "kinds": kinds})
    V = len(tok)
    bit = torch.arange(64, device="cuda")

    res = {}
    for mname, mpath in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16).cuda().eval()
        pos = []
        for d in data:
            if not d["ids"]:
                continue
            x = torch.tensor([d["pre"] + d["ids"]], device="cuda")
            with torch.no_grad():
                lg = model(x).logits[0, len(d["pre"]) - 1: len(d["pre"]) - 1 + len(d["ids"])].float()
            W = torch.from_numpy(np.stack(d["masks"]).view(np.int64)).cuda()
            mb = ((W.unsqueeze(-1) >> bit) & 1).bool().flatten(1)
            m = torch.zeros_like(lg, dtype=torch.bool)
            w = min(mb.shape[1], lg.shape[1], V)
            m[:, :w] = mb[:, :w]
            p = lg.log_softmax(-1).exp()
            illegal = 1 - (p * m).sum(-1)
            q = torch.where(m, lg, torch.tensor(-1e30, device="cuda")).log_softmax(-1).exp()
            ent = -(q * q.clamp_min(1e-30).log()).sum(-1)
            top1 = q.max(-1).values
            n05 = (q >= .05).sum(-1)
            legal = m.sum(-1)
            chosen = q[torch.arange(len(d["ids"])), torch.tensor(d["ids"], device="cuda")]
            for i in range(len(d["ids"])):
                pos.append({"found": d["found"], "kind": d["kinds"][i], "legal": int(legal[i]),
                            "illegal": float(illegal[i]), "top1": float(top1[i]), "eff": float(ent[i].exp()),
                            "n05": int(n05[i]), "chosen": float(chosen[i]), "proof": d["id"]})
        del model
        torch.cuda.empty_cache()
        res[mname] = pos

    out = {}
    for mname, pos in res.items():
        per = collections.defaultdict(lambda: [0, 0, 0])
        for p in pos:
            per[p["proof"]][0] += 1
            per[p["proof"]][1] += p["top1"] < .9
            per[p["proof"]][2] += p["eff"] >= 1.5
        per = np.array(list(per.values()))
        bykind = {}
        for k in ("line start", "formula", "rule name", "cites / quote"):
            ps = [p for p in pos if p["kind"] == k]
            if ps:
                bykind[k] = {"n": len(ps), "share": len(ps) / len(pos),
                             "legal_median": float(np.median([p["legal"] for p in ps])),
                             "eff_mean": float(np.mean([p["eff"] for p in ps])),
                             "decision_frac": float(np.mean([p["top1"] < .9 for p in ps])),
                             "illegal_mass_mean": float(np.mean([p["illegal"] for p in ps]))}
        out[mname] = {
            "proofs": len(per), "tokens_per_proof": float(per[:, 0].mean()),
            "decisions_top1_lt_0.9_per_proof": float(per[:, 1].mean()),
            "decisions_eff_ge_1.5_per_proof": float(per[:, 2].mean()),
            "legal_median": float(np.median([p["legal"] for p in pos])),
            "forced_frac_legal_eq_1": float(np.mean([p["legal"] == 1 for p in pos])),
            "eff_mean": float(np.mean([p["eff"] for p in pos])),
            "eff_p99": float(np.quantile([p["eff"] for p in pos], .99)),
            "n05_mean": float(np.mean([p["n05"] for p in pos])),
            "illegal_mass_mean": float(np.mean([p["illegal"] for p in pos])),
            "illegal_mass_p99": float(np.quantile([p["illegal"] for p in pos], .99)),
            "by_kind": bykind,
            "found_vs_failed_decisions": {
                lab: float(np.mean([p["top1"] < .9 for p in pos if p["found"] == f])) for lab, f in
                (("found", True), ("failed", False))}}
    (REPO / "analysis/token_branching.json").write_text(json.dumps(out, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    allpos = next(iter(res.values()))
    ax[0].hist(np.log10([p["legal"] for p in allpos]), bins=50, color="gray")
    ax[0].set_xlabel("log10 |legal tokens| (rlvl.Guide mask)")
    ax[0].set_title("syntactic/semantic legal set per proof token")
    for i, (mname, pos) in enumerate(res.items()):
        c = f"C{i}"
        eff = np.array([p["eff"] for p in pos])
        ax[1].hist(np.clip(eff, 1, 8), bins=np.linspace(1, 8, 57), histtype="step", color=c, label=mname, log=True)
        kinds = list(out[mname]["by_kind"])
        ax[2].bar(np.arange(len(kinds)) + .4 * i, [out[mname]["by_kind"][k]["decision_frac"] for k in kinds], .4,
                  color=c, label=mname)
        ax[2].set_xticks(np.arange(len(kinds)) + .2, kinds)
        per = collections.Counter(p["proof"] for p in pos if p["top1"] < .9)
        ax[3].hist([per.get(pid, 0) for pid in {p["proof"] for p in pos}], bins=range(0, 40), histtype="step", color=c,
                   label=mname)
    ax[1].set_xlabel("effective branching exp(H) of masked policy (clipped at 8)")
    ax[1].set_title("policy branching per token")
    ax[2].set_title("share of tokens that are decisions (top1 < 0.9)")
    ax[3].set_xlabel("decision points per proof (top1 < 0.9)")
    ax[3].set_title("decisions per proof")
    for a in ax:
        a.grid(alpha=.3)
    for a in ax[1:]:
        a.legend(fontsize=8)
    fig.suptitle("Token-level action space: legal mask vs masked-policy branching (r12 self-play proofs)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(REPO / f"reports/figures/token_branching.{ext}", dpi=130)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())
