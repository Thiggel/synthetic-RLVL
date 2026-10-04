#!/usr/bin/env python
"""Stage 3, step 1: can a value head on the policy's hidden state predict the terminal reward of a partial proof?

AlphaZero needs v(prompt + proof prefix) ~ E[terminal reward]. Before building the search, this measures how much a
frozen-backbone probe can predict, using rollouts the GRPO runs already logged (`<run>/completions/*.parquet`,
32 prompts x 8 completions per step, each with its checker reward columns `correct`, `valid`, `prem_ok`, `cvf`).

  extract   for a sample of logged completions, run the policy (default e4, the AZ-formal init) once over
            prompt + completion and keep the hidden state at the end of the prompt (depth 0, before `<proof>`) and
            after every proof line ("\\n" inside <proof>...</proof>): last layer and a middle layer.
  probe     logistic regression (torch, full batch, L2) on those features for each target, trained on a
            prompt-disjoint split (hash of the prompt), so test prompts are unseen.
  metrics   AUC overall and by relative depth (0 = prompt only, then quartiles of the proof, 1 = after the last line),
            and the within-prompt AUC: over pairs of completions of the same prompt and step with different outcomes,
            how often the probe ranks the better one higher at the same relative depth. This is what search needs
            (a prompt-only value has within-prompt AUC 0.5 by construction).

Writes <out>/features.pt (cached), analysis/az_value_probe.{md,json}, reports/figures/az_value_probe.{png,pdf}.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
DATA = Path("/vol/tmp2/laitenbf/rlvl_data")
GRPO = DATA / "grpo_formal_20260928"
TARGETS = ("correct", "cvf", "valid")
BUCKETS = ("prompt", "0-25%", "25-50%", "50-75%", "75-100%", "end")


def load_completions(specs: list[str], per_run: int, seed: int) -> pd.DataFrame:
    """specs: RUN[:first_step-last_step]. Samples whole (prompt, step) groups so within-prompt pairs survive."""
    rng = random.Random(seed)
    frames = []
    for spec in specs:
        run, _, rng_s = spec.partition(":")
        lo, hi = (int(x) for x in rng_s.split("-")) if rng_s else (0, 10 ** 9)
        fs = [f for f in sorted(glob.glob(str(GRPO / run / "completions/completions_*.parquet")))
              if lo <= int(Path(f).stem.split("_")[1]) <= hi]
        d = pd.concat([pd.read_parquet(f, columns=["step", "prompt", "completion", *TARGETS, "prem_ok", "truncated"])
                       for f in fs], ignore_index=True)
        d["run"] = run
        groups = list(d.groupby(["step", "prompt"]).groups.values())
        rng.shuffle(groups)
        keep, n = [], 0
        for g in groups:
            if n >= per_run:
                break
            keep.extend(g)
            n += len(g)
        frames.append(d.loc[keep])
        print(f"{run}: {len(fs)} files, kept {n} completions")
    d = pd.concat(frames, ignore_index=True)
    d["cvf"] = d["correct"] * d["valid"] * d["prem_ok"]  # the hardened cvf (logged cvf columns differ across arms)
    d["user"] = d["prompt"].str.extract(r"^user\n(.*)\nassistant\n", expand=False, flags=re.S)
    d = d[d["user"].notna()].reset_index(drop=True)
    d["test"] = d["user"].map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 5 == 0)
    return d


@torch.no_grad()
def extract(d: pd.DataFrame, model_dir: str, mid_layer: int, max_len: int, batch: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from formal_chat_format import render_prompt

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(nl) == 1
    rows = []  # (completion index, depth k, n_lines, feature_last, feature_mid)
    order = sorted(range(len(d)), key=lambda i: len(d.completion[i]))
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        seqs, marks = [], []
        for i in idx:
            p = tok(render_prompt(tok, d.user[i]), add_special_tokens=False)["input_ids"]
            body = d.completion[i].split("</proof>")[0]
            c = tok(body, add_special_tokens=False)["input_ids"]
            ids = (p + c)[:max_len]
            # positions: last prompt token, then every "\n" token of the proof body after the "<proof>" line
            pos = [len(p) - 1] + [len(p) + j for j, t in enumerate(c) if t == nl[0] and len(p) + j < max_len][1:]
            seqs.append(ids)
            marks.append(pos)
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(x)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        out = model(input_ids=x.cuda(), attention_mask=att.cuda(), output_hidden_states=True)
        hs_last, hs_mid = out.hidden_states[-1].float(), out.hidden_states[mid_layer].float()
        for j, (i, pos) in enumerate(zip(idx, marks)):
            n = len(pos) - 1
            for k, q in enumerate(pos):
                rows.append((i, k, n, hs_last[j, q].half().cpu(), hs_mid[j, q].half().cpu()))
        if b // batch % 50 == 0:
            print(f"extract {b}/{len(order)} ({len(rows)} positions)", flush=True)
    return {"i": torch.tensor([r[0] for r in rows]), "k": torch.tensor([r[1] for r in rows]),
            "n": torch.tensor([r[2] for r in rows]), "last": torch.stack([r[3] for r in rows]),
            "mid": torch.stack([r[4] for r in rows])}


def bucket(k: torch.Tensor, n: torch.Tensor) -> np.ndarray:
    frac = k.float() / n.clamp(min=1).float()
    b = np.where(k.numpy() == 0, 0, np.minimum(4, 1 + (frac.numpy() * 4).astype(int)))
    b[(k == n).numpy() & (k > 0).numpy()] = 5
    return b


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y > 0.5], score[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_probe(X: torch.Tensor, y: torch.Tensor, l2: float, steps: int = 300) -> tuple:
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xs = ((X - mu) / sd).cuda()
    w = torch.zeros(X.shape[1], device="cuda", requires_grad=True)
    b0 = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=steps, line_search_fn="strong_wolfe")
    yy = y.float().cuda()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xs @ w + b0, yy) + l2 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (w.detach() / sd.cuda()).cpu(), (b0.detach() - (mu.cuda() / sd.cuda()) @ w.detach()).cpu()


def within_prompt_auc(d, F, score, y_col, mask) -> float:
    """Pairs (a, b) of completions of the same (run, step, prompt) with y_a > y_b, compared at the same bucket."""
    df = pd.DataFrame({"i": F["i"].numpy()[mask], "b": F["bucket"][mask], "s": score})
    df = df.groupby(["i", "b"], as_index=False)["s"].mean()
    df["g"] = (d.run + "|" + d.step.astype(str) + "|" + d.user).to_numpy()[df.i]
    df["y"] = d[y_col].to_numpy()[df.i]
    wins = total = 0.0
    for _, g in df.groupby(["g", "b"]):
        p, n = g.s[g.y > 0.5].to_numpy(), g.s[g.y <= 0.5].to_numpy()
        if len(p) and len(n):
            c = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
            wins += c
            total += len(p) * len(n)
    return float(wins / total) if total else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DATA / "formal_mixture_sft_20260925/qwen35_2b_lc_libext_e4_lr5em6_seed3407/final"))
    ap.add_argument("--runs", nargs="+", default=["2b_e2_G16_cvffmt_overlong_noproof:500-1000",
                                                  "2b_e3_G17_cvffmt_overlong_noproof"])
    ap.add_argument("--per-run", type=int, default=6000)
    ap.add_argument("--mid-layer", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--out", default=str(DATA / "az/value_probe_e4"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_completions(args.runs, args.per_run, args.seed)
    fpath = out / "features.pt"
    if fpath.exists():
        F = torch.load(fpath)
        assert len(F["meta"]) == len(d), "cached features belong to a different sample"
    else:
        F = extract(d, args.model, args.mid_layer, args.max_len, args.batch)
        F["meta"] = d[["run", "step", "user", "completion"]].to_dict("records")
        torch.save(F, fpath)
    F["bucket"] = bucket(F["k"], F["n"])
    test = d.test.to_numpy()[F["i"].numpy()]
    res = {"model": args.model, "runs": args.runs, "n_completions": len(d), "n_positions": int(len(F["i"])),
           "n_test_prompts": int(d[d.test].user.nunique()), "n_train_prompts": int(d[~d.test].user.nunique()),
           "base_rates": {t: float(d[t].mean()) for t in TARGETS}, "probes": {}}
    for feat in ("last", "mid"):
        X = F[feat].float()
        for t in TARGETS:
            y = torch.tensor(d[t].to_numpy()[F["i"].numpy()] > 0.5)
            w, b0 = fit_probe(X[~test], y[~test], args.l2)
            s = (X[test] @ w + b0).numpy()
            yt = y[test].numpy().astype(float)
            r = {"auc": auc(s, yt), "by_bucket": {}, "within_prompt": {}}
            for bi, bn in enumerate(BUCKETS):
                m = F["bucket"][test] == bi
                r["by_bucket"][bn] = auc(s[m], yt[m])
            wp = within_prompt_auc(d, F, s, t, test)
            sub = F["bucket"][test]
            for bi, bn in enumerate(BUCKETS):
                m = sub == bi
                mask = np.zeros(len(test), bool)
                mask[np.flatnonzero(test)[m]] = True
                r["within_prompt"][bn] = within_prompt_auc(d, F, s[m], t, mask)
            r["within_prompt_all"] = wp
            res["probes"][f"{feat}:{t}"] = r
            print(feat, t, json.dumps({k: (round(v, 3) if isinstance(v, float) else
                                           {a: round(c, 3) for a, c in v.items()}) for k, v in r.items()}), flush=True)
            if feat == "last":
                torch.save({"w": w, "b": b0, "feature": feat, "layer": -1, "target": t}, out / f"probe_{t}.pt")
    name = REPO / "analysis/az_value_probe"
    name.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# Stage 3: frozen-backbone value probe on e4", "",
             f"Model `{args.model}`; rollouts from {', '.join(f'`{r}`' for r in args.runs)}; {len(d)} completions, "
             f"{res['n_positions']} prefix positions; prompt-disjoint split ({res['n_train_prompts']} train / "
             f"{res['n_test_prompts']} test prompts). Base rates: "
             + ", ".join(f"{t} {v:.3f}" for t, v in res["base_rates"].items()) + ".", "",
             "AUC of a logistic probe at the end of the prompt (`prompt`), after proof lines by relative depth, and "
             "after the last line (`end`). *Within-prompt* AUC compares completions of the same prompt and step at the "
             "same depth bucket (0.5 = no help for search).", "",
             "| features:target | AUC | " + " | ".join(BUCKETS) + " | within-prompt | " +
             " | ".join(f"wp {b}" for b in BUCKETS[1:]) + " |", "|---|" + "---:|" * (2 * len(BUCKETS) + 1)]
    for k, r in res["probes"].items():
        lines.append(f"| {k} | {r['auc']:.3f} | " + " | ".join(f"{r['by_bucket'][b]:.3f}" for b in BUCKETS) +
                     f" | {r['within_prompt_all']:.3f} | " +
                     " | ".join(f"{r['within_prompt'][b]:.3f}" for b in BUCKETS[1:]) + " |")
    name.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, r in res["probes"].items():
        ls = "-" if k.startswith("last") else "--"
        axes[0].plot(BUCKETS, [r["by_bucket"][b] for b in BUCKETS], ls, marker="o", label=k)
        axes[1].plot(BUCKETS[1:], [r["within_prompt"][b] for b in BUCKETS[1:]], ls, marker="o", label=k)
    for ax, ttl in zip(axes, ("AUC across test prompts", "within-prompt AUC (what search needs)")):
        ax.axhline(0.5, color="gray", lw=0.8)
        ax.set_title(ttl)
        ax.set_xlabel("prefix depth")
        ax.set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=7)
    fig.suptitle("Value probe on e4 hidden states: predicting the terminal reward of a partial proof")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig((REPO / "reports/figures/az_value_probe").with_suffix("." + ext), dpi=130)


if __name__ == "__main__":
    main()
