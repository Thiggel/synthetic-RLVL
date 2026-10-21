"""W3.7 midtraining v5 (RLVL-next PLAN): tl_mid5 (the mid4 data mix at 25% TL / 75% Dolci, the synthetic-RLVL instruction
tuning ratio) vs mid4 (55% TL) and mid3 on the v1 / v2 / r3 / v4 TL evals, free and guided, GSM8K (fixed <|im_end|> stop)
and Dolci held-out CE, plus r3 accuracy on templates seen / unseen in each run's own training data."""
import collections
import json
import pathlib
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from datasets import load_from_disk  # noqa: E402

E = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals")
D = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/datasets")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_mid5.md")
# name -> (eval dir, key in dolci_loss / gsm8k files, GSM8K file stem (fixed stop), training data, TL share, train dataset)
RUNS = {"mid3": ("tl_mid3_0.8b_lr2em5_b64_h60_r45", "mid3_h60_r45", "gsm8k_imend_mid3_mid4", "v2", "55%", None),
        "mid4": ("tl_mid4_0.8b_lr2em5_b64_h60_r45_v3q25", "mid4_h60_r45_v3q25", "gsm8k_imend_mid3_mid4", "v2 + 25% gen3", "55%",
                 "tl_mid_v4_20261010"),
        "mid5": ("tl_mid5_0.8b_lr2em5_b64_h60_r75_v3q25", "mid5_h60_r75_v3q25", "gsm8k_mid5_h60_r75_v3q25", "v2 + 25% gen3", "25%",
                 "tl_mid_v5_20261010")}
SETS = ("v1", "v2", "r3", "v4")
MAX_NEW = 2048
cols = {"mid3": "#fc8d62", "mid4": "#e78ac3", "mid5": "#a6d854"}


def jload(p):
    return json.load(open(p)) if p.exists() else {}


def summ(run, s, mode):
    return jload(E / RUNS[run][0] / s / mode / "summary.json")


def gens(run, s, mode):
    p = E / RUNS[run][0] / s / mode / "generations.jsonl"
    return [json.loads(x) for x in open(p)] if p.exists() else []


def outcome(g):
    if g["valid"]:
        return "valid ∧ correct" if g["correct"] else "valid, wrong answer"
    if g["dead_end"]:
        return "dead end (calc)"
    if g["n_tokens"] >= MAX_NEW:
        return f"hit {MAX_NEW} cap"
    return "invalid"


def boot(a, b, n=2000, seed=0):
    """Paired bootstrap 95% CI of mean(a) - mean(b)."""
    rng, d = random.Random(seed), [x - y for x, y in zip(a, b)]
    ms = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), ms[int(0.025 * n)], ms[int(0.975 * n)]


def vc(run, s, mode):
    return {g["example_id"]: float(g["valid"] and g["correct"]) for g in gens(run, s, mode)}


loss, gsm, gsmrows = {}, {}, {}
for n, (_, k, stem, *_) in RUNS.items():
    loss.update(jload(E / f"dolci_loss_{k}.json"))
    gsm[n] = jload(E / f"{stem}.json").get(k, {})
    p = E / f"{stem}_{k}.jsonl"
    if p.exists():
        gsmrows[n] = [json.loads(x)["correct"] for x in open(p)]
f = lambda v, s="{:.3f}": "–" if v is None else s.format(v)  # noqa: E731

lines = ["# W3.7 tl_mid5 (25% TL / 75% Dolci) vs mid4 / mid3 (55% TL) (0.8B, lr 2e-5, batch 64, rows60 head)", "",
         "## TL evals (n = 1000 each): valid ∧ correct, free / guided", "",
         "| model | data | TL share | " + " | ".join(f"{s} free | {s} guided" for s in SETS) + " |",
         "|---|---|---|" + "---|---|" * len(SETS)]
for n, (_, _, _, data, share, _) in RUNS.items():
    cells = [f(summ(n, s, m).get("all", {}).get("valid_and_correct")) for s in SETS for m in ("free", "guided")]
    lines.append(f"| {n} | {data} | {share} | " + " | ".join(cells) + " |")
for s in SETS:
    for m in ("free", "guided"):
        a, b = vc("mid5", s, m), vc("mid4", s, m)
        ids = sorted(set(a) & set(b))
        if ids:
            d, lo, hi = boot([a[i] for i in ids], [b[i] for i in ids])
            lines.append(f"\n{s} {m} mid5 − mid4: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired bootstrap, n = {len(ids)})")
lines += ["", "## General ability (GSM8K with the <|im_end|> stop)", "",
          "| model | Dolci CE/tok | GSM8K (1319) | GSM8K mean tokens | hit cap | TL tokens in answer |", "|---|---|---|---|---|---|"]
for n, (_, k, *_) in RUNS.items():
    g = gsm[n]
    lines.append(f"| {n} | {f(loss.get(k, {}).get('dolci', {}).get('ce_per_token'))} | {f(g.get('acc'))} | "
                 f"{f(g.get('mean_tokens'), '{:.0f}')} | {f(g.get('hit_cap'), '{:.1%}')} | {f(g.get('tl_leak'), '{:.1%}')} |")
for a, b in (("mid5", "mid4"), ("mid5", "mid3")):
    if a in gsmrows and b in gsmrows:
        m, lo, hi = boot(gsmrows[a], gsmrows[b])
        lines.append(f"\nGSM8K {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired bootstrap, 2000 resamples)")

# r3 per theme and guided outcomes
lines += ["", "## r3 per theme", "", "| model | theme | n | free valid | free v∧c | guided v∧c | guided dead end |",
          "|---|---|---|---|---|---|---|"]
theme_rows = {}
for n in RUNS:
    fr, gu = summ(n, "r3", "free").get("by_family", {}), summ(n, "r3", "guided").get("by_family", {})
    for t in sorted(fr):
        theme_rows[n, t] = (fr[t]["valid"], fr[t].get("valid_and_correct"), gu.get(t, {}).get("valid_and_correct"),
                            gu.get(t, {}).get("dead_end"))
        lines.append(f"| {n} | {t} | {fr[t]['n']} | " + " | ".join(f(v) for v in theme_rows[n, t]) + " |")
cats = ("valid ∧ correct", "valid, wrong answer", "dead end (calc)", f"hit {MAX_NEW} cap", "invalid")
outc = {n: collections.Counter(outcome(g) for g in gens(n, "r3", "guided")) for n in RUNS}
lines += ["", "## r3 guided outcomes", "", "| model | " + " | ".join(cats) + " |", "|---|---|---|---|---|---|"]
for n, c in outc.items():
    if c:
        lines.append(f"| {n} | " + " | ".join(str(c[k]) for k in cats) + " |")


# r3 coverage by each run's own training data: same prompt template once numbers and capitalised words (names) are masked
def norm(t):
    t = re.sub(r"^<tl>\n", "", t)
    t = " ".join(re.sub(r"^@ \d+ : ", "", x) for x in t.split("\n"))
    t = re.sub(r"\d+(\.\d+)?", "#", t)
    return re.sub(r"\b[A-Z][a-z]+\b", "N", t)


split = lambda t: re.split(r"(?<=[.?!]) ", t)  # noqa: E731
r3 = [json.loads(x) for x in open(D / "tl_eval_r3_20261010/tl_eval.jsonl")]
cov = {}  # run -> theme -> [n, seen template, all sentences seen, ok | seen, ok | unseen]
for n, (*_, ds) in RUNS.items():
    if ds is None:
        continue
    tr = [norm(r["prompt"]) for r in load_from_disk(str(D / ds))["train"] if r["family"].startswith("R:")]
    seen, sents = set(tr), {x for t in tr for x in split(t)}
    ok = vc(n, "r3", "free")
    cov[n] = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    lines += ["", f"## r3 coverage by {n} training ({len(tr)} R: rows; numbers / names masked)", "",
              "| theme | n | same template in train | all sentences in train | free v∧c, seen template | unseen template (n) |",
              "|---|---|---|---|---|---|"]
    for e in r3:
        t = norm(e["text"])
        a, o = t in seen, ok.get(e["example_id"], 0.0)
        for k in (e["family"], "all"):
            c = cov[n][k]
            c[0] += 1; c[1] += a; c[2] += all(x in sents for x in split(t)); c[3] += o * a; c[4] += o * (not a)  # noqa: E702
    for k, (m, a, b, oa, on) in sorted(cov[n].items(), key=lambda kv: (kv[0] == "all", kv[0])):
        lines.append(f"| {k} | {m} | {a / m:.2f} | {b / m:.2f} | {f(oa / a if a else None)} | {f(on / (m - a) if m > a else None)} ({m - a}) |")

# figure
fig, axs = plt.subplots(1, 5, figsize=(24, 5.2), gridspec_kw={"width_ratios": [2.4, 1.8, 1.2, 1.2, 1.8]})
ax, w = axs[0], 0.13
for i, n in enumerate(RUNS):
    for j, m in enumerate(("free", "guided")):
        xs = [k + (2 * i + j - 2.5) * w for k in range(len(SETS))]
        ys = [summ(n, s, m).get("all", {}).get("valid_and_correct") for s in SETS]
        ax.bar([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], w, color=cols[n],
               hatch="//" if m == "guided" else None, edgecolor="k", lw=0.4, label=f"{n} ({RUNS[n][4]} TL) {m}")
ax.set_xticks(range(len(SETS)), ["v1 (KB v1)", "v2 (KB v2)", "r3 (realistic)", "v4 (mid4/5 eval)"])
ax.set_ylabel("valid ∧ correct")
ax.set_title("TL evals (n = 1000)")
ax.legend(ncol=3, fontsize=7, loc="upper center")
ax.set_ylim(0, 1.25)
ax = axs[1]
themes = sorted({t for (_, t) in theme_rows})
for k, lab, mk in ((1, "free v∧c", "o"), (2, "guided v∧c", "s")):
    for j, n in enumerate(RUNS):
        ys = [theme_rows.get((n, t), (None,) * 4)[k] for t in themes]
        xs = [x + (j - 1) * 0.2 for x in range(len(themes))]
        ax.scatter([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], marker=mk, color=cols[n],
                   edgecolor="k", lw=0.4, s=30, label=f"{n} {lab}")
ax.set_xticks(range(len(themes)), [t.split(":")[1] for t in themes])
ax.set_title("r3 per theme")
ax.set_ylim(-0.03, 1.03)
ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax = axs[2]
ccol = ("#4daf4a", "#e41a1c", "#984ea3", "#bbbbbb", "#ffffff")
names = [n for n in RUNS if outc[n]]
bottom = [0] * len(names)
for c, cc in zip(cats, ccol):
    v = [outc[n][c] / 1000 for n in names]
    ax.bar(names, v, bottom=bottom, color=cc, label=c, edgecolor="k", lw=0.4)
    bottom = [a + b for a, b in zip(bottom, v)]
ax.set_title("r3 guided: outcome of each row")
ax.legend(fontsize=7, ncol=2, loc="upper center")
ax.set_ylim(0, 1.3)
ax = axs[3]
names = list(RUNS)
acc = [gsm[n].get("acc") for n in names]
ce = [loss.get(RUNS[n][1], {}).get("dolci", {}).get("ce_per_token") for n in names]
ax.bar(names, [a or 0 for a in acc], color=[cols[n] for n in names], edgecolor="k", lw=0.4)
for x, a, c in zip(range(len(names)), acc, ce):
    ax.text(x, (a or 0) + 0.065, f"{f(a)}\nCE {f(c)}", ha="center", fontsize=8)
ax.axhline(0.600, color="gray", ls=":", label="base_ext 0.600")
ax.set_ylim(0, 0.8)
ax.set_title("GSM8K (1319, greedy, im_end stop)\n+ Dolci CE", fontsize=10)
ax.legend(fontsize=7, loc="lower right")
ax = axs[4]
ks = sorted({k for n in cov for k in cov[n] if k != "all"}) + ["all"]
for j, n in enumerate(cov):
    c = cov[n]
    xs = [i + (2 * j - 1.5) * 0.18 for i in range(len(ks))]
    ax.bar(xs, [c[k][3] / c[k][1] if c[k][1] else 0 for k in ks], 0.18, color=cols[n], edgecolor="k", lw=0.4, label=f"{n} seen")
    ax.bar([x + 0.18 for x in xs], [c[k][4] / (c[k][0] - c[k][1]) if c[k][0] > c[k][1] else 0 for k in ks], 0.18,
           color="white", hatch="..", edgecolor=cols[n], lw=0.8, label=f"{n} unseen")
ax.set_xticks(range(len(ks)), [k.split(":")[-1] for k in ks])
ax.set_ylim(0, 1.3)
ax.set_title("r3 free v∧c on templates seen / unseen\nin the run's own training data", fontsize=9)
ax.legend(fontsize=7, loc="upper center", ncol=4)
fig.suptitle("W3.7 tl_mid5 (25% TL / 75% Dolci) vs mid4 / mid3 (55% TL)")
fig.tight_layout()
fig.savefig(FIG / "tl_mid5.png", dpi=150)
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
