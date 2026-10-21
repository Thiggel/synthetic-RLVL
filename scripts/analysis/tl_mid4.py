"""W3.6 midtraining v4 (RLVL-next PLAN): tl_mid4 (mid3 recipe + 25% gen3 realistic traces, eval prompts removed) vs mid3 /
r45 / h60 on the v1 / v2 / r3 / v4 TL evals, free and guided, GSM8K and Dolci held-out CE, r3 guided failure modes, and
how much of r3 is covered by mid4's training templates (numbers and names masked)."""
import collections
import json
import pathlib
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/tl/evals")
FIG = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/reports/figures")
OUT = pathlib.Path("/vol/tmp2/laitenbf/synthetic-RLVL/analysis/tl_mid4.md")
# name -> (eval dir, key in dolci_loss / gsm8k files, training data, head)
RUNS = {"r45": ("tl_mid_0.8b_lr2em5_b64_r45", "r45", "v1", "full"),
        "h60": ("tl_mid_0.8b_lr2em5_b64_h60", "h60", "v1", "rows60"),
        "mid3": ("tl_mid3_0.8b_lr2em5_b64_h60_r45", "mid3_h60_r45", "v2", "rows60"),
        "mid4": ("tl_mid4_0.8b_lr2em5_b64_h60_r45_v3q25", "mid4_h60_r45_v3q25", "v2 + 25% gen3", "rows60")}
SETS = ("v1", "v2", "r3", "v4")
MAX_NEW = 2048


def jload(p):
    return json.load(open(p)) if p.exists() else {}


def edir(run, s, mode):
    d = E / RUNS[run][0]
    # the v1-era runs were evaluated on v1 before named eval sets existed
    return d / mode if s == "v1" and run not in ("mid3", "mid4") else d / s / mode


def summ(run, s, mode):
    return jload(edir(run, s, mode) / "summary.json")


def gens(run, s, mode):
    p = edir(run, s, mode) / "generations.jsonl"
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


loss, gsm, gsmrows = {}, {}, {}
for n, (_, k, _, _) in RUNS.items():
    loss.update(jload(E / f"dolci_loss_{k}.json"))
    gsm.update(jload(E / f"gsm8k_{k}.json"))
    p = E / f"gsm8k_{k}_{k}.jsonl"
    if p.exists():
        gsmrows[n] = [json.loads(x)["correct"] for x in open(p)]
f = lambda v, s="{:.3f}": "–" if v is None else s.format(v)  # noqa: E731

lines = ["# W3.6 tl_mid4 (+25% gen3 realistic traces) vs mid3 / v1-data runs (0.8B, lr 2e-5, batch 64)", "",
         "## TL evals (n = 1000 each): valid ∧ correct, free / guided", "",
         "| model | data | head | " + " | ".join(f"{s} free | {s} guided" for s in SETS) + " |",
         "|---|---|---|" + "---|---|" * len(SETS)]
for n, (_, _, data, head) in RUNS.items():
    cells = [f(summ(n, s, m).get("all", {}).get("valid_and_correct")) for s in SETS for m in ("free", "guided")]
    lines.append(f"| {n} | {data} | {head} | " + " | ".join(cells) + " |")
lines += ["", "## General ability", "", "| model | Dolci CE/tok | GSM8K (1319) | GSM8K mean tokens | hit cap | TL tokens in answer |",
          "|---|---|---|---|---|---|"]
for n, (_, k, _, _) in RUNS.items():
    g = gsm.get(k, {})
    lines.append(f"| {n} | {f(loss.get(k, {}).get('dolci', {}).get('ce_per_token'))} | {f(g.get('acc'))} | "
                 f"{f(g.get('mean_tokens'), '{:.0f}')} | {f(g.get('hit_cap'), '{:.1%}')} | {f(g.get('tl_leak'), '{:.1%}')} |")
for a, b in (("mid4", "mid3"), ("mid3", "r45"), ("mid3", "h60")):
    if a in gsmrows and b in gsmrows:
        m, lo, hi = boot(gsmrows[a], gsmrows[b])
        lines.append(f"\nGSM8K {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (paired bootstrap, 2000 resamples)")

# r3 per theme and guided outcomes
lines += ["", "## r3 (realistic themes, held out): per theme", "",
          "| model | theme | n | free valid | free correct | guided valid∧correct | guided dead end | guided hit cap |",
          "|---|---|---|---|---|---|---|---|"]
theme_rows = {}
for n in RUNS:
    fr, gu = summ(n, "r3", "free").get("by_family", {}), summ(n, "r3", "guided").get("by_family", {})
    gg = collections.defaultdict(list)
    for g in gens(n, "r3", "guided"):
        gg[g["family"]].append(g["n_tokens"] >= MAX_NEW)
    for t in sorted(fr):
        cap = sum(gg[t]) / len(gg[t]) if gg[t] else None
        theme_rows[n, t] = (fr[t]["valid"], fr[t]["correct"], gu.get(t, {}).get("valid_and_correct"),
                            gu.get(t, {}).get("dead_end"), cap)
        lines.append(f"| {n} | {t} | {fr[t]['n']} | " + " | ".join(f(v) for v in theme_rows[n, t]) + " |")
outc = {n: collections.Counter(outcome(g) for g in gens(n, "r3", "guided")) for n in RUNS}
def ekind(e):
    e = re.sub(r"^line \d+: ", "", str(e))
    for k in ("new: no such word", "type mismatch", "calc: not entailed", "trailing tokens", "application of", "expected"):
        if e.startswith(k):
            return k
    return "other"


errs = {n: collections.Counter(ekind(g["error"]) for g in gens(n, "r3", "free") if not g["valid"]) for n in RUNS}
lines += ["", "## r3 guided outcomes", "", "| model | " + " | ".join(("valid ∧ correct", "valid, wrong answer", "dead end (calc)", f"hit {MAX_NEW} cap", "invalid")) + " |",
          "|---|---|---|---|---|---|"]
for n, c in outc.items():
    if c:
        lines.append(f"| {n} | " + " | ".join(str(c[k]) for k in ("valid ∧ correct", "valid, wrong answer", "dead end (calc)", f"hit {MAX_NEW} cap", "invalid")) + " |")
lines += ["", "## r3 free: top check errors (invalid rows)", ""]
for n, c in errs.items():
    if c:
        lines.append(f"- {n}: " + ", ".join(f"{k} {v}" for k, v in c.most_common(6)))

# figure
fig, axs = plt.subplots(1, 5, figsize=(24, 5.2), gridspec_kw={"width_ratios": [2.6, 1.8, 1.3, 1.3, 1.5]})
cols = {"r45": "#8da0cb", "h60": "#66c2a5", "mid3": "#fc8d62", "mid4": "#e78ac3"}
ax, w = axs[0], 0.1
for i, n in enumerate(RUNS):
    for j, m in enumerate(("free", "guided")):
        xs = [k + (2 * i + j - 3.5) * w for k in range(len(SETS))]
        ys = [summ(n, s, m).get("all", {}).get("valid_and_correct") for s in SETS]
        ax.bar([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], w, color=cols[n],
               hatch="//" if m == "guided" else None, edgecolor="k", lw=0.4, label=f"{n} {m}")
ax.set_xticks(range(len(SETS)), ["v1 (KB v1)", "v2 (KB v2)", "r3 (realistic)", "v4 (mid4 own eval)"])
ax.set_ylabel("valid ∧ correct")
ax.set_title("TL evals (n = 1000)")
ax.legend(ncol=4, fontsize=7, loc="upper center")
ax.set_ylim(0, 1.25)
ax = axs[1]
themes = sorted({t for (_, t) in theme_rows})
for i, (k, lab, mk) in enumerate(((1, "free correct", "o"), (0, "free valid", "^"), (2, "guided v∧c", "s"))):
    for j, n in enumerate(RUNS):
        ys = [theme_rows.get((n, t), (None,) * 5)[k] for t in themes]
        xs = [x + (j - 1.5) * 0.18 for x in range(len(themes))]
        ax.scatter([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], marker=mk, color=cols[n],
                   edgecolor="k", lw=0.4, s=30, label=f"{n} {lab}")
ax.set_xticks(range(len(themes)), [t.split(":")[1] for t in themes])
ax.set_title("r3 per theme")
ax.set_ylim(-0.03, 1.03)
ax.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax = axs[2]
cats = ("valid ∧ correct", "valid, wrong answer", "dead end (calc)", f"hit {MAX_NEW} cap", "invalid")
ccol = ("#4daf4a", "#e41a1c", "#984ea3", "#bbbbbb")
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
acc = [gsm.get(RUNS[n][1], {}).get("acc") for n in names]
ce = [loss.get(RUNS[n][1], {}).get("dolci", {}).get("ce_per_token") for n in names]
ax.bar(names, [a or 0 for a in acc], color=[cols[n] for n in names], edgecolor="k", lw=0.4)
for x, a, c in zip(range(len(names)), acc, ce):
    ax.text(x, (a or 0) + 0.065, f"{f(a)}\nCE {f(c)}", ha="center", fontsize=8)
ax.axhline(0.600, color="gray", ls=":", label="base_ext 0.600")
ax.set_ylim(0, 0.8)
ax.set_title("GSM8K (1319, greedy) + Dolci CE")
ax.legend(fontsize=7, loc="lower right")

# r3 coverage by mid4's training data: same prompt template once numbers and capitalised words (names) are masked
from datasets import load_from_disk  # noqa: E402

D = pathlib.Path("/vol/tmp2/laitenbf/rlvl_data/datasets")


def norm(t):
    t = re.sub(r"^<tl>\n", "", t)
    t = " ".join(re.sub(r"^@ \d+ : ", "", x) for x in t.split("\n"))
    t = re.sub(r"\d+(\.\d+)?", "#", t)
    return re.sub(r"\b[A-Z][a-z]+\b", "N", t)


split = lambda t: re.split(r"(?<=[.?!]) ", t)  # noqa: E731
tr = [norm(r["prompt"]) for r in load_from_disk(str(D / "tl_mid_v4_20261010"))["train"] if r["family"].startswith("R:")]
seen, sents = set(tr), {x for t in tr for x in split(t)}
ok4 = {g["example_id"]: g["valid"] and g["correct"] for g in gens("mid4", "r3", "free")}
cov = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # n, seen template, all sentences seen, ok | seen, ok | unseen
for e in map(json.loads, open(D / "tl_eval_r3_20261010/tl_eval.jsonl")):
    t = norm(e["text"])
    a, ok = t in seen, ok4.get(e["example_id"], False)
    for k in (e["family"], "all"):
        c = cov[k]
        c[0] += 1; c[1] += a; c[2] += all(x in sents for x in split(t)); c[3] += ok and a; c[4] += ok and not a  # noqa: E702
lines += ["", "## r3 coverage by mid4 training (numbers / names masked)", "",
          "| theme | n | same template in train | all sentences in train | mid4 free v∧c, seen template | unseen template (n) |",
          "|---|---|---|---|---|---|"]
for k, (n, a, b, oa, on) in sorted(cov.items(), key=lambda kv: (kv[0] == "all", kv[0])):
    lines.append(f"| {k} | {n} | {a / n:.2f} | {b / n:.2f} | {f(oa / a if a else None)} | {f(on / (n - a) if n > a else None)} ({n - a}) |")
ax = axs[4]
ks = [k for k in sorted(cov) if k != "all"]
x = range(len(ks))
ax.bar([i - 0.2 for i in x], [cov[k][3] / cov[k][1] if cov[k][1] else 0 for k in ks], 0.4, color="#e78ac3", edgecolor="k", lw=0.4, label="seen template")
ax.bar([i + 0.2 for i in x], [cov[k][4] / (cov[k][0] - cov[k][1]) if cov[k][0] > cov[k][1] else 0 for k in ks], 0.4,
       color="white", hatch="..", edgecolor="#c51b7d", lw=0.6, label="unseen template")
for i, k in enumerate(ks):
    ax.text(i, 1.06, f"{cov[k][1] / cov[k][0]:.0%}", ha="center", fontsize=7)
ax.set_xticks(list(x), [k.split(":")[1] for k in ks])
ax.set_ylim(0, 1.3)
ax.set_title("mid4 r3 free v∧c by template coverage\n(% = share of rows with a seen template)", fontsize=9)
ax.legend(fontsize=7, loc="upper right", ncol=2)
fig.suptitle("W3.6 tl_mid4 (+25% gen3 realistic traces) vs mid3 / r45 / h60")
fig.tight_layout()
fig.savefig(FIG / "tl_mid4.png", dpi=150)
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
