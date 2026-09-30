# 2026-09-30: library extension, checker regression check, and why sampled proofs fail on Dolci

Context: the approved plan (self-distillation on real prompts, "lemmas everywhere", the Dolci mix, and an RL push).
This note covers three things: the library extension, a check that the new checker does not change any
existing verdict, and a breakdown of how sampled proofs fail on the Dolci RL gate. That breakdown shows where
the remaining validity has to come from.

## 1. Library extension (RLVL-next `rlvl/lib`, `gen/rlvlgen`)

- 75 new lemmas in five modules:

  | module | lemmas | covers |
  |---|---|---|
  | `nt` | 26 | int, mod, parity, dvd, gcd, lcm, prime |
  | `alg` | 13 | squares, sums 1..n, means, linear equations |
  | `comb` | 5 | factorials, permutations, combinations |
  | `geom` | 22 | areas, perimeters, volumes, circles, angles, triangles, midpoints |
  | `rate` | 9 | distance/speed/time, work, unit prices, percent |

- They come with six opt-in generator families (`dataset.EXTRA`): `lemmas`, `mathlemmas`, `numth`, `geom`,
  `rates`, `counting`. See DESIGN.md §5 and §8.1.
- These families are kept out of `FAMILIES`, so every existing pool can still be rebuilt bit for bit.
- The demand side came from `analysis/lib_coverage.json` (`scripts/analysis/lib_coverage.py`). On the Dolci
  gate, the x3 @1562 SFT model makes 453 lemma citations, and 285 of them name lemmas that did not exist.
  The most frequent: `math.gcd_def`, `count.add`, `calc.sub`, `set.sum_def`, `motion.density`, `gcd.step*`,
  `geometry.cut_iff`.

![lib coverage](figures/lib_coverage.png)

The live checker .so in RLVL-next changed at 17:50 today. Every comparative run and eval stays on the frozen
snapshot `rlvl_data/checker_snapshot_pre_libext_20260930` (via `RLVL_PYTHONPATH`) until a new arm is trained on
the new families. Pending evals that would have picked up the new .so were resubmitted pinned: 8169, 8170, 8171.

## 2. Old vs new checker on existing outputs: no regressions, no gains

The lib agent flagged a possible name clash: policies often `def` the symbols `area`, `base` and `prime`,
which are now lib symbols. To test this, I rescored two sets with `formal_rewards.components` (the GRPO
reward) under both checkers:

- all 27,598 passing completions of the G8-final harvest (`rl_filter_20260930`);
- all 15,200 G8-final k16 samples on the Dolci gate.

| set | n | valid old → new | cvf old → new | verdicts changed |
|---|---|---|---|---|
| harvest: gen | 26,549 | 1.000 → 1.000 | 1.000 → 1.000 | 0 |
| harvest: wordprob / yesno / math | 911 / 111 / 27 | 1.000 → 1.000 | 1.000 → 1.000 | 0 |
| gate: wordprob | 3,200 | .0619 → .0619 | .0472 → .0472 | 0 |
| gate: yesno | 3,200 | .0206 → .0206 | .0181 → .0181 | 0 |
| gate: knowledge | 1,600 | .0281 → .0281 | 0 → 0 | 0 |
| gate: math / dapo | 4,800 / 2,400 | 0 → 0 | 0 → 0 | 0 |

- The two binaries differ (md5), but no verdict changed on any of the 42,798 samples.
- Only 2 error messages changed. In both, an ill-typed formula became a later-stage error: an unknown lemma
  `route.dist` in one, a quote mismatch in the other.
- So the name clash does not hurt existing policies. The new lemmas also cannot help them yet: G8 never cites
  them, and most samples fail before any lemma lookup (section 3).
- The gain from the extension has to come from training on the new families, followed by an eval under the new
  checker.

## 3. Why sampled proofs fail on the Dolci gate

`scripts/analysis/gate_error_breakdown.py` → `analysis/gate_error_breakdown.json`

Setup:
- Data: the 700 in-domain Dolci prompts (math, wordprob, yesno) × 16 samples at T=1.0, scored with the frozen
  checker.
- Left panel: the first checker stage each sample fails.
- Right panel: parse errors split by the text at the error position. This split is a heuristic regex
  classifier (see the script docstring); treat its shares as ±a few points.

![gate error breakdown](figures/gate_error_breakdown.png)

| model | no proof | parse | rule | quote | literal | valid |
|---|---|---|---|---|---|---|
| SFT p25 fp32m | .638 | .259 | .046 | .023 | .014 | .008 |
| SFT x3 @1562 | .665 | .233 | .041 | .025 | .017 | .008 |
| SFT p50 + lemma catalog | .454 | .434 | .059 | .028 | .008 | .004 |
| GRPO G8 final (cvf, 200 steps) | .142 | .701 | .058 | .031 | .015 | .035 |

**Findings:**

- **GRPO mostly turned "no proof" into "parse error".** Proof attempts rose from 55% to 86%, but 70% of all
  samples now fail to parse. Valid rose 9× (.004 → .035), yet that is still only 1 in 29 samples.
- **Rule and quote errors are small and flat.** Rule errors (unknown or misapplied lemmas) hold at 5–6% and
  quote errors at about 3% across all models. The library extension targets the rule errors, so on its own it
  can recover at most about 6 points, and only after training.
- **The parse errors are the model reaching past the language.** G8, as a share of all samples:

  | kind | share | examples |
  |---|---|---|
  | notation the language lacks | .22 | `{k=1..n}`, `[2, 5, ...]`, `|A|`, `a_n`, `%`, `**`, `==`, `+=`, chained `a < x < b`, tuples |
  | malformed justification | .17 | no `; rule`, or bad rule arguments like `; know formula`, `; split into ...` |
  | juxtaposition | .14 | multi-word names (`total outcomes = 6`, `goal people eat pizza ?`), implicit products (`2(x+1)`) |
  | `?` placeholders | .04 | — |
  | prose in formulas | .04 | `n is the smallest ...` |
  | other | .10 | — |

  These errors are not "wrong lemma" failures. The model writes math notation or English that the formal
  syntax does not accept.

**Implications for the plan:**

1. **Self-distillation is the right lever.** The harvest (and the GSM8K-train harvest now running, 8176) consists
   of real-prompt proofs that parse. Training on it teaches the model to stay inside the syntax on real prompts,
   which is exactly the 70% failure mode. The generator data alone cannot teach this, because generator
   prompts never invite `{k=1..n}` or multi-word names.
2. **RL sees almost no signal here.** With binary cvf, a parse failure scores the same as no proof, so GRPO gets
   no gradient toward "parseable". A shaped reward like `lines` or `frac_hard` (already in `formal_rewards`)
   gives partial credit for parsed lines. It is worth an arm after L1, if the L1 cvf arm stalls on Dolci.
3. **Juxtaposition and notation can be taught cheaply.** The generator could render quantities with multi-word
   natural names ("total cost") that must be written as `total_cost`, and sums and lists could be spelled out
   with `alg.sum_formula`. Making the parser accept the common notations instead would be a language change and
   is out of scope for now.

## 4. Update 2026-10-01 01:00: the x3 SFT finished (150k generator rows), and pass@k barely moves

![passk ladder](figures/passk_ladder.png)

On the Dolci gate: 950 prompts, 16 samples at T=1.0, frozen checker (`analysis/passk_ladder.md`, `scripts/analysis/passk_ladder.py`, job 8171).

| SFT generator rows | has_proof@16 | valid@8 | valid@16 | valid·correct@16 |
|---|---|---|---|---|
| ~25k (p25 fp32m) | .629 | .041 | .059 | .022 |
| ~50k (x3 @781) | .442 | .044 | .063 | .019 |
| ~100k (x3 @1562) | .537 | .047 | .066 | .023 |
| ~150k (x3 final) | .513 | .055 | .076 | .021 |
| p50 + lemma catalog | .780 | .026 | .043 | .023 |
| + GRPO G8 (200 steps cvf) | .971 | .096 | .121 | .065 |

- Going from 25k to 150k generator rows adds +.017 valid@16 and nothing in valid·correct@16.
- 200 GRPO steps add +.078 valid@16 and ×3 valid·correct@16.

This confirms that dropping plain data-scale midtraining was right, and that the budget belongs with self-distillation (EI), the new lemma families and RL (the c/l/e/le arms and L1).

**Data loss and fix.** `scripts/train_formal_mixture_sft.py` deleted every `checkpoint-*` dir when training ended. That included the x3 @781/@1562 gate outputs stored inside them. Their numbers survive in the committed tables (`analysis/passk_ladder.md`, `analysis/gate_error_breakdown.json`), and both analysis scripts now fall back to those records (`analysis/passk_ladder.json`). The cleanup now keeps any subdir of a checkpoint that holds a `summary.json`.
