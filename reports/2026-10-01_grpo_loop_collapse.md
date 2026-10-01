# GRPO, 2026-10-01: L1 held-out gates, truncation masking and the G14 fix arm

**Summary.**
- Correct-only GRPO (L1_correct): correctness on its own training prompts converged by step ~600, at .46 on Dolci and .95 on the generator. Held-out greedy correctness is still creeping up: .460 at step 500, .481 at step 1000 (paired z = 1.5). Validity has been 0 since step ~180.
- The cvf arm (L1_cvf) at step 250 nearly triples clean-gate validity, .014 → .039. Correctness is flat (.261 → .269).
- On held-out generator problems, though, L1_cvf *loses* 20 points of validity (.807 → .605).
- About a quarter of that loss is length truncation, mostly repetition loops. TRL masks truncated completions out of the GRPO loss, so the loops are never penalised.
- G14 is a new fix arm: G13 with truncated completions kept in the loss plus a 0.5 overlong penalty. To keep the GPU budget constant, G12 (old library, flat reward) is paused at checkpoint-150.
- The other three quarters of the loss are sloppier proofs (parse and rule errors). These point to policy drift; GRPO runs with beta = 0, i.e. no KL anchor to the SFT model.

All gate numbers use the **clean** gate subset: the 713 of 950 items with no near-duplicate in the training-prompt pool. They are greedy and scored by the frozen pre-libext checker.

## 1. L1 held-out gates (steps 0–1000)

| checkpoint | gate correct (clean) | gate valid (clean) | gate valid·correct (clean) | OOD correct (250) | gen_test correct | gen_test valid | gen_test cvf |
|---|---:|---:|---:|---:|---:|---:|---:|
| base (`p50_cont_lc_fp32m` final) | .261 | .014 | .0042 | .056 | .913 | .786 | .747 |
| L1 correct @250 | .438 | 0 | 0 | .164 | .949 | 0 | 0 |
| L1 correct @500 | **.460** | 0 | 0 | .180 | .968 | 0 | 0 |
| L1 correct @750 | .446 | 0 | 0 | .164 | .951 | 0 | 0 |
| L1 correct @1000 | **.481** | 0 | 0 | .184 | .949 | 0 | 0 |
| L1 cvf @250 | .269 | **.039** | **.022** | .032 | .828 | .571 | .530 |
| L1 cvf @500 | .309 | **.077** | **.055** | .044 | .792 | .484 | .459 |

Sources:
- `analysis/gate_clean_rescore.md` and `analysis/l1_convergence.json`.
- gen_test: 1837 held-out generator problems, rescored with the training reward.
- OOD: the 250 gate items from benches that are not in the RL pool.

![held-out](figures/l1_convergence_heldout.png)

**Correct only.** Held-out correctness rises .261 → .460 by step 500, then stalls (.446 at step 750). Training-prompt correctness also flattens: the fit gives a Dolci asymptote of .45 with t95 ≈ 540 steps. **Correctness looks converged from about step 500, at about .45 (clean gate) / .47 (Dolci training prompts) / .96 (generator).** The final call comes at step 1000, when the plateau test has enough steps.

**Update, correct @1000 (gate 9859, 18:35).**
- *Training prompts:* converged. The plateau test (last 100 steps within the fit's noise band) fires for Dolci correct at .459 from step ~970 (fit asymptote .452, t95 ≈ 525). Generator correct plateaus at .953 from step ~630.
- *Held-out, clean gate (greedy):* not flat yet. Correctness goes .438 → .460 → .446 → .481 at steps 250/500/750/1000.
  - Paired over the 713 clean items: step 250 → 1000 = +4.3 pp (SE 1.5, z = 2.9); step 500 → 1000 = +2.1 pp (z = 1.5).
  - So held-out correctness still gains about 1 pp per 250 steps. Training-prompt sampled correctness has stopped moving, so the most likely source is the greedy decode sharpening. OOD correctness (250 items) is .184, level with step 500.
- *Answer so far, correctness reward:* correctness converges at ≈ .46 (Dolci, sampled) / .95 (generator) by step ~600, with a slow held-out tail (.48 at step 1000). Validity converges to exactly 0 by step ~180: the policy drops `<proof>` completely. The run continues toward 5000 steps to test for a late tail. Validity converged to 0 by step ~180 (see the 2026-09-30 interim report).

**cvf.** By step 250, clean-gate validity has nearly tripled (.014 → .039) and valid·correct has grown fivefold (.004 → .022). Both are still tiny. Gate correctness is flat. On the generator, where the base model is already strong, every metric drops: valid .786 → .571, cvf .747 → .530. Training-prompt cvf on generator prompts falls too, .584 (first 25 steps) → .541 (last 100). So the arm is getting worse at its own reward on the prompts where it was good. Section 2 looks at why.

**Update, cvf @500 (gate 9858, 17:50).** The trade-off continues in both directions.
- Clean-gate validity doubles again (.039 → .077), as does valid·correct (.022 → .055). Gate correctness rises a little too (.269 → .309).
- Generator validity keeps falling: .571 → .484 (base .786).
- The cvf arm is converting generator validity into Dolci validity, at roughly 1 point gained on the gate per 2 points lost on the generator. Neither side has plateaued.

## 2. Why the cvf arm loses generator validity

The held-out generator test (`formal_eval`, 2000 items, greedy) classifies each completion by its first failure:

| first failure | base | L1 cvf @250 | Δ (pp) |
|---|---:|---:|---:|
| valid | .807 | .605 | −20.2 |
| rule error (lib/mp/and/calc/…) | .127 | .194 | +6.7 |
| parse error (not truncated) | .019 | .093 | +7.4 |
| truncated at 2048 tokens | .004 | .051 | +4.7 |
| literal / type error | .002 | .028 | +2.6 |
| other (answer, quote, scope, …) | .041 | .029 | −1.2 |

Two other shifts at step 250:
- **Repetition loops** (a proof line repeated ≥ 8 times, digits masked) went from .004 to .040 of completions. Every looped completion is invalid. Typical example: `94 heavier(erin, ravi) ; given …`, `95 lighter(…)`, `96 heavier(…) ; mp 95 94`, repeated with renumbered lines until the 2048-token limit.
- **Mean length** went from 260 to 321 tokens.

So **truncation and loops cost about 5 of the 20 points.** The other 15 are proofs that end but are wrong: malformed lines, rules applied to lines that do not license them, and type slips.

## 3. Training dynamics: truncated completions get no gradient

`grpo_formal.py` ran every arm with TRL's default `mask_truncated_completions=True`. A completion that reaches the 2048-token limit is dropped from the loss. Most such completions are loops: P(loop | truncated) is .60–.90 for the cvf arms. A loop therefore never receives the negative advantage its reward of 0 would give it.

![loop collapse](figures/grpo_loop_collapse.png)

| arm | steps | truncated (clipped ratio) | entropy | zero-variance groups | mean length (tok) | loop share | rollout reward |
|---|---|---|---|---|---|---|---|
| L1 correct (old lib) | 1–870 | .044 → .233 | .316 → .327 | .59 → .75 | 437 → 976 | .052 → .078 | .447 → .654 (correct) |
| L1 cvf (old lib) | 1–395 | .033 → .134 | .343 → .162 | .73 → .90 | 411 → 551 | .047 → .137 | .225 → .174 (cvf) |
| G12 cvf_fmt (G10@500, old lib) | 1–144 | .236 → .153 | .031 → .037 | .90 → .89 | 710 → 559 | .214 → .133 | .239 → .254 (cvf) |
| G13 cvf_fmt (le SFT, new lib) | 1–103 | .165 → .144 | .135 → .101 | .73 → .80 | 687 → 624 | .125 → .103 | .236 → .319 (cvf) |

Notes:
- The first and last columns of each range are means over the first and last 25 logged steps.
- Loop share and reward are 25-step bins of the 256 training rollouts per step.
- Source: `scripts/analysis/grpo_loop_collapse.py` → `analysis/grpo_loop_collapse.md`.

Truncated rollouts over the last 50 logged steps:

| arm | truncated share | P(loop \| truncated) | P(truncated \| loop) | mean advantage of truncated (masked) |
|---|---:|---:|---:|---:|
| L1 cvf | .111 | .84 | .72 | −.019 |
| G12 | .164 | .90 | .84 | −.039 |
| G13 | .142 | .60 | .76 | −.010 |
| L1 correct | .225 | .16 | – | −.178 |

What the curves show:
- **L1 cvf collapses slowly:**
  - truncation .03 → .13;
  - loops .05 → .14;
  - entropy halves (.34 → .16);
  - 90% of groups end up with zero reward variance, so they carry no signal;
  - the reward itself falls.
- **L1 correct** went through a loop phase: loops peaked at .40 around step 290. Then it abandoned `<proof>` altogether. Its truncations are now long natural-language reasoning (P(loop | truncated) = .16).
- **G12 and G13** are not collapsing yet; their loop share falls.

Masking alone is not the whole story, though. Under cvf most truncated loops sit in groups where every member has reward 0. Unmasking them would add only a weak signal: their mean advantage is about −.01 to −.04.

## 4. G14: unmask truncated completions and penalise them

G14 (run `2b_le_G14_cvffmt_overlong`; jobs 9825 → 9826 on gruenau12, started 14:43) is G13 with two flags: `--no-mask-truncated --overlong-penalty 0.5`.
- The start, prompts, reward (cvf_fmt), new checker and hyperparameters are all the same as G13.
- Truncated completions stay in the loss and get an extra reward term of −0.5.
- In an otherwise all-zero group, the loop now gets a clearly negative advantage and its siblings a small positive one. Groups that used to carry no signal now do; this is DAPO-style overlong shaping.
- Code: `scripts/formal_rewards.py`, `make_truncated` and `reward_funcs(arm, overlong_penalty, max_len)`; `scripts/grpo_formal.py`, `--no-mask-truncated` and `--overlong-penalty`. The defaults are unchanged, so the running L1 and G13 chains behave as before.

**First step (14:50).** The reward is .225, i.e. cvf_fmt .293 − 0.5 × truncated share .137, so the penalty is active. Only .28 of groups have zero reward variance; G13 at step 1 had 0.62, so most groups now carry a learning signal.

What to compare at matched steps (100, 250), G14 vs G13:
- truncation share;
- loop share;
- entropy;
- zero-variance groups;
- clean gate valid / valid·correct;
- gen_test valid.

Caveat on entropy: TRL averages entropy over the tokens in the loss. G14's entropy therefore also covers truncated completions, which G13's excludes, so entropy is not strictly comparable between the two. Truncation share, length, loop share and the gates are comparable.

**Steps 1–15 (15:38).**
- Truncation: G14 .175 → .152, G13 .172 → .177.
- Mean length: G14 692 → 657, G13 689 → 714.
- Zero-variance groups: G14 .29–.31, G13 .69–.72.
- Early and within noise, but in the expected direction.

Section 2 caps what this can fix: about a quarter of the cvf arm's generator loss. If G14 removes loops but the parse/rule drift remains, the next arm is a KL anchor (beta 0.02–0.04; GRPO currently runs with beta = 0).

**GPU budget.** G14 replaces G12 on gruenau12. G12 is paused after checkpoint-150:
- 9655 cancelled after the save;
- its next chain link 9656 is on `scontrol hold`; `scontrol release 9656` resumes it from checkpoint-150.

Why G12: it is the weakest arm.
- It uses the old library.
- Entropy is .03.
- Rollout cvf is flat (.239 → .254 over 144 steps).
- At 200 s/step, the remaining 850 steps would need about 47 h.

**Steps 1–125 (20:50): the fix works on its target, at no cost in reward.** Training rollouts, 25-step bins (`analysis/grpo_loop_collapse.json`):

| steps | G13 truncated | G14 truncated | G13 length (tok) | G14 length (tok) | G13 loops | G14 loops | G13 rollout cvf | G14 rollout cvf | G13 zero-var | G14 zero-var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–25 | .165 | .106 | 687 | 557 | .125 | .096 | .228 | .221 | .73 | .42 |
| 26–50 | .179 | .004 | 696 | 303 | .127 | .030 | .256 | .245 | .72 | .70 |
| 51–75 | .146 | .002 | 641 | 269 | .121 | .023 | .299 | .291 | .76 | .72 |
| 76–100 | .138 | .002 | 618 | 253 | .103 | .018 | .315 | .308 | .80 | .76 |
| 101–125 | .175 | .001 | 691 | 252 | .124 | .014 | .270 | .286 | .81 | .80 |

- **Truncation.** G14 stops hitting the 2048-token limit within about 30 steps: .106 → .001. G13 stays at .14–.18 throughout.
- **Loops.** The loop share falls by a factor of 7 (.096 → .014). G13's stays at .10–.13.
- **Length.** Completions get much shorter, 557 → 252 tokens, close to the SFT model's 260 on gen_test.
- **Speed.** Steps are 38% faster: 127 s against G13's 204 s.
- **Reward.** Training-rollout cvf tracks G13 within ±.015 in every bin, so the penalty took the loops away without costing reward.
- **Zero-variance groups.** G14's early advantage (.42 vs .73) was transient. From step 50 both arms sit at .70–.80, because most groups are all-0 or all-1 on cvf.
- **Entropy.** G14 .28 → .42; G13 .135 → .058. Truncation is now ~0 in G14, so masking no longer distorts the comparison. Length still confounds it, though: the boilerplate tokens of long completions have low entropy. Read it as "G14 is not collapsing", not as a quantitative gap.
- **Still open.** Whether G14 improves *held-out* validity: the clean gate and gen_test at step 250, around 13:00 on 2026-10-02.

**Side observation (L1 arms, same table in §3).**
- L1_correct is now truncating 25% of its rollouts (from 4%), with a mean length of 1010 tokens, but only 1.3% of rollouts are loops. Its truncations are long natural-language reasoning, not loops. Because they are masked, they get no gradient.
- L1_cvf's loop share keeps climbing, .047 → .182 by step 658, with 19% truncated.
- Neither L1 arm was launched with the fix, and both keep running unchanged for comparability with the convergence question.
