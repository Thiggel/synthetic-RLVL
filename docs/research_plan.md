# Research Plan: Checkable Formal Reasoning → RL → AlphaZero for LMs

Written 2026-09-28 from the user's stated plan. **This is the plan to follow.**
- **User's own wording and hypotheses:** kept verbatim in quotes and are binding.
- **Items marked *(impl)*:** implementation choices by Claude. They may be adjusted when evidence demands it, and every change gets a dated note at the bottom of this file.

Standing constraints:
- At most half the GPUs on any node for laitenbf.
- Keep the Slurm dashboard docs updated (see AGENTS.md).
- The reasoning language is the RLVL-next system (`<formal>` tag, Rust checker `rlvl.check`, generator `rlvlgen`).

---

## Stage 1: Mixture sweep (running now)

> "we find the best percentage of our training data and how much (1) accuracy, (2) validity, (3) grammaticality it induces on all kinds of benchmarks and all kinds of models"

**Setup:**
- **Models:** Qwen3.5-Base 0.8B, 2B and 9B. "7B" in the plan means the 9B.
- **Data:** Dolci-100k instruction data, with X% replaced by rlvlgen proofs.
- **Mix levels:** X ∈ {0, 5, 10, …, 50}. For 9B only X ∈ {0, 10, 25, 50}, because a run takes about 17 h.
- **Baseline:** X=0, pure instruction data.

**Metrics per model × X:**
1. **In-domain unseen synthetic problems:** faithful prompt→premises, grammatical, valid, answer accuracy (`formal_eval/`).
2. **Untagged downstream suite:** tests whether the skill transfers (`lm_eval_results/formal_mix_20260926/`):
   - PW d0–3, d5, PW CoT
   - FOLIO
   - GPQA-Diamond and GPQA-quant
   - BBH (all, web of lies)
   - HotpotQA, 2Wiki, MuSiQue
   - GSM8K, MMLU, ARC-C, LogiQA, HellaSwag, PIQA, WinoGrande, HumanEval, MBPP
3. **Tagged suite:** the same benchmarks asked with `<formal>` (`formal_bench_tagged/`). It measures grammatical %, valid %, correct % and in_system % (valid and the proof's own answer is right).

**Output:**
- **Best X per model size.** *(impl)* Choose it by the highest tagged in_system rate, subject to losing no more than about 2 points on the untagged general average (GSM8K, MMLU, ARC-C, BBH) relative to X=0. Report the full trade-off curve either way.
- **Plots and tables** in `analysis/formal_mixture_sweep_20260925/`.

**Gate to Stage 2:** the chosen model must have a tagged in_system rate clearly above 0 on the RL prompt distribution; measure this on a held-out sample of the Stage-2 RL data.
- If it is about 0, the validity rewards give no learning signal (cold start).
- *(impl)* Fixes in that case: raise X, add a short SFT on generator problems that look like the RL data, or improve the generator families. Do this before running Stage 2.

---

## Stage 2: GRPO on the Olmo 3 RL data with different rewards

> "we take one such model (e.g. 7B) and RL train it with GRPO on the Olmo3 RL training data. We try this with different rewards"

**Policies:**
- **P-mix:** the best-X model from Stage 1, prompted with the `<formal>` tag.
- **P-base:** the X=0 model (100% instruction tuning), prompted normally.

**Reward arms:** the user's list is binding.

| Arm | Policy | Reward |
|---|---|---|
| G0 (baseline) | P-base, no formal tag | correctness only |
| G1 | P-mix, formal | (1) correctness only |
| G2 | P-mix, formal | (2) correctness × validity (1 only if both are true) |
| G3 | P-mix, formal | (3) grammaticality + validity + correctness |
| G4 | P-mix, formal | (4) validity only, without correctness |
| G1-NL *(impl, optional)* | P-mix, no tag | correctness only. Separates "mixture SFT" from "reasoning in the format". |

**User hypothesis:**
> "validity reward will make training more sample efficient and give larger gains than correctness only for ambiguous problems cause it teaches to reason instead of just doing anything that gets us correctness"

**Testable predictions:**
- G2 and G3 should reach G1's final accuracy in fewer samples.
- The gap G2 − G1 should be largest on *ambiguous* items, where correctness can be hit by guessing: yes/no, true/false/unknown and multiple-choice. It should be smallest on free-form numeric items.
- *(impl)* Split every result by answer type (binary / MC / numeric / free-text) to test this directly.

**Implementation choices *(impl)*:**
- **RL data:** the Olmo 3 / Dolci RL mix. Verify the exact HF id locally.
  - Use subsets with a checkable final answer: math, logic, verifiable QA.
  - Code and chat are excluded from the reward because they cannot be answered in the system. Record the fraction excluded.
- **Validity:** `rlvl.check` strict ok, plus grounding: at least one `given`, no quote errors, and the proof's `ans` equals the Answer tag.
  - Without grounding, G4 can be hacked with a valid but irrelevant proof.
  - Track valid_wrong (valid proof, wrong answer) as the unfaithful-formalization signal, especially for G4.
- **Framework:** verl or TRL GRPO, whichever runs on our venvs. The Rust checker is the reward function (fast, CPU).
  - First iterate on 2B.
  - Confirm on 9B with the same hyperparameters.
  - The group size, KL, learning rate and number of steps are identical across arms.
- **Metrics:**
  - Accuracy and in_system rate vs. number of RL prompts and samples consumed (sample-efficiency curves).
  - The Stage 1 benchmark suites (tagged and untagged) at fixed checkpoints.
  - 3 seeds for the main arms if compute allows.
- **Compute:** gruenau, guppi, and "alex" (back online), under the half-per-node rule.

---

## Stage 3: AlphaZero on a language model

> "our model sees the prompt and then only reasons until it outputs an answer in answer tags right. + our reasoning system has very low vocabulary and branching factor. These are nice prerequisits to try the alphazero algorithm on a language model (usually vocabulary size is way too large). I want you to train models with alphazero (our reasoning format + alphazero + varying rewards) and compare them to baselines"

**Arms:**
- **AZ-formal:** our reasoning format + AlphaZero, with the same reward variants as Stage 2 (correctness, correctness × validity, gram + valid + correct, validity only).
- **Baseline (1):** the Stage-2 GRPO models, with and without our reasoning format, across reward variants.
- **Baseline (2):** AlphaZero trained in normal natural-language reasoning (AZ-NL).

**User hypothesis:**
> "our reasoning system will make alphazero work on language models and thus will make training so much more sample efficient."

**Design *(impl)*:**
- **State and actions:**
  - The state is the prompt plus the proof prefix.
  - The action is the next token, restricted by the grammar Guide (constrained decoding). Symbols are single tokens; quotes are restricted to prompt spans.
  - Also try step-level actions (one proof line = one action, candidates sampled from the policy), because token-level trees can be deep.
  - AZ-NL must use the equivalent granularity (token or sentence/line) so the comparison is fair.
- **Legal moves:** the checker validates each line as soon as it is written, so invalid lines can be pruned like illegal moves in Go. NL has no such rule set, and this is the mechanism behind the hypothesis. Also run an ablation without pruning to isolate it.
- **Network:**
  - The policy is the LM.
  - The value head sits on the last hidden state and predicts the terminal reward.
  - Priors come from the constrained softmax.
- **Training loop:**
  - Self-play on the RL prompts with MCTS (PUCT, N simulations).
  - Policy target is the visit distribution; value target is the terminal reward.
  - Periodic evaluation.
  - Start from the Stage-1 best-X model (formal) or the X=0 model (NL).
- **Comparisons:**
  - Hold the number of RL prompts and the generated tokens/FLOPs equal.
  - Report accuracy, valid %, in_system % vs. samples and vs. compute.
  - Add an expert-iteration baseline (best-of-N filtered by reward → SFT) as a cheap non-MCTS search baseline. This shows whether gains come from MCTS or just from search plus filtering.
- **Start small:** 0.8B/2B on generator problems and the logic/math RL subsets. Scale to 9B only if AZ-formal beats GRPO on the small scale.

**Predicted ordering to test:** AZ-formal > GRPO-formal (validity rewards) > GRPO-formal (correctness) ≳ GRPO-NL ≈ AZ-NL, in sample efficiency.

---

## Order of work

1. Finish Stage 1: sweep, untagged suite, tagged suite for 0.8B/2B/9B. Pick the best X, and check the Stage-2 gate.
2. Stage 2 on 2B:
   - GRPO infra plus the checker-based reward.
   - Smoke run.
   - Arms G0–G4 (+G1-NL).
   - Sample-efficiency curves, split by answer type.
   - Then 9B for the main arms.
3. Stage 3 on 0.8B/2B:
   - MCTS plus a value head on top of the Guide.
   - Smoke run on generator problems.
   - AZ-formal vs AZ-NL vs expert iteration vs GRPO.
   - Then scale.

## Reporting (binding, user 2026-09-28)

> "always log everything we find in reports with nice figures"

Every finding, including negative results, goes into a report with figures, not only into logs or tables:
- **Report file:** `reports/<date>_<topic>.md`, with its figures in `reports/figures/` (PNG, plus PDF for the paper).
- **Figures:** sample-efficiency curves, mix-% trade-off curves, per-benchmark bars, and example proofs.
- **Also** add a short entry in `docs/project_log.md` that links the report.
- **Publish** each report as a private artifact page so it can be shared.

## Change log
- 2026-09-28: plan written.
