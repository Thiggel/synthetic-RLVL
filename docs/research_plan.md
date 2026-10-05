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
| G5 *(user 2026-09-28)* | P-mix, formal | dense line credit, "(#grammatical lines + #valid lines) × (1 + correctness)", hardened against padding (`formal_rewards.py`, arm `lines`) |
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
- 2026-09-28: **Stage-2 gate failed.** Greedy tagged in_system on the Dolci-RL gate set (`scripts/build_rl_gate_set.py`, 950 held-out prompts) is 0.2–0.3% for 0.8B p25/p50 and 2B p25/p50 (`reports/2026-09-28_stage2_gate.md`).
  - Causes: most math/DAPO problems are outside the system (goal-line notation); prose justifications; bad `subst` citations.
  - Remedy chosen: expert iteration (rejection-sampling SFT on the policy's own in-system proofs for RL training prompts, plus replay of the p50 mixture), repeated until the gate passes. Generator families for word problems are the fallback.
  - The Stage-2 RL subset drops DAPO. Sampling probe: `scripts/rl_signal_probe.py`.
  - Answer extraction now accepts `**Answer:** x` / `the answer is x`, and finished runs were re-scored. This matters for the G0 correctness reward on natural-language outputs.
  - GRPO infrastructure is ready: `.venv_rlvl_grpo` (TRL 1.14), `scripts/grpo_formal.py`, `scripts/formal_rewards.py`, `scripts/slurm/jobs/grpo_formal.slurm`. Smoke job 6934 ran end to end with zero reward variance. The venv lacks causal_conv1d, so the conv kernel is slow.
- 2026-09-28: **G5 added** (user: "if it only works partially, then perhaps it will start working if we use RL rewards that reward grammaticality + validity + correctness or even (#grammatical-lines + #valid-lines) * (1 + correctness)").
  - Hardened version: counts only the distinct lines the conclusion depends on; `given` lines earn no validity credit; capped at 8; validity bonus. So `((g+v)/2 + valid)(1+correct)/4`. The literal formula is kept as `lines_raw`.
  - The gate has not passed, but GRPO pilots G0/G1/G3/G5 on 2B (6949/6942/6948/6947, 200 steps; 6940/6941/6943 OOMed) run in parallel with EI. They test whether the dense arms give signal before EI does. G2/G4 wait: only 1% of groups have signal.
- 2026-10-04: **Stage 3 started; EI stopped after round 4** (user: "Stopping to distill is alright, but should we not just go on to alphazero for the moment then?").
  - EI rounds e1–e4 compounded validity (gate greedy strict-valid .227 → .415 from e3 to e4) but left correctness flat at ~.26 (`reports/2026-09-30_libext_and_gate_failures.md` §12). No e5.
  - Deviation: AZ-formal starts from e4 (`qwen35_2b_lc_libext_e4_lr5em6_seed3407/final`), not the Stage-1 best-X model. The Stage-1 model almost never produces in-system proofs on Dolci, so search would have nothing to prune. AZ-NL starts from the X=0 model.
  - The full G0–G4 grid is deferred. Running GRPO baselines: L1_correct / L1_cvf (correct-only vs correct·valid, 5k steps; these are G1 and G2 at 2B), G17, G18.
  - First build: a value head on e4 trained on logged GRPO completions (terminal cvf / correct). Then line-level PUCT reusing the checker pruning of `scripts/guided_decode.py`. Lesson from guided decoding (`reports/2026-10-02_guided_decoding.md`): pruning alone doubles validity but not correctness, so the value target must include correctness.
- 2026-10-05: **Stage 3 switched to online AlphaZero with a jointly trained value head** (user: "i would really wish that we just let the model explore so that it finds solutions and thus learn a value head and model online, both ntp and value head on top of the hidden state").
  - `scripts/az/az_train.py`: line-level PUCT self-play. Search and training share one process and one GPU: vLLM in colocate sleep mode, with weight sync after every iteration. The prior is the policy LM's line samples; the value is a linear head on the last hidden state (initialised from the e4 cvf probe). Terminal reward is gold cvf (correct · valid · premises stated). Checker-dead lines count as 0.
  - Losses on each iteration's search data (plus a 2-iteration replay buffer):
    - policy cross-entropy toward visit counts over sibling lines;
    - value BCE on states: the trajectory outcome z mixed with the root Q, terminal children → gold, dead children → 0, visited children → Q;
    - NTP on found correct proofs.
  - Memory: bf16 weights on the GPU, with fp32 master weights and AdamW on the CPU (peak ≈ 23 GB; the gruenau L40s are shared).
  - Init from e6 (6 EI rounds). Pool `rlvl_data/az/online_pool/pool.jsonl`: 7708 GSM8K-train / Dolci prompts on which e6 sometimes fails. Run: `rlvl_data/az/online_e6_r1`, chained 2-day jobs.
  - Metrics: z, found rate, online value AUC (root value vs z), and the gate_subset_300 greedy eval every 4 iterations.
