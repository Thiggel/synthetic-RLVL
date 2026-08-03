# Nanotron Mixture-Schedule Audit (2026-07-10)

## Production update (2026-08-04 01:40 CEST)

Terminal matched-NL continuation `3875833_2` started at 23:40 CEST on eight
verified A100-80GB GPUs on `a0933`. It restored the accepted step-5000 TP4/DP2
model, optimizer, scheduler, and RNG state plus exact
`5000/640000/2621440000` step/sample/token offsets. The resumed dataloader
records exact `2490368000` Dolmino and `131072000` matched-NL tokens, and the
live Qwen2.5 config retains RoPE `1000000`.

Training is healthy through logged step `5371/9537` at about 31.0K tokens/s
with finite loss and gradient diagnostics and an ETA near 19.6 hours. Disabled
W&B initialized without launching the prior failing service path. No OOM,
quota, restart, scheduler, or device failure signature is present. Direct eval
`3944017_2` and post-SFT LR pilot `3944071_[0-1]` remain correctly held on
successful terminal completion. Audit the first new complete checkpoint before
any guarded rotation. Successor watcher `3948110` is dependency-free,
CPU-only/no-GRES, and scheduled for 07:33 CEST; preserve it because the
terminal, direct-eval, and post-SFT paths remain incomplete.

## Production update (2026-08-03 19:40 CEST)

Replacement full-parameter FSDP smoke `3945777_0` completed `0:0` in
`00:27:11` on four A100-80GB GPUs. Its complete step-9537 control checkpoint
gate passed after optimizer offload, all 32 training steps and both eval passes
were finite, and held-out loss improved `0.93337 -> 0.92020`. The smoke
serialized 44 files totaling `152,344,783,631` bytes, wrote an accepted audit,
and removed the temporary output through the guarded path.

Slurm consumed the fulfilled smoke dependency from `3944071_[0-1]`; the LR
pilot now waits only on terminal NL `3875833_2`. That job remains
dependency-free and priority-pending for a current 02:19 CEST start with its
accepted 645-file step-5000 restart intact. Matched direct eval `3944017_2`
waits on the same terminal state. No safe scheduler edit is useful while all
compatible A100-80GB nodes are allocated, completing, or draining. Successor
watcher `3947425` remains required because the terminal, LR-selection, and
treatment post-SFT paths are incomplete.

## Production update (2026-08-03 14:10 CEST)

After accepted terminal direct readouts, control/formal step-9537 optimizer
states remain potentially necessary for the predeclared 10B--20B continuation
gate but do not need to occupy Vault. Job `3946030` moves their eight shards
(`365,583,141,888` bytes) to
`$WORK/synthetic-RLVL/checkpoint_state_offload/nanotron_dolmino_5b`, leaves
symlinked optimizer directories at the original checkpoints, and reruns both
full checkpoint audits. Post-SFT smoke `3945777_0` now waits for this gate.
The live NL step-5000 optimizer and all current Dolmino Nanosets/model shards
remain untouched. Separate immediate cleanup restored user Vault from
`949G` to `924G/1000G`; see `analysis/storage_audit_20260803.md`.

## Production update (2026-08-03 13:40 CEST)

The control post-SFT full-model smoke `3944070_0` failed `126:0` only after
the complete step-9537 TP4/DP2 checkpoint audit and Nanotron-to-HF conversion
succeeded. The failure was a stale `torchrun` shebang in
`$HPCVAULT/.venv_rlvl_posttrain` pointing to a deleted Atuin Python, before
FSDP training began. The Slurm wrapper now invokes the verified venv Python as
`python -m torch.distributed.run`; `bash -n`, module-launch, and Slurm
test-only validation pass.

Exact 32-step cleanup replacement `3945777_0` preserves the original
1,024-train/128-eval, four-A100-80GB smoke protocol and is priority-pending.
Control LR rows `3944071_[0-1]` were dependency-repaired to
`afterok:3945777:3875833_2`. Terminal NL `3875833_2` remains unchanged and
dependency-free; matched terminal eval `3944017_2` remains held. Current
watcher `3943787` is running CPU-only/no-GRES and successor `3945763` is
preserved because terminal and post-SFT acceptance remain incomplete.
The failed smoke epilogue measured user Vault at `994.1G/1048.6G` soft and
about `181k/200k` files; the project subtree is about `833G`. No protected or
pending-job-referenced state was removed in this pass. Guarded rotation remains
mandatory before terminal and post-SFT outputs accumulate.

## Production update (2026-08-03 07:36 CEST)

Terminal NL continuation `3875833_2` remains dependency-free and
priority-pending with the required one-node TP4/DP2, eight-A100-80GB request.
Slurm now projects `12:58:36` CEST on `a0535`. Every healthy A100-80GB node
is allocated or mixed, so no compatible partition widening or resource edit
can improve the scientifically fixed topology.

The accepted step-5000 restart remains intact at 645 files, zero empty files,
and `106,628,387,172` bytes. Its accepted audit still records exact
`5000/640000/2621440000` step/sample/token offsets and exact
`2490368000 + 131072000` Dolmino/NL tokens. Pending-job stdout/stderr are
correctly absent. Vault remains about `948G/1000G`, 182k files; group Work is
about `471k/500k` files.

Current watcher `3943558` is running CPU-only/no-GRES. Its recorded successor
`3943787` is dependency-free, CPU-only/no-GRES on `a100mig`, and BeginTime-held
for `13:32:24` CEST. The terminal and matched direct/post-SFT plan remains
incomplete, so the successor is preserved. No broader mixture grid or report
trigger is open.

## Production update (2026-08-03 01:36 CEST)

Matched NL step-5000 direct readout `3942598_2` completed `0:0` in `00:54:31`
on A40 node `a1721`. Its local four-shard HF conversion and finite forward
pass succeeded, both legacy and modern consumers resolved Qwen2.5 RoPE
`1000000`, and the production audits accepted six multi-hop files/600 rows and
105 standard-suite leaf files/10,600 rows. The schema-v4
`answer_prefix_math_verify,none` sidecar accepted MATH-500 at `0.3300` with
zero lost stock-exact positives.

The matched-NL ten-task macro is `0.6327`, versus control `0.5932` and formal
`0.6341`. Stock/tagged/first-head multi-hop macros are
`0.1067/0.3420/0.3863`. Stock continuation is `93%/94%/96%` on
2Wiki/HotpotQA/MuSiQue, essentially the formal `91%/94%/97%` pattern. BBH and
MMLU-Pro invalid extraction are `5.56%` and `5.00%`; correct and incorrect raw
review found intended prompts/extraction, genuine task errors, tagged-boundary
adherence, and long repetitive invalid MMLU-Pro tails. No assistant-preamble
marker appeared. This resolves the intermediate causal gate in favor of the
shared long-document/full-loss intervention, not formal syntax. It does not
trigger a broader mixture grid or preprint update. Full detail:
`analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.

Terminal NL continuation `3875833_2` remains priority-pending from the audited
step-5000 state, with a provisional 16:59 CEST start. The accepted checkpoint
still has 645 nonempty files. Vault is about `948G/1000G`, 182k files; group
Work is about `471k/500k` files. Successor oversight `3943558` is
dependency-free, CPU-only/no-GRES on `a100mig`, and scheduled for 07:31 CEST.

## Production update (2026-08-02 19:44 CEST)

Exact from-base NL retry `3875832_2` reached step 5061 and ended at the planned
24-hour boundary with Slurm state `TIMEOUT`, exit `0:0`, and no OOM, quota, or
unexpected fatal signature. Its step-5000 checkpoint passes the full restart
gate: 645 files, no zero-byte file, TP4/DP2, 625 model files, four equal
optimizer shards, four scheduler shards, eight RNG shards, Qwen2.5 RoPE
`1000000`, and exact step/sample/token offsets
`5000/640000/2621440000`. Dataset offsets are exact `2490368000` Dolmino plus
`131072000` matched-NL tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step5000_20260802.json`.

Only after acceptance were superseded steps 3500/4000/4500 removed,
reclaiming `319,885,161,515` bytes. Step 5000 is the sole numeric NL restart
state and passes a post-delete re-audit unchanged; Vault is about
`948G/1000G`, 182k files. Terminal continuation `3875833_2` is
priority-pending and must resume this state. The matched limit-100 NL
step-5000 direct readout was added to the existing control/formal wrapper and
submitted as A40 job `3942598_2`; it preserves the standard reviewer suite,
32,768-window tagged/stock multi-hop protocol, retained generations, RoPE
preflight, production audits, and corrected MATH sidecar gates.

## Production update (2026-08-02 13:40 CEST)

Exact from-base NL retry `3875832_2` remains healthy beyond step 3861 on eight
A100-80GB GPUs at about 31.1K tokens/s with finite loss and gradient
diagnostics. Its complete step-3500 checkpoint passes the full restart gate:
645 files, no zero-byte file, TP4/DP2, 625 model files, four equal optimizer
shards, four scheduler shards, eight RNG shards, Qwen2.5 RoPE `1000000`, and
exact step/sample/token offsets `3500/448000/1835008000`. Dataset offsets are
exact `1743257600` Dolmino plus `91750400` matched-NL tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step3500_20260802.json`.

Only after acceptance were superseded steps 2500/3000 removed, reclaiming
`213,256,774,342` bytes. Step 3500 is the sole numeric NL restart state and
passes a post-delete re-audit unchanged. User Vault usage is about
`1068G/1000G`, `182k/200k` files. Continue guarded latest-complete-state
rotation, but preserve and audit step 5000 for the matched limited readout.

## Production update (2026-08-02 07:35 CEST)

Exact from-base NL retry `3875832_2` remains healthy beyond step 2581 on eight
A100-80GB GPUs at about 31.1K tokens/s with finite loss and gradient
diagnostics. Its complete step-2500 checkpoint passes the full restart gate:
645 files, no zero-byte file, TP4/DP2, 625 model files, four equal optimizer
shards, four scheduler shards, eight RNG shards, and exact step/sample/token
offsets `2500/320000/1310720000`. Dataset offsets are exact `1245184000`
Dolmino plus `65536000` matched-NL tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step2500_20260802.json`.

Only after acceptance were superseded steps 1000/1500/2000 removed,
reclaiming `319,885,161,508` bytes. Step 2500 is the sole numeric NL restart
state and passes a post-delete re-audit unchanged. User Vault usage is about
`948G/1000G`, `182k/200k` files. Continue guarded latest-complete-state
rotation, but preserve and audit step 5000 for the matched limited readout.

## Production update (2026-08-02 01:35 CEST)

Exact from-base NL retry `3875832_2` remains healthy beyond step 1301 on eight
A100-80GB GPUs at about 31.1K tokens/s with finite loss and gradient
diagnostics. Its complete step-1000 checkpoint passes the full restart gate:
645 files, no zero-byte file, TP4/DP2, 625 model files, four equal optimizer
shards, four scheduler shards, eight RNG shards, Qwen2.5 RoPE `1000000`, and
exact step/sample/token offsets `1000/128000/524288000`. Dataset offsets are
exact `498073600` Dolmino plus `26214400` matched-NL tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_nl_exact_step1000_20260802.json`.

Only after acceptance was superseded step 500 removed, reclaiming
`106,628,387,164` bytes. Step 1000 is the sole numeric NL restart state and
passes a post-delete re-audit unchanged. User Vault usage fell from about
`1266G` to `1068G` against the 1,000G soft quota; continue the guarded
latest-complete-state rotation and preserve the future step-5000 state for the
matched limited readout.

## Production update (2026-08-01 19:35 CEST)

Formal continuation `3875829_1` completed `0:0` at terminal step 9537 in
`21:37:48`. Its terminal checkpoint passes the full restart gate: 645 files,
no zero-byte file, TP4/DP2, 625 model files, four equal
`22,848,937,060`-byte optimizer shards, four scheduler shards, eight RNG
shards, Qwen2.5 RoPE 1000000, and exact step/sample/token offsets
`9537/1220736/5000134656`. Dataset offsets are exact `4750127104` Dolmino
plus `250007552` formal tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step9537_20260801.json`.

Only after acceptance was redundant unstarted formal stage `3875830_1`
canceled and superseded formal steps 8000/8500/9000/9500 removed, reclaiming
`426,513,548,736` bytes. Step 9537 is the sole numeric formal restart state
and passes a post-delete re-audit unchanged. Vault settled to `869G/1000G`,
181k files. Exact from-base NL retry `3875832_2` started on eight A100-80GB
GPUs, loaded the untouched base, built the exact 0.95/0.05 blend, and is
healthy beyond step 41 at about 31.2K tokens/s with finite diagnostics and no
repeated W&B-service failure. Protect and audit its step-5000 state before the
matched limited readout.

## Production update (2026-08-01 13:35 CEST)

Formal continuation `3875829_1` remains healthy beyond step 8331 on eight
A100-80GB GPUs with about 31.0K tokens/s and finite diagnostics. Its complete
step-8000 state passes the full restart gate: 645 files, no zero-byte file,
TP4/DP2, 625 model files, four equal `22,848,937,060`-byte optimizer shards,
four scheduler shards, eight RNG shards, and exact step/sample/token offsets
`8000/1024000/4194304000`. Dataset offsets are exact `3984588800` Dolmino
plus `209715200` formal tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step8000_20260801.json`.

Only after acceptance were superseded formal steps 7000/7500 removed,
reclaiming `213,256,774,362` bytes. Step 8000 is the sole numeric formal
restart state and passes a post-delete re-audit unchanged. The project Vault
tree is `753G` with 8,755 files. The job remains projected to finish within
its current allocation. NL retry `3875832_2` remains account-GRES pending.

## Production update (2026-08-01 07:33 CEST)

Formal continuation `3875829_1` remains healthy at step 7051 on eight
A100-80GB GPUs with about 31.0K tokens/s and finite diagnostics. Its complete
step-7000 state passes the full restart gate: 645 files, no zero-byte file,
TP4/DP2, 625 model files, four equal `22,848,937,060`-byte optimizer shards,
four scheduler shards, eight RNG shards, and exact step/sample/token offsets
`7000/896000/3670016000`. Dataset offsets are exact
`3486515200` Dolmino plus `183500800` formal tokens, or 95:5. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step7000_20260801.json`.

Only after acceptance were superseded formal steps 5500/6000/6500 removed,
reclaiming about 319.9 GB. Step 7000 is the sole numeric formal restart state;
Vault is `869G/1000G`, 181k files. The job remains projected to finish within
its current allocation. NL retry `3875832_2` remains account-GRES pending.

## Production update (2026-08-01 01:33 CEST)

Control continuation `3875826_0` completed `0:0` at step 9537. Its terminal
checkpoint passes the complete 645-file, zero-byte, TP4/DP2
model/optimizer/scheduler/RNG gate with exact offsets
`9537/1220736/5000134656`, all from Dolmino. Formal continuation
`3875829_1` started on eight A100-80GB GPUs, restored the accepted step-5000
state and exact sampler offsets, and reached step 5781 at about 31.0K tokens/s
with finite diagnostics. Its complete step-5500 state passes the same gate and
records `2739404800` Dolmino plus `144179200` formal tokens, exactly 95:5.
Audits are
`analysis/nanotron_checkpoint_audits/dolmino_control_step9537_20260801.json`
and `analysis/nanotron_checkpoint_audits/dolmino_logic_step5500_20260801.json`.

Only after those acceptances were superseded control steps 9000/9500 and
formal step 5000 removed, reclaiming `319,885,161,145` bytes. Terminal control
9537 and live formal 5500 are the sole numeric restart states; Vault returned
from `1465G/1000G` to `869G/1000G`, `181k/200k` files. Redundant unstarted
control stage `3875827_0` was canceled after terminal acceptance. NL retry
`3875832_2` remains account-GRES pending.

## Production update (2026-07-31 19:32 CEST)

Control continuation `3875826_0` reached iteration `9071/9537` at about
31.1K tokens/s with finite loss and gradient diagnostics. Step 9000 passed
the complete restart gate: 645 files, no zero-byte file, TP4/DP2, 625 model
files, four equal `22,848,937,060`-byte optimizer shards, four scheduler
shards, eight RNG shards, and exact step/sample/token offsets
`9000/1152000/4718592000`, all charged to the normal Nanoset. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_control_step9000_20260731.json`.
Only after acceptance was superseded step 8500 removed, reclaiming
`106,628,386,982` bytes. Step 9000 is the sole numeric control restart state;
Vault is `748G/1000G`, `181k/200k` files. Formal continuation
`3875829_1` and exact from-base NL retry `3875832_2` remain account-GRES
pending.

## Production update (2026-07-31)

Control continuation `3875826_0` started at 00:05 CEST on eight A100-80GB
GPUs. Startup restored the audited step-5000 model, optimizer, scheduler, RNG,
sample offset `640000`, and token offset `2621440000`, then began at iteration
5001. It reached iteration 6541 with finite loss/gradient diagnostics and
about 31.1K tokens/s. A live process inspection found no `wandb-core` or W&B
service process, so the disabled/no-op logging repair is effective.

Step 6500 independently passed the complete restart gate: 645 files, no
zero-byte file, TP4/DP2, 625 model files, four equal
`22,848,937,060`-byte optimizer shards, four scheduler shards, eight RNG
shards, and exact step/sample/token offsets
`6500/832000/3407872000`, all charged to the normal Nanoset. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_control_step6500_20260731.json`.
Only after acceptance were superseded complete steps 5000/5500/6000 removed,
reclaiming `319,885,160,905` bytes. Step 6500 is the sole numeric control
restart state and Vault returned from `1344G/1000G` soft to `748G/1000G`,
`181k/200k` files. Formal continuation `3875829_1` and exact from-base NL
retry `3875832_2` remain account-GRES pending.

## Production update (2026-07-28)

The matched 5B Dolmino control and formal runs each reached an audited
step-5000 state (`2,621,440,000` consumed tokens); their continuations are
pending. The first NL allocation failed before optimizer step 1 because W&B
offline mode attempted to start a local service and timed out waiting for its
port file. The shared production wrapper now defaults to
`WANDB_MODE=disabled`. This changes only experiment logging: installed W&B
source maps disabled mode to a no-op run and does not launch the service.
Pending outer Slurm jobs call the shared wrapper from disk at runtime, so the
fix applies without resubmission or loss of queue priority. Both wrappers pass
shell syntax validation. The next allocation must confirm disabled/no-op W&B
before accepting optimizer progress.

## Conclusion

The corrected Qwen2.5-7B continual-pretraining pilot implements the intended matched
token mixture. Nanotron blends fixed 4,096-token packed chunks. For each 15%
proof condition, the full 8,192-step schedule contains 157,287 proof chunks
and 891,289 normal-text chunks, corresponding to 644,247,552 proof tokens out
of 4,294,967,296 total tokens (`15.000057%`). Logic and NL therefore receive
the same proof-token exposure to within the identity of the source corpus.

The source blend, capacity, and data-offset resume are correct. A later
scheduler audit did find a separate Nanotron resume bug: after loading the
optimizer, the scheduler builder normalized its lambda by the checkpoint's
current LR instead of the preserved original LR. All three matched runs
therefore jumped from about `5.94e-6` before step 4096 to `6.25e-6` after
resume. The bug is identical across control, logic, and NL and does not explain
their relative ordering, but it must be fixed before another training wave.
The three-step integration smokes realize one proof chunk out of three
(`33.3%`) because they are intentionally too small to represent 15% finely;
this smoke-only granularity does not apply to production.

## Terminology and next background corpus

The completed experiment uses FineWeb-Edu, a broad pretraining corpus, as its
background. It is therefore a continual-pretraining pilot, not a midtraining
experiment. The scientifically appropriate replacement is Dolma 3 Dolmino:
the 100B release is the mixture used for OLMo 3 7B's second training stage,
and the 10B release is the official micro-anneal mix. The 10B release is enough
for a 4.3B-token no-repeat pilot and avoids downloading the 180GB full release.

Representative released Dolmino records show that the mixture has no single
chat or reasoning wrapper. Its sources preserve heterogeneous plain-text
formats, including raw web prose, question-answer text, `User:`/`Assistant:`
dialogue, reading-comprehension records, and long reasoning solutions. The
common contract is a `text` document followed by EOS. Future preprocessing
must preserve those records rather than rewriting all Dolmino sources into an
artificial schema.

Only the paired formal and NL intervention records need an identical outer
format. Use a modality-neutral document such as:

```text
{problem}

Solution:
{formal or natural-language trace}

Final answer: {answer}
```

Do not include `<formal>`, `<think>`, modality names, or modality-specific
outer tags. Append the same Qwen EOS token used for Dolmino records, tokenize
all sources with the Qwen2.5 tokenizer, and pack them identically at 4,096
tokens. Formal versus NL content should differ only inside the trace field.

Mixture percentages must replace Dolmino tokens, not add extra training:
control is 100% Dolmino, while an intervention at `x%` is `(100-x)%` of the
same Dolmino stream plus `x%` paired formal or NL tokens. Use matched Dolmino
sample indices/order, total steps, and token counts across conditions. Since
the official 100B mix already contains about 8.3% thinking data, the old
15--25% single-generator injections are aggressive. The next gate should be a
bounded `{0,2,5,10}%` pilot before considering larger fractions.

### Submitted LR and data gate (2026-07-15)

The released Dolmino `default` Hugging Face config cannot be streamed through
the installed `datasets` JSON loader because source metadata schemas differ.
The production exporter therefore reads the repository's 1,113 JSONL.zst
shards directly, shuffles the shard order with seed 42, retains only native
`text`, and records realized Qwen-token counts by source. Build array
`3858584_[0-2]` creates 4.8B Dolmino tokens and 550M tokens for each neutral
paired proof modality.

Before full training, one shared LR is tuned on 100% Dolmino. Sequential gate
`3859297` compares `6e-6`, `3e-6`, and `1e-5` for 256 steps each on identical
chunks in one 12-hour 8xA100-80GB allocation, with 32 warmup steps and constant
LR thereafter. Independent alternatives `3859711_[0-2%3]` run each LR on
4xA100-80GB with TP4/DP1, microbatch 4, and accumulation 32. This preserves
global batch 128 and 134,217,728 tokens per row relative to TP4/DP2,
accumulation 16 on eight GPUs. Keep both scheduling paths until one can cover
all rows, then cancel redundant unstarted work. Earlier `3858587/3858588` and
dependency-dead `3858902` were canceled before start.
Each row consumes 134,217,728 tokens and saves no model/optimizer checkpoint.
The selected shared LR must then pass 256-step formal-5% and NL-5%
confirmations. Full 4.3B-token runs are not submitted before those checks.

### Staged 20B production design (2026-07-15)

The eventual production experiment should use one continuous 20B-token
schedule with evaluation pauses rather than independently optimized endpoints:

| Milestone | Optimizer step | Realized tokens |
| --- | ---: | ---: |
| 5B | 9,537 | 5,000,134,656 |
| 10B | 19,073 | 9,999,745,024 |
| 15B | 28,610 | 14,999,879,680 |
| 20B | 38,147 | 20,000,014,336 |

The scheduler must be configured for the final 38,147-step horizon from step
1, with the selected shared peak LR, 256 linear warmup steps, and cosine decay
to approximately `1e-6`. At 5B, pause all three conditions, export immutable
BF16 weight snapshots, and run identical direct and post-instruction readouts.
Instruction tuning branches from the exported weights and never modifies the
resumable base state. Continue control, formal-5%, and NL-5% together only if
a predeclared 5B gate finds a nontrivial, directionally coherent downstream
difference without a material broad-capability regression. A practical pilot
gate is at least one absolute percentage point on the preregistered reasoning
macro, directionally supported across task groups, with no greater than one
point all-primary regression. This is a continuation rule, not a substitute
for training-seed uncertainty.

For storage, retain only the latest full model/optimizer/scheduler state needed
to resume each active condition. After the next milestone is verified, export
the prior milestone as weight-only BF16 plus config/manifest and remove its
superseded optimizer state. Preserve weight-only snapshots at 5B, 10B, 15B,
and 20B; evaluate every milestone if continuation is triggered. Do not assume
the current Hugging Face account can hold all 12 snapshots: retain local
verified snapshots and upload only under an audited rotation/archive policy.

The current 10B micro-anneal release cannot provide a unique 20B stream. Build
the production stash from the official 100B Dolmino release with at least 21B
packed Qwen tokens, plus 1.1B formal and 1.1B matched-NL tokens. Build normal
data in roughly 5B shards and delete raw JSONL only after each packed shard,
manifest, EOS count, and sample audit passes. Precompute one deterministic slot
schedule over all `38,147 * 128 = 4,882,816` chunks: intervention conditions
replace the same approximately 5% normal slots with formal or NL chunks, while
all remaining normal chunk identities and order match control. This is stronger
than relying on independent weighted samplers.

At 12:44 CEST Dolmino task `3858584_0` failed after writing a resumable
`4,128,298,342/4,800,000,000` tokens because Xet/CAS returned HTTP 500 for
shuffled shard position 410. This is a transport failure, not data exhaustion,
quota, or manifest corruption. The direct-shard exporter now retries each Hub
download up to five times with bounded backoff; focused tests pass. Exact
resume `3859296_0` continued from the recorded byte offset without replay and
finished the raw export at `4,800,000,272` tokens in `10,705,908` records
across 120 source groups. Manifest source-token/record sums and five
byte-spaced raw records pass. Nanoset preprocessing completed with one
nonempty 19.24GB shard and `4,810,706,180` packed tokens. The exact
`10,705,908`-token delta confirms one appended EOS per record; metadata,
capacity, and terminal-state gates pass. The unstarted LR gate `3858902` had
become dependency-dead and was replaced by `3859297`, now normal
account-GRES pending. Matched 4-GPU row `3859711_0` (`6e-6`) completed all
256 steps and 134,217,728 tokens in `02:27:56` on 2026-07-16. It sustained
about 15.4K tokens/s, retained finite losses and gradient norms, and wrote a
terminal manifest. Row `3859711_1` (`3e-6`) then completed in `02:27:31` at
matched throughput. On the 224 identical post-warmup batches, `6e-6` has lower
loss on 210; mean paired `3e-6 - 6e-6` loss is `+0.0109`. Large-loss batches
align by step across both runs, indicating source variation rather than
LR-specific divergence. Defer selection until row 2 (`1e-5`), which started
on four A100s at 14:21 CEST. Sequential fallback `3859297` was canceled once
that independent row was confirmed running.

## Downstream acceptance gate

The installed lm-eval task registry does not contain `folio`; the original
default would therefore fail all downstream jobs during task resolution.
Legacy `logiqa` and `logiqa2` also depend on dataset scripts rejected by the
installed `datasets`; `agieval_logiqa_en` is the maintained replacement.
Diagnostic raw samples additionally showed that installed FLD prompts request
a proof while exact-match scoring expects only a class label. Because that
creates an extraction-confounded floor, both FLD surfaces are excluded from
the production transfer claim rather than patched post hoc.

The final suite is `gsm8k`, `hendrycks_math500`, `arc_challenge`,
`hellaswag`, `winogrande`, `piqa`, `agieval_logiqa_en`, `bbh`, `mmlu`, and
`mmlu_pro`. MMLU formal logic and BBH's logic subtasks provide targeted logic
readouts; the full groups and remaining tasks measure broader transfer.

Smokes `3834728/3834737` found the LogiQA incompatibilities. Diagnostic smoke
`3834738` completed direct and native-chat execution and exposed the FLD
metric problem through retained generations. Production evals are gated on
final ten-task smoke `3834836`.
The eval wrapper preflights actual task/dataset construction and archives incomplete output directories;
the previous directory-exists guard could incorrectly suppress a retry after
lm-eval wrote only `command.json` and then failed.

Production output acceptance is also explicit. The downstream wrapper runs
`scripts/analysis/audit_nanotron_downstream_eval.py` before accepting either a
new or pre-existing result. The audit requires the exact ten-task command, all
ten top-level task/group results, all 105 expanded leaf-task sample files,
full unique-document coverage from lm-eval's `n-samples` metadata, finite
primary metrics, an un-limited production run, and direct versus Qwen-chat
prompt rendering consistent with the evaluation branch. Five focused tests
and both final-smoke branches pass; the smoke has 106 JSONL rows because GSM8K
stores separate strict and flexible-extraction filter rows for the same
document, which the audit correctly counts once by `doc_id`.

The matched comparison is also fixed before results. Control evals
`3847792/3835928`, logic `3847804/3847806`, and NL `3834904/3834905` all write
to the unified corrected root and feed CPU-only strict aggregate `3847807`.
Canceled pending control evals `3834906/3834907` used the legacy default root. The
aggregate requires all six production audits and reports each primary task,
task stderr, deltas from control, and instruction-minus-direct deltas. Its
predeclared unweighted macros are all-primary, reasoning-core, general
multiple-choice, and targeted logic. Targeted logic contains LogiQA, MMLU
formal logic, BBH formal fallacies, and BBH logical deduction at three, five,
and seven objects. Representative correct/incorrect samples are indexed with
the exact primary filter, avoiding GSM8K strict-versus-flexible mixing. Because
there is one training run per condition, macro results do not estimate
training-seed variance.

## Production Configuration

The production job uses:

- sequence length: `4096`
- tensor parallelism: `4`
- data parallelism: `2`
- microbatch size per replica: `4`
- gradient accumulation: `16`
- optimizer steps: `8192`
- global chunks per optimizer step: `4 * 2 * 16 = 128`
- global tokens per optimizer step: `128 * 4096 = 524,288`
- total scheduled chunks: `8192 * 128 = 1,048,576`
- total scheduled tokens: `1,048,576 * 4096 = 4,294,967,296`

The p15 job configuration lists normal text first and corrected BranchProof-v2
logic or NL second, with normalized weights `[0.85, 0.15]`.

## Exact Realized Counts

Counts were recomputed with Nanotron's compiled
`helpers.build_blending_indices` implementation, the same helper called by
`BlendableDataset`:

| Prefix | Normal chunks | Proof chunks | Proof fraction | Proof tokens |
| ---: | ---: | ---: | ---: | ---: |
| First global batch (128 chunks) | 108 | 20 | 15.625000% | 81,920 |
| Checkpoint 4096 (524,288 chunks) | 445,644 | 78,644 | 15.000153% | 322,125,824 |
| Final step 8192 (1,048,576 chunks) | 891,289 | 157,287 | 15.000057% | 644,247,552 |

The final difference from an exact real-valued 15% allocation is 0.6 of one
4,096-token chunk. Logic and NL use the same weights, seed, and total schedule,
so their realized source counts are identical.

The exact weighted index is subsequently shuffled by
`Nanoset.build_nanoset_index` with production seed 42. Replaying that full
path gives random, not optimizer-stratified, batches. Across 8,192 updates, a
128-chunk global batch contains 6--35 proof chunks (mean `19.2001`, standard
deviation `4.0848`), close to the binomial standard deviation `4.0398`; no
global update is proof-empty. The two 64-chunk data-parallel replicas average
`9.6088` and `9.5913` proof chunks with standard deviation about `2.866` and
ranges 1--21 and 0--22. Four-chunk microbatches can contain 0--4 proof chunks.
This is well randomized and globally mixed, but it is not strict per-update
stratification.

Within the generated proof source, record indices are sequential but depths
are hash-randomized. In the first 100,000 records, all depths 1--25 have
`3,861--4,178` examples, adjacent records share a depth at rate `0.0393`, and
lag-one depth correlation is `-0.0046`. This rules out a hidden depth
curriculum or meaningful local depth clustering.

Pretokenization produced only one nonempty `*_unshuffled.ds` shard per source,
so file-level shuffling alone would be ineffective. This does not leave the
training stream sequential: `Nanoset.build_nanoset_index` applies the same
seeded random permutation to source and sample indices, producing a random
permutation of packed chunks before distributed sampling. All three runs also
use the same FineWeb chunks, while logic/NL use paired proof corpora.

## Corpus Capacity

Pretokenized metadata reports:

| Corpus | Available packed tokens | Maximum production use | Wraparound |
| --- | ---: | ---: | --- |
| Normal continuation | 4,804,719,208 | 4,294,967,296 in control; 3,650,719,744 in p15 | No |
| Corrected logic | 1,200,313,814 | 644,247,552 | No |
| Corrected NL | 1,200,312,018 | 644,247,552 | No |

Thus neither the control nor either p15 arm repeats its packed stream during
the planned run.

## Resume Semantics

`BlendableDataset` builds a deterministic full-run index from weights and
seed. Its dataloader resumes at `consumed_train_samples`, while checkpoint
metadata stores `consumed_tokens_per_dataset_folder`. Consumption accounting
uses the absolute optimizer-step interval in that same full index. The normal
control checkpoint at step 4096 independently records exactly 524,288 chunks
and 2,147,483,648 normal tokens.

Consequently, the pre-submitted recovery jobs continue from the next absolute
chunk and preserve both source proportions and per-source offsets. They do not
restart the blend or replay the first half.

### Learning-rate resume correction

Production used peak LR `1e-5`, linear warmup for 256 updates, cosine decay,
and floor `1e-6`. Losses and gradient norms remained finite, so the logs do
not support divergence or an obviously excessive peak LR. There was no LR
ablation, however, so the run cannot establish that `1e-5` is optimal.

Two schedule implementation issues were found after completion:

1. `lr_decay_steps` was explicitly set to all 8,192 updates even though decay
   begins after warmup. The run therefore ended near `1.70e-6`, not the
   configured `1e-6` floor. Future configs leave this field null so Nanotron
   derives `8192 - 256 = 7936` decay updates.
2. On resume, `lr_scheduler_builder` captured the optimizer's current
   checkpoint LR (`~5.94e-6`) while PyTorch `LambdaLR` retained the original
   base LR (`1e-5`). An isolated reproduction gives `6.2476e-6` after resume;
   normalizing by `param_group["initial_lr"]` gives the correct `5.9394e-6`.
   The local Nanotron checkout now uses the preserved base LR and includes a
   regression test.

Because control, logic, and NL all checkpointed and resumed at the same update
with the same optimizer/scheduler settings, these defects are matched. They
slightly change the absolute training trajectory but cannot selectively cause
the logic/NL differences in the downstream table.

Operational update 2026-07-11 15:30 CEST: normal-control recovery
`3828946_0` resolved the complete step-4096 checkpoint and logged
`start_iteration_step: 4096`, `consumed_samples: 524288`, and exactly
`2147483648` consumed normal tokens. It then failed before its first resumed
optimizer step because W&B's local service did not publish its port file on
node `a0831`; no checkpoint or sampler state changed. Replacement control
recovery `3835438_0` disables W&B and excludes `a0831`. Untouched logic/NL
recoveries were proactively replaced by W&B-disabled `3835442_3/3835443_8`
with the same after-any parents, run roots, corpus overrides, and step-4096
checkpoint interval. Upload jobs `3831119/3831123/3831113` now depend on those
replacement recoveries.

Operational update 2026-07-12 00:01 CEST: both corrected p15 runs reached
step 4096. `verify_training_checkpoint.py` accepted each complete 645-file
tree: 625 model files, four scheduler shards, eight RNG shards, no zero-byte
files, four equal `22,848,937,060`-byte optimizer shards, topology TP=4/DP=2,
and metadata offsets step `4096`, samples `524288`, tokens `2147483648`.
Per-dataset metadata exactly matches the schedule above: `1,825,357,824`
normal tokens and `322,125,824` corrected logic or NL tokens. Audit reports are
stored in `analysis/nanotron_checkpoint_audits/`. Parents
`3830927_3/3831111_8` were canceled only after acceptance so the released
recoveries `3835442_3/3835443_8` resume at 4096 instead of spending the final
approximately five allocation hours on work that could not be checkpointed
before timeout.

Capacity update 2026-07-12 00:46 CEST: the project tree measures `858G`, and
each accepted step-4096 checkpoint measures `199G`. The highest recent Slurm
epilogue usage was `1072.3G`; conservatively adding three simultaneous final
checkpoints projects about `1.67T`, leaving roughly `428G` under the documented
`2097.2G` hard quota. The run will cross the `1048.6G` soft warning threshold,
but the observed hard-space and file-count margins cover all three final
checkpoints plus HF staging. Step 4096 must remain until the corresponding
step-8192 tree is verified; the guarded upload path is responsible for later
cleanup.

Control resume update 2026-07-12 10:29 CEST: replacement recovery
`3835438_0` started on full A100-80 node `a0531`. Its generated config enables
optimizer and LR-scheduler loading from the accepted step-4096 run checkpoint.
The runtime restored `start_iteration_step=4096`, `consumed_samples=524288`,
and `consumed_tokens_total=2147483648`, then logged iteration `4101/8192` at
about `30.9K` tokens/s with finite loss `2.07`. This is direct evidence that
the control resumed without replay or a weight-only optimizer reset. Logic/NL
recoveries `3835442_3/3835443_8` remain account-GRES pending near 12:27 CEST
and require the same check when they start.

Follow-up 2026-07-12 10:39 CEST: the control remained healthy through
iteration `4141/8192`, with `2.17B` consumed tokens, `30.8K` tokens/s, finite
loss `1.98`, and no fatal/OOM/quota signature. Logic/NL recoveries still have
provisional 12:27 starts. Oversight successor `3841073` was advanced from
16:35 to 12:45 CEST so their first resumed iterations can be checked promptly.

NL resume update 2026-07-12 18:50 CEST: corrected recovery `3835443_8`
started on full A100-80 node `a0532` and passed the same resume gate. Its
config loads optimizer and LR-scheduler state from the accepted step-4096
checkpoint; runtime metadata restored step/sample/token offsets
`4096/524288/2147483648` and the exact `1825357824` normal plus `322125824`
NL-token split. It logged iteration `4101/8192` at `31K` tokens/s with finite
loss `1.71`, then advanced to `5421/8192` at `30.9K` tokens/s with finite loss
`1.74`. Control simultaneously reached `5871/8192`. Neither log has a
fatal/OOM/quota signature. Logic recovery `3835442_3` remains dependency-free
and blocked only by the account GPU ceiling; no scheduler edit is warranted.

Progress update 2026-07-12 21:12 CEST: control and NL remained healthy at
iterations `6371/8192` and `5921/8192`, respectively, each near `30.9K`
tokens/s with finite losses and no fatal/OOM/quota signature. Logic has a
provisional 07:00 CEST start. Only accepted step-4096 checkpoint trees exist;
no partial final tree or cleanup trigger is present.

Progress update 2026-07-13 00:55 CEST: control advanced to iteration
`7161/8192` with `3.75B` consumed tokens and finite loss `2.01`; corrected NL
advanced to `6711/8192` with `3.52B` consumed tokens and finite loss `1.63`.
Both remain near `30.9K` tokens/s with no fatal/OOM/quota signature. Latest
runtime ETAs were about `04:51/06:59`; only their accepted step-4096 trees
exist, so there is no final-verification or guarded-cleanup trigger yet. Logic
recovery `3835442_3` remains dependency-free and account-GRES pending with a
provisional 07:00 start. Project usage remains `871G`, and the only idle A100
nodes are incompatible A100-40 nodes. Successor oversight `3845763` is queued
for 06:46 CEST to inspect the final-checkpoint/control-upload and logic-resume
transition if the current estimates hold.

Final-control and logic-resume update 2026-07-13 06:53 CEST: control recovery
`3835438_0` completed `0:0` at 05:48 after reaching step `8192`. Independent
verification is persisted at
`analysis/nanotron_checkpoint_audits/control_step8192.json` and accepts the
complete tree: 645 files, no zero-byte files, TP=4/DP=2, 625 model files, four
LR-scheduler shards, eight RNG shards, four equal `22,848,937,060`-byte
optimizer shards, and exact metadata offsets step/sample/token
`8192/1048576/4294967296`. Control upload `3831119` is released and account-GRES
pending; local checkpoints remain intact. Logic recovery `3835442_3` started
at 05:48 and passed the same no-replay gate: optimizer and scheduler loading
are enabled, offsets restored at `4096/524288/2147483648`, the per-dataset
accounting restored exactly `1825357824` normal plus `322125824` logic tokens,
and resumed iteration `4101` had finite loss `1.71`. It advanced to
`4301/8192` at `30.9K` tokens/s with finite loss `1.76`. NL recovery
`3835443_8` remained healthy at `7971/8192` with finite loss `1.65` and about
one hour remaining. Vault usage is `1211.8G` against the `2097.2G` hard quota;
the three matched Nanotron roots occupy `795G`. No cleanup is authorized until
each corresponding fail-closed upload verification succeeds.

NL-final and logic-progress update 2026-07-13 10:10 CEST: corrected NL
recovery `3835443_8` completed `0:0` at 07:55 after saving step `8192`.
Independent verification is persisted at
`analysis/nanotron_checkpoint_audits/nl_exact_step8192.json` and accepts the
same complete 645-file state layout as control: no zero-byte files, TP=4/DP=2,
625 model files, four LR-scheduler shards, eight RNG shards, four equal
`22,848,937,060`-byte optimizer shards, exact offsets
`8192/1048576/4294967296`, and exact per-dataset accounting of `3650719744`
normal plus `644247552` NL tokens. Upload `3831113` is released and waits only
on account GRES; no cleanup has run. Logic recovery `3835442_3` reached
`4991/8192` at `30.9K` tokens/s with finite loss `1.76`; measured throughput
projects completion near 01:12 CEST on July 14, before its 05:48 allocation
limit. Vault usage is `1268G` against the `2097.2G` hard quota.

Conversion/downstream repair update 2026-07-13 10:36 CEST: first control/NL
uploads `3831119/3831113` failed before weight loading because the custom
Nanotron-to-HF converter was launched with plain Python and lacked
`WORLD_SIZE`. The wrapper now uses single-rank `torchrun`; replacements
`3847569/3847570` passed mapping, four-shard integrity, complete CUDA reload,
finite logits `[1,152064]`, and remote-manifest parity before guarded local
checkpoint cleanup. The two HF repos are recorded in their run roots. A second
cross-environment issue was caught before downstream eval: Transformers 5.12.1
saved Qwen's 13 special tokens as `extra_special_tokens`, while downstream
Transformers 4.57.3 requires `additional_special_tokens`. The converter now
normalizes this metadata and the final verifier explicitly runs in the
downstream environment. Both existing repos were repaired in place; fresh
4.57.3/5.12.1 loads preserve token IDs `151644..151656` and native-chat
rendering. NL direct eval `3834904` is released; replacement control direct
`3847792` follows a transient-download failure in `3835927`, and replacement
instruction parents `3847661/3847662` feed existing evals `3835928/3834905`.

Instruction/downstream preflight update 2026-07-13 10:46 CEST: exact dry runs
against both repaired remote checkpoints loaded UltraChat, retained all sampled
`8/8` train and `4/4` eval rows, rendered the native Qwen chat template, and
supervised only assistant tokens. The focused instruction-format,
downstream-artifact, and aggregate tests pass (`12 passed`). NL direct eval
`3834904_8` then loaded all four shards, initialized vLLM and its KV cache, and
began full-suite context construction without the former tokenizer exception.

Download recovery update 2026-07-13 11:06 CEST: control direct eval
`3835927_0` failed before weight loading because inherited `hf_transfer`
received a transient 403 for the first shard. The direct and instruction
wrappers now force standard resumable HTTP. A100-80 replacement `3847792` is
pending, and CPU-only aggregate `3847793` replaces dependency-unsatisfiable
`3836159`. Focused downstream, aggregate, and instruction tests pass
(`17 passed`).

Stored-payload recovery update 2026-07-13 11:12 CEST: dependency-held logic
upload/direct/instruction jobs `3831123/3834908/3831125/3834909` still
contained the pre-repair script bodies captured by Slurm. Verified
replacements are `3847802/3847804/3847805/3847806`; they include single-rank
conversion, downstream checkpoint verification, standard HTTP downloads,
automatic instruction resume, and production artifact audits. CPU-only strict
aggregate `3847807` has the corrected six dependencies. Old jobs were canceled
only after stored-script and dependency verification.

Instruction recovery update 2026-07-13 10:53 CEST: before any replacement
instruction parent started, all three exact output roots were verified absent
and the live wrapper was extended with `--resume-from-checkpoint auto`. Clean
launches therefore remain unchanged, while timeout/node-loss replacements can
resume the latest Trainer checkpoint. Resolver, wrapper, format, and exact
remote-checkpoint dry-run checks pass (`7 passed`).

HF storage cleanup update 2026-07-13 10:36 CEST: account inventory measured
`83.942G` of models and `21.896G` of datasets. Only the three superseded merged
OLMo SFT seed repos were deleted (`43.818G`); their retained LoRA repos and
public base reconstruct them. All datasets, unrelated model repos, and new
Qwen checkpoints remain. Verified model storage is `40.125G`, total storage is
about `62.021G`, and the projected total after logic upload is `77.264G`.
Details are in `analysis/hf_storage_cleanup_2026-07-13.json`.

HF storage cleanup update 2026-07-13 11:57 CEST: nine older
`autoformalization-*` iterations were downloaded at exact commits, verified
file-for-file by size and SHA-256, and archived under
`$HPCVAULT/hf_model_archives/2026-07-13_autoformalization_superseded/` before
their Hub repositories were deleted. The latest adapter for every task remains
remote. Retained LFS is now `63.487G`; the pending logic checkpoint projects
to `78.730G`. The matched p15 chain is safe, but future full checkpoints still
require guarded upload/evaluate/audit/delete rotation.

HF storage reconciliation update 2026-07-13 12:20 CEST: authenticated
per-repository `usedStorage`, after the control instruction adapter upload,
totals `63.610G` (`35.257G` models and `28.352G` datasets). The pending logic
checkpoint projects to `78.852G`; one further full checkpoint fits at
`94.095G`, but two would reach `109.338G`. The retained latest
autoformalization artifacts total only `4.119G` and have no replacement, so
they were preserved. The broader grid must use guarded checkpoint rotation.

NL instruction completion and storage update 2026-07-13 12:55 CEST:
replacement instruction SFT `3847662_8` completed all `10000` steps in
`01:04:38`, wrote a nonzero local final, and uploaded adapter commit
`cddf739f4b4332e1d9f3d71b825e52c836476679` with the intended repaired NL p15
base and Transformers-4-compatible special-token metadata. It released
post-instruction eval `3834905_8`, which is account-GRES pending. Control
post-instruction eval `3835928_0` started at 12:48 CEST on A100-80GB and
initialized the merged 8192-context checkpoint. Authenticated Hub
`usedStorage` is now `63.782G` (`35.430G` models and `28.352G` datasets),
projecting to `79.025G` after logic, `94.268G` after logic plus one further
full checkpoint, and `109.511G` after logic plus two. No repo was deleted.

Final storage reconciliation 2026-07-14 19:47 CEST: after the corrected logic
checkpoint and instruction adapter completed uploading, authenticated
per-repository `usedStorage` is `79.281G` (`50.846G` models and `28.435G`
datasets across 66 repositories), leaving `20.719G` against the nominal quota.
No repository was deleted. Preserve all three p15 checkpoints and their current
adapters through multi-hop jobs `3855271/3855272/3855273`. The broader grid is
now rejected by the null/mixed, sample-unclean p15 gate; guarded rotation is
still required if that decision is revisited. The inventory is recorded in
`analysis/hf_storage_cleanup_2026-07-13.json`.

Downstream scoring update 2026-07-13 12:14 CEST: stock
`hendrycks_math500/exact_match,none` rejected correct answer prefixes whenever
the continuation included explanation. Whole-response symbolic matching also
proved unsafe because later prompt repetition and wrong structured answers can
contain a gold scalar. The production audit now generates a deterministic
`answer_prefix_math_verify,none` sidecar, preserves stock exact as a diagnostic,
and requires complete row/hash coverage with no lost stock positives. Focused
tests pass (`18 passed`). NL direct scores `0.160` post-hoc versus `0.028`
stock exact. Pending stored Slurm jobs call the live audit code, so they inherit
the scorer without cancellation or GPU reruns. See
`docs/nanotron_math500_scoring_audit_2026-07-13.md`.

NL-direct qualitative update 2026-07-13 12:31 CEST: the accepted production
bundle contains coherent correct GSM8K, BBH, and MMLU-Pro reasoning, while
incorrect rows expose omitted constraints, false implication reversals, and
repetition to the generation cap. BBH/MMLU-Pro invalid-extraction rates are
`9.1%/20.5%`; a literal generated next-document assistant preamble occurs in
`22.9%/3.7%` of rows and in zero corresponding prompts. The matched aggregate
now emits the same condition-blind diagnostics for every control/logic/NL
direct and post-instruction bundle. This is not yet evidence for a modality
effect; comparison remains gated on the other five accepted runs. See
`docs/nanotron_nl_direct_generation_audit_2026-07-13.md`.

Logic completion update 2026-07-14 01:16 CEST: no-replay recovery
`3835442_3` completed step 8192 in `19:26:16`. Terminal logged loss remained
finite (`1.80`) at about `30.8K` tokens/s. Independent checkpoint verification
accepted 645 files: 625 model, four optimizer, four scheduler, and eight RNG
files, with no zero-byte files and exact metadata offsets
`8192/1048576/4294967296`. Dataset-token offsets are `3650719744` normal plus
`644247552` formal proof, exactly `4294967296` total and `15.000057%` proof.
The persisted audit is
`analysis/nanotron_checkpoint_audits/qwen25_7b_midtrain_logic_p15_bp_unique_v2_4p3b_step8192_20260714.json`.
Repaired-payload upload `3847802_3` released and is capacity-pending. It must
pass conversion, dual-field/Transformers-4 RoPE resolution, CUDA finite-logit,
remote-parity, and guarded-cleanup gates before downstream jobs release.

Corrected direct multi-hop smoke update 2026-07-14 01:36 CEST:
`3850353_0` completed `0:0` in `00:04:10`. The generated audit accepted the
control checkpoint's resolved RoPE `1000000`, exact 32,768-token model window,
all six HotpotQA/2Wiki/MuSiQue stock/tagged cells, six sample files, and 12
rows. All raw generations were inspected. Tagged extraction was nonempty in
`6/6` and stopped cleanly before a next-question continuation. Stock output
continued into a generated question or assistant preamble in `4/6`; both
Hotpot rows started with the gold answer but the suffix reduced their QA F1 to
`0.273/0.300`. Tagged exact match was `1/6`; the sampled 2Wiki and MuSiQue rows
were wrong under both protocols. This is a prompt/extraction smoke at two rows
per dataset, not a transfer estimate. Dependency-held instruction smoke
`3850354_0` must pass the same raw review before any full six-condition grid is
submitted.

## 2026-07-14 Corrected Downstream Recovery and Sample Audit

Logic repaired-payload upload `3847802_3` passed the single-rank conversion,
dual-field consumer RoPE resolution (`1000000`), CUDA finite-logit, remote file
parity, and guarded cleanup gates. The local Nanotron checkpoint tree is gone;
the run config, repository/verifier reports, and remote checkpoint remain.
Logic direct reviewer eval `3847804_3` completed. Native-chat instruction SFT
`3847805_3` also completed all 10,000 steps with terminal train/eval loss
`0.942806/0.936798`, but its Slurm task failed only when a transient Hugging
Face Xet 401 interrupted the final adapter upload. The complete local adapter
was uploaded without retraining and verified at commit
`3d1e4a751150fffbb26e23e6f759c402bf203b4d` with all expected config,
tokenizer, and adapter files. Stale dependency-held eval `3847806` and mixed
aggregate `3850389` were canceled; exact replacement logic instruction eval
`3854824_[3%1]` is pending, and strict aggregate `3854847` depends on it.

The schema-v4 MATH-500 sidecar had one delimiter bug: an escaped currency
symbol such as `\$36` was mistaken for an opening math delimiter. The parser
now recognizes only unescaped dollar delimiters. Forced rescore plus production
audit accepted all five complete corrected bundles, with no stock-positive
row lost. Sidecar correct counts are control direct `98/500`, control
instruction `53/500`, logic direct `104/500`, NL direct `105/500`, and NL
instruction `61/500`. Stock exact remains only a format diagnostic.

Condition-blind raw diagnostics and manual correct/incorrect review found:

| Condition | BBH invalid / marker | MMLU-Pro invalid / marker |
| --- | --- | --- |
| Control direct | `4.377% / 35.601%` | `4.555% / 12.068%` |
| Logic direct | `4.239% / 59.960%` | `3.981% / 45.479%` |
| NL direct | `4.393% / 58.010%` | `4.737% / 49.510%` |
| Control instruction | `3.655% / 0%` | `1.222% / 0%` |
| NL instruction | `3.701% / 0%` | `1.263% / 0%` |

Prompt marker incidence is zero in every cell. Direct logic/NL frequently
continue into proof or new-document material. Instruction tuning removes the
literal markers but often produces extremely long repetitive continuations or
multiple candidate answers. These rates supersede the older `9.1/20.5%`
invalid and `22.9/3.7%` marker diagnostics, which came from the RoPE-invalid NL
bundle. No modality-transfer claim is accepted before logic instruction eval
and the matched aggregate.

Corrected instruction multi-hop smoke `3850354_0` passed the RoPE/32,768-window
structural gate, and all 12 generations were inspected. Tagged Hotpot was
`2/2`; tagged 2Wiki and MuSiQue were `0/2`. MuSiQue failed the sample-clean
gate through missing closing tags, context copying, and repetition to the
512-token cap; stock responses also leaked or repeated prompt-like text.
Therefore no full six-condition multi-hop grid was submitted. Its next trigger
is both a positive/sample-clean p15 aggregate and a revised tagged instruction
protocol that passes another raw smoke.

Prompt-fixed smoke update 2026-07-14 13:13 CEST: direct smoke `3855269_0`
completed and passed. Instruction smoke `3855270_0` completed all 12 retained
generations, resolved RoPE `1000000`, and used the audited 32,768-token window,
but exited `1:0` because the structural audit required the stock prompt to
start at byte zero even when lm-eval had applied Qwen's chat template. The
retained instruction prompts actually contain exactly one system/user/assistant
wrapper, one passage header, no duplicated `Question:` prefix, and the intended
32/64-token caps. The audit now extracts the single Qwen user turn before
checking task structure, with a regression for that mode. Focused tests and the
full repository suite pass (`223 passed, 3 skipped`). The existing smoke
re-audited accepted; CPU-only gate `3856131` completed `0:0`, and full
instruction array `3855272` now depends on that gate. Direct full array
`3855271` and instruction full array `3855272` are both account-GRES pending.

All smoke generations were inspected. The prompt and cap corrections contain
the earlier runaway behavior, but model quality remains visibly weak: some
tagged 2Wiki/MuSiQue outputs omit a valid tag, and instruction responses often
add explanations or repeat short phrases within the cap. These are generation
diagnostics to preserve in the full comparison, not a reason to treat the old
wrong-RoPE/truncated/nested-prompt bundles as evidence.

Replacement logic instruction reviewer eval `3854824_3` is running on a
verified A100-80GB consumer with RoPE `1000000`. At about 37 minutes of
generation it had processed `5,793/20,362` requests without a fatal signature;
aggregate `3854847` remains dependency-held. The broader mixture grid remains
blocked on its accepted six-bundle aggregate and raw sample comparison.

## 2026-07-14 Final Corrected p15 Comparison

Replacement logic instruction eval `3854824_3` completed `0:0` in `02:20:26`,
and strict CPU aggregate `3854847` completed `0:0`. Its accepted manifest
contains all six corrected control/logic/NL direct/instruction bundles, exact
checkpoint step 8192, all production task/sample coverage, schema-v4 MATH
sidecars, 24 generation-diagnostic rows, and 57 indexed qualitative samples.
Artifacts are under
`analysis/nanotron_branchproof_unique_v2_p15_20260711/`.

The matched macro result is null/mixed rather than positive:

| Branch | Condition | all-primary | reasoning | general MC | targeted logic |
| --- | --- | ---: | ---: | ---: | ---: |
| direct | control | `0.6052` | `0.4946` | `0.7158` | `0.5316` |
| direct | logic delta | `+0.0033` | `+0.0071` | `-0.0004` | `-0.0116` |
| direct | NL delta | `-0.0011` | `-0.0012` | `-0.0011` | `-0.0069` |
| instruction | control | `0.4992` | `0.3051` | `0.6932` | `0.1631` |
| instruction | logic delta | `+0.0018` | `+0.0038` | `-0.0001` | `-0.0005` |
| instruction | NL delta | `+0.0027` | `+0.0096` | `-0.0041` | `-0.0013` |

These are single training runs per condition and do not estimate seed
variance. Task deltas are mixed: logic-direct GSM8K and MATH-500 improve by
about `+0.030` and `+0.012`, but targeted logic tasks are mostly flat or
negative within their reported task stderr. NL-instruction GSM8K improves by
about `+0.041`, without a corresponding broad or targeted-logic gain.

Every condition/branch had correct and incorrect generations inspected.
Direct logic/NL BBH and MMLU-Pro next-document marker rates are
`60.0/45.5%` and `58.0/49.5%`, compared with control `35.6/12.1%`; prompt
marker incidence remains zero. Instruction tuning removes the literal marker,
but responses become long and repetitive. In BBH, correct leading choices are
often followed by extra phrases, multilingual text, or role-token junk. The
installed `get-answer` path therefore gives every instruction condition zero
on the targeted BBH leaves. Those cells and the large instruction-minus-direct
macro drops are an extraction/generation floor, not modality-transfer
evidence. A future evidentiary use would require an independently audited
answer-prefix sidecar or explicit exclusion of instruction BBH.

The broader Nanotron mixture grid is not triggered: the corrected p15 result
is neither positive nor sample-clean. Prompt-fixed multi-hop arrays
`3855271/3855272` remain submitted as bounded evaluation of the already-trained
three checkpoints; they are not an expansion of the training grid.

## 2026-07-15 Corrected Multi-Hop Completion

Prompt-fixed direct `3855271_[0-2]`, instruction `3855272_[0-2]`, and strict
aggregate `3855273` completed `0:0`. Each bundle retained all six tasks and
1,200 samples, resolved the consumer config with `rope_theta=1000000`, used a
32,768-token model window, and passed the prompt, 32/64-token cap, and full-run
audit. Accepted artifacts are under
`analysis/nanotron_branchproof_unique_v2_multihop_promptfix_20260714/`.

The stock direct QA-F1 macro is `0.189/0.250/0.238` for control/logic/NL.
Manual correct/partial/incorrect review and a persisted answer-head sensitivity
rescore show that this is not clean reasoning transfer: after removing only
obvious generated continuation beyond the initial answer span, the same macro
is `0.349/0.361/0.367`, reducing the logic and NL deltas to `+0.012/+0.018`.
Direct tagged generations expose stronger response-manifold interference:
logic opens `<formal>` in `98.5--99.0%` of rows and NL opens `<think>` in
`97.0--99.0%`, normally consuming the 64-token diagnostic before a usable
answer. Instruction SFT removes those learned-substrate openings, but the
stock branch remains cap-limited and gives control/logic/NL QA-F1
`0.097/0.100/0.085`.

This closes the bounded multi-hop evaluation as a response-control and format
transfer diagnostic. It does not overturn the null/mixed corrected p15
downstream result and does not satisfy the trigger for broader mixture
training.

## Failure Diagnosis

The completed evidence does not support insufficient source mixing as the
cause of the null/mixed result. The actual shuffled order matches ordinary
random mixing: every global update contains proof chunks, the observed batch
variance is near binomial expectation, exact exposure is 644M tokens, and both
modalities share the same seeded source schedule. Strict 19/20 per-update
stratification would reduce gradient-composition variance and is a reasonable
small ablation, but the present distribution is not anomalous. The LR resume
bug and overlong decay are real but matched across conditions.

The strongest observed failure mode is objective/response-manifold mismatch.
Continuation training applies loss to the entire packed document: question,
premises, trace wrapper, proof, answer, and EOS. It does not isolate reasoning
steps or supervise a downstream instruction-response contract. The injected
models demonstrably learn the special response surfaces: direct logic starts
`<formal>` on `98.5--99.0%` of tagged multi-hop rows and direct NL starts
`<think>` on `97.0--99.0%`, often exhausting the answer budget. Direct BBH and
MMLU-Pro also show substantially more next-document continuation than the
control. Thus the intervention changes generation behavior strongly, but much
of that change is surface continuation rather than transferable task
reasoning.

Before any new large wave, the minimum defensible pilot is: use the corrected
resume scheduler and a post-warmup decay span; log held-out Dolmino and
proof-source losses separately; compare random mixing with exact global-batch
stratification; and compare full-document continuation against a proof-focused
objective that masks copied question/premise tokens. Run this as a small
LR/objective pilot with at least two checkpoints before committing another
multi-billion-token grid.

The completed continual-pretraining background and intervention formats are
not aligned. FineWeb
records are raw document text followed by Qwen's EOS token; they have no user,
assistant, question, reasoning, or answer fields. Corrected proof records use
`<question>`, modality-specific `<formal>` or `<think>`, nested proof-section
tags, and `<answer>`. In a future Dolmino pilot, paired formal and matched NL
records should use the same modality-neutral `problem`, `Solution:`, and
`Final answer:` document envelope specified above. The modality should differ
only in the solution contents. Native Dolmino source records should remain
unchanged rather than being converted into artificial QA records.

Downstream readout also requires a cleaner test. The completed generic
UltraChat branch used the same assistant-only LoRA for every checkpoint, but
physical/effective batch size one, rank 16, 10,000 steps, and no examples that
teach a shared reasoning response contract. Its repetition and extraction
failures make it a weak alignment probe. A future pilot should retain direct
evaluation and add two identical post-midtraining readouts for every condition:
(1) a small answer-only instruction/format calibration with no formal traces,
and (2) a high-quality modality-neutral reasoning SFT with no BranchProof or
evaluation-task overlap. An adaptation curve over small SFT budgets can test
whether formal midtraining lowers the amount of downstream supervision needed;
using formal traces only in the formal checkpoint's readout would confound the
midtraining intervention.

The upload boundary is fail-closed before local checkpoint deletion. The
Nanotron-to-HF converter rejects any HF parameter absent from its explicit
mapping. After synchronous upload, `verify_qwen2_hf_checkpoint.py` checks the
local safetensors index and every referenced shard, reloads the complete 7B
checkpoint on CUDA and requires finite logits of the configured vocabulary
size, and compares the remote HF file list with the staged local manifest.
Only a successful verifier allows guarded staging/checkpoint cleanup; its JSON
report remains in the run root. Pending upload jobs invoke the current scripts
at runtime and therefore need no resubmission.

## Scientific Interpretation

This is packed continual pretraining, not whole-example SFT. A source proof
can cross a 4,096-token boundary, and a packed chunk can contain the end of one
record and the start of another. The experimental intervention is therefore
best described as adding 15% formal-proof or matched NL-proof *tokens* to a
normal continuation stream. The logic/NL comparison remains exposure-matched,
but claims should not imply that every optimizer example contains one intact
proof problem.

## Evidence

- Official Dolma 3 dataset taxonomy:
  `https://github.com/allenai/dolma3`
- Official 10B Dolmino micro-anneal release:
  `https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025`
- Official 100B OLMo-3-7B second-stage mix and source composition:
  `https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025`
- Production wrapper:
  `scripts/slurm/jobs/nanotron_qwen25_midtrain_grid_2026-06-24.slurm`
- Nanotron dataset construction:
  `../nanotron/src/nanotron/data/tokenized_bytes.py`
- Exact blend helper and resume accounting:
  `../nanotron/src/nanotron/data/nemo_dataset/blendable_dataset.py` and
  `../nanotron/src/nanotron/data/nemo_dataset/helpers.cpp`
- Verified control metadata:
  `$HPCVAULT/synthetic-RLVL/nanotron_midtrain/qwen25_7b_midtrain_control_p0_4p3b/checkpoints/4096/checkpoint_metadata.json`
- Persisted step-4096 checkpoint audits:
  `analysis/nanotron_checkpoint_audits/`
- Corrected corpus audit:
  `analysis/branchproof_unique_v2_corpus_audit_2026-07-10.json`

## 2026-07-16 Dolmino Shared-LR Gate

Matched TP4/DP1 four-A100 rows `3859711_[0-2]` completed 256 steps and exactly
134,217,728 scheduled tokens each. All logged losses, gradient norms, and
throughput values are finite. Over identical post-warmup steps 33--256, mean
loss is `0.956750` at `1e-5`, `0.960402` at `6e-6`, and `0.971304` at `3e-6`.
The `1e-5` loss is lower than `6e-6` on 125 steps with 73 ties (mean paired
delta `1e-5 - 6e-6 = -0.003652`) and lower than `3e-6` on 161 steps with 40
ties (mean paired delta `-0.014554`). Throughput is matched at roughly
15.3--15.4K tokens/s. No divergence signature appears; nominate `1e-5` as the
shared LR for the short formal/NL p5 confirmations.

Artifact caveat: Nanotron did not materialize the configured `benchmark.csv`
for any row even though each `complete.json` names that path. The complete
per-step Slurm logs are therefore the authoritative LR-gate record and must be
retained. No broader staged run is triggered by training loss alone.

## 2026-07-20 Dolmino p5 confirmation launch

The capacity and quota gates were cleared at 08:52 CEST. Submitted
`3872664_[0-1%2]` with the validated TP4/DP1 four-A100 topology, global batch
128, 256 steps, 32 warmup steps, and shared peak LR `1e-5`. Row 0 mixes
Dolmino/formal at `0.95/0.05`; row 1 mixes Dolmino/matched-NL at `0.95/0.05`.
Both resume the same immutable pretrained Qwen2.5-7B Nanotron checkpoint and
use the neutral-tag proof Nanosets built by the accepted prerequisite. Config
generation was checked for both rows, including paths, weights, topology,
step count, and LR. Full staged production remains blocked until both rows
complete with finite optimization traces and sample-clean mixture behavior.

## 2026-07-21 Dolmino p5 rendezvous recovery

Both four-GPU confirmation rows were scheduled concurrently on the two halves
of A100-80GB node `a0536`. Logic row `3872664_0` initialized first and used
torchrun's default port 29500. NL row `3872664_1` then failed before model or
data initialization with `EADDRINUSE` on the same port. It consumed only
`00:01:19` and wrote no training result, so no scientific state needs replay.

The shared LR wrapper now passes a deterministic per-allocation port,
`20000 + SLURM_JOB_ID % 40000`, to torchrun while preserving an explicit
`MASTER_PORT` override. Shell syntax checks pass. Exact NL-only replacement
`3875623_1` preserves the original model, seed, data, 95/5 weights, LR,
warmup, 256 steps, global batch 128, and TP4/DP1 topology. A malformed
single-task submission `3875622` was canceled before start. Logic row 0
continued unaffected: at step 189 it sustained about 15.5K tokens/s with
finite loss and gradients; Nanotron's realized blend was
`0.949982/0.0500183`. Production remains gated on both completed confirmation
rows and the planned mixture/sample audit.

## 2026-07-21 5B production launch

At user request, canceled the never-started repaired NL confirmation
`3875623_1` and advanced to the three-condition 5B gate. Control and formal
short gates pass; the NL corpus layout and decoded neutral-format samples pass,
and its only observed runtime failure was the pre-optimizer port collision.

The existing normal Nanoset has 4,810,706,180 packed tokens, which is too short
for a unique 5,000,134,656-token control. Build `3875824` therefore creates a
new deterministic 5.1B-token normal Nanoset and removes its raw JSONL only
after the packed-token threshold passes. Production chains are control
`3875825/26/27`, formal-5% `3875828/29/30`, and NL-5% `3875831/32/33`.
Each uses eight A100-80GB GPUs, TP4/DP2, global batch 128, step target 9,537,
peak LR `1e-5`, 256-step warmup, cosine decay to `1e-6` over 37,891 decay
steps (the preregistered 20B horizon), and restart states every 500 steps.
Three 24-hour allocations are queued per condition; complete 5B states skip
remaining stages.

This launch shares the normal Nanoset and seed but uses Nanotron's deterministic
weighted sampler for `1.0` control and `0.95/0.05` interventions. It does not
implement the stronger precomputed schedule with identical normal chunk IDs at
every cross-condition slot. Results are distribution-matched, not exactly
sample-paired, and must be described that way.

## 2026-07-26 First 5B state and guarded rotation

Control stage `3875825_0` started on eight A100-80GB GPUs at 08:09 CEST. At
13:22 it had reached step `1081/9537` at about `30.7K` tokens/s with finite
loss and gradient norm. The step-1000 checkpoint independently passes the
complete restart gate: 645 files, no zero-byte file, TP4/DP2, 625 model files,
four equal `22,848,937,060`-byte optimizer shards, four scheduler shards,
eight RNG shards, and exact step/sample/token offsets
`1000/128000/524288000`. Its sole dataset offset is exactly 524,288,000
normal tokens. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_control_step1000_20260726.json`.

The current Nanotron writer retains every 500-step state. Once step 1000
passed the gate, the exact superseded step-500 tree was removed, reclaiming
`106,628,386,940` bytes; step 1000 remains intact as the sole numeric restart
state. The shared live wrapper now has an opt-in restart-time guard that
resolves the newest complete run checkpoint first, then removes only strictly
older states that independently pass `checkpoint_is_complete`; the Dolmino
5B wrapper enables it. Both scripts pass `bash -n`. During an active stage,
oversight must continue rotating only strictly older complete states after a
newer state passes the same gate.

At 19:23 CEST the control had advanced beyond step 2341 and the writer had
retained steps 1000, 1500, and 2000, temporarily raising Vault to
`1064G/1000G`. Step 2000 passed the same complete restart gate with exact
step/sample/token offsets `2000/256000/1048576000`, all charged to the normal
Nanoset; audit:
`analysis/nanotron_checkpoint_audits/dolmino_control_step2000_20260726.json`.
Only then were superseded complete steps 1000 and 1500 removed, reclaiming
`213,256,773,890` bytes. Step 2000 remains the sole numeric restart state and
Vault returned to `667G/1000G`.

At 01:31 CEST July 27 the control was healthy beyond step 3631 at about
`30.7K` tokens/s. Step 3500 passed the same 645-file, zero-byte, TP4/DP2,
model/optimizer/scheduler/RNG gate with exact step/sample/token offsets
`3500/448000/1835008000`, all charged to the normal Nanoset. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_control_step3500_20260727.json`.
Only after acceptance were superseded complete steps 2000, 2500, and 3000
removed, reclaiming `319,885,160,841` bytes. Step 3500 remains the sole
numeric restart state; user-wide Vault returned to `547G/1000G` with
`180k/200k` files.

At 08:17 CEST July 27, steps 4500 and 5000 had independently passed the same
complete restart gate. Step 5000 contains 645 files with no empty file, 625
model files, four equal `22,848,937,060`-byte optimizer shards, four scheduler
shards, eight RNG shards, and exact step/sample/token offsets
`5000/640000/2621440000`, all charged to the normal Nanoset. Audits:
`analysis/nanotron_checkpoint_audits/dolmino_control_step4500_20260727.json`
and `analysis/nanotron_checkpoint_audits/dolmino_control_step5000_20260727.json`.
Only after acceptance were superseded steps 3500/4000/4500 removed, reclaiming
`319,885,160,841` bytes. Step 5000 is the sole restart state and Vault is
`553G/1000G`, `180k/200k` files.

## 2026-07-27 First formal 5B restart state

Formal first stage `3875828_1` reached iteration 1071 with finite loss and
gradient diagnostics at about 30.7K tokens/s. Step 1000 independently passes
the complete 645-file, zero-byte, TP4/DP2 model/optimizer/scheduler/RNG gate,
Qwen2.5 RoPE `1000000`, and exact offsets
`1000/128000/524288000`. Its checkpoint metadata records exactly
`498073600` normal Dolmino tokens and `26214400` formal tokens, an exact 95:5
realized split. The accepted audit is
`analysis/nanotron_checkpoint_audits/dolmino_logic_step1000_20260727.json`.

The active writer retained both steps 500 and 1000 and temporarily raised
Vault usage to `950G/1000G`. After the newer state passed every restart gate,
only superseded step 500 was removed, reclaiming `106,628,387,143` bytes.
Step 1000 remains the sole numeric formal restart state. Continue the same
latest-complete-state rotation; never remove the newest accepted checkpoint.

Control first stage `3875825_0` then reached iteration 5021 and ended in the
expected `TIMEOUT` after `1-00:00:10`; no OOM, quota, or unexpected fatal
signature was present. Continuation `3875826_0` released account-GRES pending.
The freed eight-A100-80GB allocation immediately started formal stage
`3875828_1`. Its live config preserves RoPE base `1000000`, sequence length
4096, TP4/DP2, global batch 128, target 9537, peak LR `1e-5`, 256-step warmup,
and the preregistered decay horizon. Nanotron realized normal/formal weights
as `0.95/0.0500002`; iteration 1 reported finite loss and gradient norm.

## 2026-07-27 Formal step-2000 restart state

Formal first stage `3875828_1` reached iteration 2331 with finite loss and
gradient diagnostics at about 30.7K tokens/s. Step 2000 independently passes
the complete 645-file, zero-byte, TP4/DP2 model/optimizer/scheduler/RNG gate,
Qwen2.5 RoPE `1000000`, and exact offsets
`2000/256000/1048576000`. Checkpoint metadata records exactly `996147200`
normal Dolmino tokens and `52428800` formal tokens, preserving the exact 95:5
realized split. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step2000_20260727.json`.

The writer had retained steps 1000, 1500, and 2000 and raised user Vault use
to `1149G/1000G`. Only after step 2000 passed every restart gate, the exact
superseded step-1000 and step-1500 trees were removed, reclaiming
`213,256,774,296` bytes. Step 2000 remains the sole numeric formal restart
state and Vault returned to `751G/1000G`, `181k/200k` files.

## 2026-07-28 Formal step-3500 restart state

Formal first stage `3875828_1` reached iteration 3591 with finite loss and
gradient diagnostics at about 30.7K tokens/s. Step 3500 independently passes
the complete 645-file, zero-byte, TP4/DP2
model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE `1000000`, and exact offsets
`3500/448000/1835008000`. Checkpoint metadata records exactly `1743257600`
normal Dolmino tokens and `91750400` formal tokens, preserving the exact 95:5
realized split. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step3500_20260728.json`.

The writer had retained steps 2000, 2500, 3000, and 3500 and raised user Vault
use to `1347G/1000G`. Only after step 3500 passed every restart gate, the exact
superseded step-2000/2500/3000 trees were removed, reclaiming
`319,885,161,449` bytes. Step 3500 remains the sole numeric formal restart
state and Vault returned to `751G/1000G`, `181k/200k` files.

## 2026-07-28 Formal step-4500 restart state

Formal first stage `3875828_1` reached iteration 4851 with finite loss and
gradient diagnostics at about 30.8K tokens/s. Step 4500 independently passes
the complete 645-file, zero-byte, TP4/DP2
model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE `1000000`, and exact offsets
`4500/576000/2359296000`. Checkpoint metadata records exactly `2241331200`
normal Dolmino tokens and `117964800` formal tokens, preserving the exact 95:5
realized split. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step4500_20260728.json`.

The writer retained steps 3500, 4000, and 4500 and raised user Vault use to
`1149G/1000G`. Only after step 4500 passed every restart gate, the exact
superseded step-3500/4000 trees were removed, reclaiming
`213,256,774,301` bytes. Step 4500 remains the sole numeric formal restart
state and Vault returned to `751G/1000G`, `181k/200k` files.

## 2026-07-28 Formal boundary and NL startup retry

Formal first stage `3875828_1` logged iteration 5021 and ended at the expected
24-hour boundary. Step 5000 independently passes the complete 645-file,
zero-byte, TP4/DP2 model/optimizer/scheduler/RNG gate, Qwen2.5 RoPE
`1000000`, and exact offsets `5000/640000/2621440000`. Its metadata records
exactly `2490368000` normal Dolmino tokens and `131072000` formal tokens,
preserving the exact 95:5 split. Audit:
`analysis/nanotron_checkpoint_audits/dolmino_logic_step5000_20260728.json`.
Only after acceptance was superseded step 4500 removed, reclaiming
about `106.63 GB`; step 5000 is the sole formal restart state.

NL first stage `3875831_2` reached model and dataloader initialization but
failed before optimizer step 1 because the local W&B service did not create
its port file within 30 seconds. It wrote no training checkpoint. The
dependency-released continuation `3875832_2` is therefore the exact from-base
retry with no scientific state or token exposure to replay.

## 2026-07-30 Step-5000 intermediate readout

The matched control/formal A40 readout completed all inference. A missing
terminal-percent normalization in the MATH-500 sidecar initially rejected
formal sample 67 (`$10\%$` for gold `10`); the focused scorer fix, tests,
forced CPU rescore, and production re-audit now accept both retained bundles
without rerunning inference. Limit-100 reviewer-suite formal-minus-control
macro is `+0.0408`, but stock multi-hop QA-F1 falls `0.3171 -> 0.1093`
because formal generations commonly continue into another QA record; tagged
QA-F1 changes `0.3104 -> 0.3348`. This is provisional intermediate evidence
only. Full artifact and raw-review notes:
`analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.

## 2026-07-31 Format and post-SFT decision

Exact Nanoset index inspection strengthens the full-document diagnosis:
formal/NL p95 document lengths are `7321/7392` tokens and `44.0%/47.0%`
of records exceed the active 4,096-token window. The active
`TokenizedBytesFileDataset` slices the concatenated token stream into fixed
windows rather than constructing document-preserving instances. The
implemented neutral envelope also contains more repeated context and
scaffolding than the compact format selected on July 10. EOS insertion and
the configured Qwen EOS ID agree.

Do not change the active matched series before the NL step-5000 discriminator.
After terminal checkpoints, add identical modern full-parameter,
assistant-only instruction SFT for control/formal/NL and keep direct evaluation
as a separate branch. Gate a corrected 5B formal/NL rerun on a roughly
0.5B-token compact/document-preserving objective pilot. The prior
effective-batch-one UltraChat LoRA remains a failed alignment diagnostic.
Exact pilot and post-SFT design:
`analysis/dolmino_format_and_post_sft_plan_20260731.md`.

## 2026-07-31 Step-5000 generation diagnosis

Raw review of all 1,200 retained stock/tagged multi-hop generations and
representative standard-suite outputs separates competence from response
control. The formal checkpoint emits a new `Question:` record after its first
answer in 91--97% of stock 2WikiMultiHopQA, HotpotQA, and MuSiQue samples,
versus 28--49% for control. This also occurs in 87--100% of formal samples
whose first answer is exactly correct. Explicit `<answer>...</answer>` prompts
nearly eliminate the continuation and recover tagged macro QA-F1
`0.3104 -> 0.3348`; first-answer-only rescoring gives `0.3893 -> 0.3799`, so
there is no broad hidden multi-hop gain. HotpotQA improves, while 2Wiki and
MuSiQue do not.

The failure is not literal proof-format leakage: no inspected formal output
emits the training envelope or formal tags. The likely mechanism is indirect
response-boundary drift. The intervention examples are long full documents
with repeated problem premises, `Context/Derivation/Conclusion` sections, and
EOS after `Final answer:`; stock LongBench prompts end in bare `Answer:` and
provide no stop string. Proof records average about 3.8K tokens versus 456 for
Dolmino records, reducing document-boundary density in the mixed stream.
MMLU-Pro also shows a genuine long-tail control regression: invalid extracted
choices rise from 2.50% to 4.86% and affected generations become repetitive
runaways. At the same time, likelihood-scored MMLU and several inspected
GSM8K, MATH-500, BBH, and MMLU-Pro examples show real correctness gains.

Formal-specific causality remains unresolved until the matched NL checkpoint
reaches step 5000. Run the identical limited readout then; at terminal
checkpoints evaluate control/formal/NL under stock, tagged, and a common
answer-only calibration. Keep the current full-document format for the matched
run, but compare a genuinely minimal envelope and proof-focused loss masking
in any future format pilot. Full metrics, examples, and artifact paths are in
`analysis/nanotron_dolmino_step5000_intermediate_20260730.md`.
