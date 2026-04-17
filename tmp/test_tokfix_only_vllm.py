from __future__ import annotations

import gc
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from synthrlvl.metrics import OutputEvaluator
from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def main() -> None:
    base_local = "/home/atuin/c107fa/c107fa12/RLVL/finetune/olmo3-7b-logic-lora-full-1ep-lr5e-4-seed101-v4-merged"
    wrapper = Path("/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/localbase_adapter_final_lr1e4").resolve()
    steps = [1, 11, 12, 13]

    records = []
    for s in steps:
        task_cfg = TaskConfig(
            template=TemplateName.LOGIC,
            prefill=PrefillMode.NONE,
            distractor_ratio=0.5,
            train_steps=StepRange(1, 10),
            val_steps=StepRange(s, s),
            seed=3407,
        )
        sample = TaskBuilder(task_cfg).build_samples(1, train=False)[0]
        records.append((s, sample))

    llm = LLM(
        model=base_local,
        tokenizer=str(wrapper),
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_num_seqs=32,
        enable_lora=True,
        disable_log_stats=True,
    )
    req = LoRARequest(
        lora_name="tokfix",
        lora_int_id=1,
        lora_path=str(wrapper),
        base_model_name=base_local,
    )
    outs = llm.generate(
        [s.prompt for _, s in records],
        sampling_params=SamplingParams(n=1, max_tokens=256, temperature=0.0, top_p=1.0),
        use_tqdm=False,
        lora_request=req,
    )
    gens = [o.outputs[0].text if o.outputs else "" for o in outs]

    ev = OutputEvaluator()
    print("=== tokenizer-fix only ===")
    for (step, sample), gen in zip(records, gens):
        m = ev.evaluate(
            gen,
            template=TemplateName.LOGIC,
            gold_answer=sample.answer,
            gold_logic_premises=sample.logic_premises,
            gold_logic_conclusion=sample.logic_conclusion,
            prefill=PrefillMode.NONE,
            gold_first_modality_lines=sample.gold_first_modality_lines,
        )
        print(f"step={step} | format={m.format_ok} correct={m.correct} valid={m.valid}")
        print(gen[:320].replace("\n", "\\n"))

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
