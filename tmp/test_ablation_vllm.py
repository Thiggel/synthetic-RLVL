from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from posttrain_grpo_verl import _merge_lora_adapter_checkpoint
from synthrlvl.metrics import OutputEvaluator
from synthrlvl.task import TaskBuilder
from synthrlvl.types import PrefillMode, StepRange, TaskConfig, TemplateName


def build_wrapper(orig_adapter: Path, wrapper: Path, base_local: str) -> Path:
    wrapper.mkdir(parents=True, exist_ok=True)
    for name in [
        "README.md",
        "adapter_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "training_args.bin",
    ]:
        src = orig_adapter / name
        if src.exists():
            dst = wrapper / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst)

    w_src = orig_adapter / "adapter_model.safetensors"
    w_dst = wrapper / "adapter_model.safetensors"
    if w_dst.exists() or w_dst.is_symlink():
        w_dst.unlink()
    w_dst.symlink_to(w_src)

    cfg_p = wrapper / "adapter_config.json"
    cfg = json.loads(cfg_p.read_text())
    cfg["base_model_name_or_path"] = base_local
    cfg_p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return wrapper


def build_records(steps: list[int]):
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
    return records


def score_rows(name: str, records, gens):
    ev = OutputEvaluator()
    print(f"=== {name} ===")
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


def run_tokfix_only(base_local: str, wrapper: Path, prompts: list[str]):
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
    sampling = SamplingParams(n=1, max_tokens=256, temperature=0.0, top_p=1.0)
    outs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False, lora_request=req)
    gens = [o.outputs[0].text if o.outputs else "" for o in outs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return gens


def run_merge_only(wrapper: Path, prompts: list[str]):
    merged = Path("/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/merged_final_lr1e4_smoke").resolve()
    merged_path = _merge_lora_adapter_checkpoint(adapter_path=str(wrapper), merged_output_dir=merged)

    llm = LLM(
        model=merged_path,
        tokenizer=merged_path,
        trust_remote_code=False,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.55,
        max_num_seqs=32,
        enable_lora=False,
        disable_log_stats=True,
    )
    sampling = SamplingParams(n=1, max_tokens=256, temperature=0.0, top_p=1.0)
    outs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    gens = [o.outputs[0].text if o.outputs else "" for o in outs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return merged_path, gens


def main():
    base_local = "/home/atuin/c107fa/c107fa12/RLVL/finetune/olmo3-7b-logic-lora-full-1ep-lr5e-4-seed101-v4-merged"
    orig_adapter = Path("/home/atuin/c107fa/c107fa12/synthetic-RLVL/runs/sft_lr_lr1e-4_seed3409/final").resolve()
    wrapper = Path("/home/atuin/c107fa/c107fa12/synthetic-RLVL/tmp/localbase_adapter_final_lr1e4").resolve()

    build_wrapper(orig_adapter=orig_adapter, wrapper=wrapper, base_local=base_local)
    steps = [1, 11, 12, 13]
    records = build_records(steps)
    prompts = [s.prompt for _, s in records]

    gens_a = run_tokfix_only(base_local=base_local, wrapper=wrapper, prompts=prompts)
    score_rows("tokenizer-fix only", records, gens_a)

    merged_path, gens_b = run_merge_only(wrapper=wrapper, prompts=prompts)
    print(f"merged_path={merged_path}")
    score_rows("merge-only", records, gens_b)


if __name__ == "__main__":
    main()
