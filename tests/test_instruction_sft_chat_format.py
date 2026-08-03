from __future__ import annotations

import importlib.util
from pathlib import Path

from datasets import Dataset


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_instruction_sft.py"
WRAPPER = (
    Path(__file__).parents[1]
    / "scripts"
    / "slurm"
    / "jobs"
    / "nanotron_qwen25_instruction_sft_2026-06-24.slurm"
)
SPEC = importlib.util.spec_from_file_location("train_instruction_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    eos_token_id = 0
    chat_template = "fake"

    @staticmethod
    def _ids(text: str) -> list[int]:
        return [ord(char) + 1 for char in text]

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": self._ids(text)}

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize
        text = f"<user>{messages[0]['content']}</user><assistant>"
        if len(messages) == 2:
            text += f"{messages[1]['content']}</assistant>"
        elif not add_generation_prompt:
            raise AssertionError("user-only chat must request an assistant generation prompt")
        return self._ids(text)


def test_chat_format_uses_raw_user_and_assistant_text():
    row = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
    formatted = MODULE._row_to_prompt_target(
        row,
        wrap_question_tags=False,
        wrap_answer_tags=False,
    )
    assert formatted == {"prompt": "Question", "target": "Answer"}


def test_chat_tokenization_masks_prompt_and_keeps_assistant_tokens():
    tokenizer = FakeTokenizer()
    dataset = Dataset.from_list([{"prompt": "Q", "target": "A"}])
    tokenized = MODULE._tokenize_dataset(
        dataset,
        tokenizer,
        max_length=128,
        format_mode="chat",
    )
    row = tokenized[0]
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Q"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert row["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert row["labels"][len(prompt_ids) :] == row["input_ids"][len(prompt_ids) :]
    assert row["labels"][len(prompt_ids) :]


def test_tokenization_drops_rows_truncated_before_assistant_target():
    tokenizer = FakeTokenizer()
    short = {"prompt": "Q", "target": "A"}
    long = {"prompt": "Q" * 200, "target": "A"}
    dataset = Dataset.from_list([long, short])
    tokenized = MODULE._tokenize_dataset(
        dataset,
        tokenizer,
        max_length=64,
        format_mode="chat",
    )
    assert len(tokenized) == 1
    assert any(label != -100 for label in tokenized[0]["labels"])


def test_tagged_format_remains_available_for_legacy_runs():
    formatted = MODULE._row_to_prompt_target(
        {"instruction": "Question", "output": "Answer"},
        wrap_question_tags=True,
        wrap_answer_tags=True,
    )
    assert formatted == {
        "prompt": "<question>\nQuestion\n</question>\n",
        "target": "<answer>\nAnswer\n</answer>",
    }


def test_single_turn_filter_excludes_multiturn_rows():
    raw = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
        ]
    )
    formatted = MODULE._format_dataset(
        raw,
        limit=2,
        seed=3407,
        format_mode="chat",
        wrap_answer_tags=False,
        single_turn_only=True,
    )
    assert formatted.to_list() == [{"prompt": "Q", "target": "A"}]


def test_auto_resume_selects_latest_trainer_checkpoint(tmp_path: Path):
    (tmp_path / "checkpoint-100").mkdir()
    latest = tmp_path / "checkpoint-200"
    latest.mkdir()
    assert MODULE._resolve_resume_checkpoint(tmp_path, "auto") == str(latest)


def test_auto_resume_is_noop_for_clean_output_root(tmp_path: Path):
    assert MODULE._resolve_resume_checkpoint(tmp_path / "absent", "auto") is None


def test_nanotron_instruction_wrapper_enables_auto_resume():
    assert "--resume-from-checkpoint auto" in WRAPPER.read_text(encoding="utf-8")
