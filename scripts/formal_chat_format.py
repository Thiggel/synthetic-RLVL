"""Shared chat rendering and tool-result masking for the formal-mixture SFT.

The trainer (scripts/train_formal_mixture_sft.py) and the vLLM evaluator
(scripts/eval_formal_vllm.py) both build token ids through this module, so the
model sees the same token boundaries at train and at test time.

Template: the model's own chat template (Qwen3.5 base models ship ChatML).
  prompt     = apply_chat_template([user], add_generation_prompt=True, enable_thinking=False)
             = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
  completion = "{assistant.strip()}<|im_end|>\n"   (see render_completion)
The prompt and the completion are tokenized separately, exactly as a vLLM text
prompt followed by generation would be.

Tool results. A synthetic `tools` proof contains blocks
    N do search("...")\n<result>{content}</result>\n
The content is written by the tool simulator at test time, not by the model,
so it is masked from the loss together with the closing `</result>\n`; the
opening `<result>` stays in the loss because the model must learn to emit it
(vLLM stops on it). Qwen's BPE merges `>` with the first content word (`>About`),
which would put tool text into a supervised token. To keep a clean boundary the
assistant text is cut after every `<result>` and after every `</result>\n` and
each piece is tokenized on its own; the evaluator rebuilds its continuation
prompts with the same cut, so the ids match training.
"""
from __future__ import annotations

import re

RESULT_OPEN = "<result>"
RESULT_CLOSE = "</result>\n"
END_OF_TURN = "<|im_end|>\n"  # ChatML turn end (Qwen3.5 base templates)
_RESULT_RE = re.compile(r"<result>(.*?)</result>\n", re.DOTALL)


def render_prompt(tokenizer, user_text: str) -> str:
    # enable_thinking=False pins the empty think block. The 0.8B/2B templates
    # emit it by default, but the 9B template defaults to an open "<think>\n"
    # generation prompt, which would not match the rendered training turn.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def render_completion(tokenizer, user_text: str, assistant_text: str) -> str:
    """The assistant turn as the template renders it: stripped content + "<|im_end|>\n".

    For ChatML/Qwen3.5 this equals full_template[len(prompt):] for every
    Dolci and rlvlgen row except the 13/102,046 Dolci rows whose answers contain
    a literal "</think>", which the Qwen template would move into a reasoning
    block. Those keep their literal text, which is what the model would have to
    generate after the generation prompt.
    """
    return assistant_text.strip() + END_OF_TURN


def template_matches(tokenizer, user_text: str, assistant_text: str) -> bool:
    prompt = render_prompt(tokenizer, user_text)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return full == prompt + render_completion(tokenizer, user_text, assistant_text)


def split_tool_segments(text: str, mask_tools: bool) -> list[tuple[str, bool]]:
    """Cut text into (piece, trainable) pieces at tool-result boundaries.

    Without mask_tools the whole text is one trainable piece.
    """
    if not mask_tools:
        return [(text, True)] if text else []
    pieces: list[tuple[str, bool]] = []
    pos = 0
    for m in _RESULT_RE.finditer(text):
        open_end = m.start() + len(RESULT_OPEN)
        pieces.append((text[pos:open_end], True))
        pieces.append((text[open_end:m.end()], False))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], True))
    # A continuation prompt at eval ends right after "<result>" + result text +
    # "</result>\n"; an unterminated trailing "<result>" is still cut after the tag.
    return [p for p in pieces if p[0]]


def encode_pieces(tokenizer, pieces: list[tuple[str, bool]]) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    train: list[bool] = []
    for text, trainable in pieces:
        piece_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(piece_ids)
        train.extend([trainable] * len(piece_ids))
    return ids, train


def encode_example(tokenizer, user_text: str, assistant_text: str, mask_tools: bool) -> dict:
    """input_ids and labels (-100 = no loss) for one single-turn example."""
    prompt_ids = tokenizer(render_prompt(tokenizer, user_text), add_special_tokens=False)["input_ids"]
    completion = render_completion(tokenizer, user_text, assistant_text)
    comp_ids, comp_train = encode_pieces(tokenizer, split_tool_segments(completion, mask_tools))
    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + [t if keep else -100 for t, keep in zip(comp_ids, comp_train)]
    return {"input_ids": input_ids, "labels": labels, "n_prompt": len(prompt_ids)}


def encode_continuation(tokenizer, prompt_ids: list[int], assistant_so_far: str) -> list[int]:
    """Prompt ids for resuming generation after a tool result (eval side)."""
    ids, _ = encode_pieces(tokenizer, split_tool_segments(assistant_so_far, True))
    return list(prompt_ids) + ids
