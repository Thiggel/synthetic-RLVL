from __future__ import annotations

import pytest

from synthrlvl.metrics import _is_answer_match


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("sparse", True),
        ("Yara is sparse.", True),
        ("The answer is sparse.", True),
        ("dense sparse", False),
        ("dense\nsparse", False),
        ("Yara is dense, not sparse.", False),
    ],
)
def test_answer_match_rejects_alternative_lists(prediction: str, expected: bool) -> None:
    assert _is_answer_match(prediction, "sparse") is expected
