import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "nanotron"
    / "convert_qwen2_nanotron_to_hf.py"
)


def test_converter_normalizes_transformers_5_special_token_field() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_tokenizer_config"
    )
    assigned_keys = {
        target.slice.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    }
    popped_keys = {
        node.args[0].value
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert "additional_special_tokens" in assigned_keys
    assert "extra_special_tokens" in popped_keys
