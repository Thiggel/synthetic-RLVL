from synthetic_dataset import DatasetConfig, LogicDatasetGenerator


def test_position_shortcut_puts_gold_branch_first_when_enabled():
    gen = LogicDatasetGenerator(
        DatasetConfig(
            depth=5,
            difficulty="hard_fsa_schema",
            branching_factor=4,
            shortcut_rate=1.0,
            shortcut_kind="position",
            seed=3407,
        )
    )
    ex = gen.generate(0)

    assert ex.metadata["shortcut_kind"] == "position"
    assert ex.metadata["shortcut_enabled"] is True
    assert all(order[0].startswith("branch0:") for order in ex.metadata["branch_orders"])


def test_initial_marker_shortcut_fixes_gold_initial_marker_when_enabled():
    gen = LogicDatasetGenerator(
        DatasetConfig(
            depth=5,
            difficulty="hard_fsa_schema",
            branching_factor=4,
            shortcut_rate=1.0,
            shortcut_kind="initial_marker",
            seed=3407,
        )
    )
    ex = gen.generate(0)

    assert ex.metadata["shortcut_kind"] == "initial_marker"
    assert ex.metadata["shortcut_enabled"] is True
    assert ex.metadata["path_markers"][0] == "north"
