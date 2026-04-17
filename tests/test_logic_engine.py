from logic_engine import LogicEngine


def test_validate_simple_fol_proof():
    engine = LogicEngine()
    ok = engine.validate_proof(
        premises="P(a) -> Q(a), P(a)",
        conclusion="Q(a)",
        proof="Q(a) ; IE, 1,2",
    )
    assert ok is True


def test_per_line_validity_reports_invalid_line():
    engine = LogicEngine()
    report = engine.analyze_proof(
        premises="P(a) -> Q(a), P(a)",
        conclusion="Q(a)",
        proof="Q(a) ; IE, 1,3",
    )
    assert report.ok is False
    assert len(report.lines) == 1
    assert report.lines[0].valid is False
    assert report.lines[0].error is not None


def test_arithmetic_extension_with_arith_rule():
    engine = LogicEngine()
    report = engine.analyze_proof(
        premises="NA",
        conclusion="2 + 2 = 4",
        proof="2+2=4 ; ARITH",
    )
    assert report.ok is True
    assert report.lines[0].valid is True


def test_formula_equivalence_implication_vs_disjunction():
    engine = LogicEngine()
    assert engine.are_equivalent("P(a) -> Q(a)", "not P(a) or Q(a)") is True


def test_equation_equivalence_via_polynomial_normalization():
    engine = LogicEngine()
    assert engine.are_equivalent("x + 1 = 2", "2 = 1 + x") is True


def test_relevance_novelty_and_graph():
    engine = LogicEngine()
    report = engine.analyze_proof(
        premises="P(a)->Q(a), P(a), R(a)",
        conclusion="Q(a)",
        proof="""
Q(a) ; IE, 1,2
R(a) ; R, 3
""",
    )

    assert report.lines[0].valid is True
    assert report.lines[0].relevant is True
    assert report.lines[0].novel is True
    assert report.lines[0].relevant_and_novel is True

    assert report.lines[1].valid is True
    assert report.lines[1].relevant is False
    assert report.lines[1].novel is False
    assert report.lines[1].relevant_and_novel is False

    assert (1, 4) in report.graph.edges
    assert (2, 4) in report.graph.edges
    assert (3, 5) in report.graph.edges
