"""Tests for the Venture Capital Term Sheet Ontology and rule validation."""

from termsheet_ai.ontology import (
    build_ontology_graph,
    get_canonical_clause_families,
    get_market_norms,
    load_rules_from_yaml,
    validate_rule_syntax,
)


def test_canonical_families():
    families = get_canonical_clause_families()
    assert len(families) == 12
    assert "liquidation_preference" in families
    assert "anti_dilution" in families
    assert "board_composition" in families
    assert "founder_vesting" in families


def test_ontology_graph_structure():
    graph = build_ontology_graph()
    assert graph.number_of_nodes() >= 13  # Root + 12 families
    assert graph.has_node("TermSheet")

    # Verify root links to families
    for fam in get_canonical_clause_families():
        assert graph.has_node(fam)
        assert graph.has_edge("TermSheet", fam)

    # Verify market norms attached
    norms = get_market_norms()
    assert "liquidation_preference" in norms
    assert norms["liquidation_preference"]["standard_multiple"] == 1.0


def test_load_and_validate_rules_v1():
    rules = load_rules_from_yaml()
    assert len(rules) >= 20  # Expanded rule base has 25 rules

    valid, errors = validate_rule_syntax(rules)
    assert valid is True, f"Rule syntax errors: {errors}"
    assert len(errors) == 0
