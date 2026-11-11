"""Comprehensive unit tests for the Venture Capital Term Sheet Ontology, Rule Base,
Severity Weights, and Norm Calibration (Member 5 — KR&R).
"""

import pytest

from termsheet_ai.ontology import (
    CORE_GOLD_RULE_IDS,
    DEFAULT_SEVERITY_WEIGHTS,
    build_ontology_graph,
    calibrate_severity_weights,
    compute_weighted_risk_score,
    evaluate_norm_calibration,
    get_canonical_clause_families,
    get_family_norm,
    get_market_norms,
    get_ontology_summary,
    get_rule_provenance,
    get_rules_for_family,
    get_severity_weights,
    load_rules_from_yaml,
    load_severity_weights_from_yaml,
    normalize_family_name,
    validate_rule_syntax,
    validate_rules_against_ontology,
)
from termsheet_ai.schemas.contracts import SeverityLevel


def test_canonical_families():
    families = get_canonical_clause_families()
    assert len(families) == 12
    assert "liquidation_preference" in families
    assert "anti_dilution" in families
    assert "board_composition" in families
    assert "founder_vesting" in families
    assert "protective_provisions" in families
    assert "information_rights" in families
    assert "rofr_co_sale" in families
    assert "drag_along" in families
    assert "option_pool" in families
    assert "exclusivity_no_shop" in families
    assert "conversion_rights" in families
    assert "redemption_rights" in families


def test_normalize_family_name_aliases():
    assert normalize_family_name("board_of_directors") == "board_composition"
    assert normalize_family_name("vesting") == "founder_vesting"
    assert normalize_family_name("exclusivity") == "exclusivity_no_shop"
    assert normalize_family_name("redemption") == "redemption_rights"
    assert normalize_family_name("liquidation_preference") == "liquidation_preference"


def test_market_norms_cover_all_canonical_families():
    norms = get_market_norms()
    for fam in get_canonical_clause_families():
        assert fam in norms, f"Missing market norm for canonical family {fam}"
        fam_norm = get_family_norm(fam)
        assert fam_norm is not None
        assert "market_source" in fam_norm
        assert "norm_id" in fam_norm

    assert norms["liquidation_preference"]["standard_multiple"] == 1.0
    assert norms["anti_dilution"]["standard_type"] == "broad_based_weighted_average"
    assert norms["founder_vesting"]["standard_vesting_months"] == 48
    assert norms["founder_vesting"]["standard_cliff_months"] == 12
    assert norms["exclusivity_no_shop"]["max_standard_days"] == 45
    assert norms["option_pool"]["typical_range_max"] == 20.0


def test_ontology_graph_structure_and_semantic_edges():
    graph = build_ontology_graph(include_rules=True)
    assert graph.number_of_nodes() >= 80
    assert graph.has_node("TermSheet")
    assert graph.nodes["TermSheet"]["type"] == "RootConcept"

    # Verify SourceDoc nodes
    for src_doc in (
        "NVCA_Model_Term_Sheet",
        "NVCA_Enhanced_Model_Term_Sheet",
        "YC_SAFE",
        "Market_Practice_Studies",
    ):
        assert graph.has_node(src_doc)
        assert graph.nodes[src_doc]["type"] == "SourceDoc"

    # Verify root links to all 12 canonical families and each family has attributes and norms
    for fam in get_canonical_clause_families():
        assert graph.has_node(fam)
        assert graph.nodes[fam]["type"] == "ClauseFamily"
        assert graph.has_edge("TermSheet", fam)
        assert graph.edges["TermSheet", fam]["relation"] == "containsClause"

        # Every canonical family has at least one Attribute via hasAttribute
        attr_edges = [
            target
            for _, target, d in graph.out_edges(fam, data=True)
            if d.get("relation") == "hasAttribute"
        ]
        assert len(attr_edges) >= 1, f"Family {fam} missing hasAttribute edges"

        # Every canonical family is governed by at least one MarketNorm
        norm_edges = [
            src for src, _, d in graph.in_edges(fam, data=True) if d.get("relation") == "governs"
        ]
        assert len(norm_edges) >= 1, f"Family {fam} missing governing MarketNorm"

    # Verify summary counts
    summary = get_ontology_summary(graph)
    assert summary["node_counts_by_type"]["SourceDoc"] == 4
    assert summary["node_counts_by_type"]["MarketNorm"] >= 12
    assert summary["node_counts_by_type"]["Rule"] >= 25
    assert summary["edge_counts_by_relation"]["deviatesFrom"] >= 20
    assert summary["edge_counts_by_relation"]["conformsTo"] >= 3
    assert summary["edge_counts_by_relation"]["dependsOn"] >= 1


def test_rule_provenance_and_chaining_in_ontology():
    graph = build_ontology_graph(include_rules=True)

    # Check chained rule R-CT-05 -> R-CT-02
    prov_ct05 = get_rule_provenance(graph, "R-CT-05")
    assert prov_ct05["rule_id"] == "R-CT-05"
    assert prov_ct05["severity"] == "HIGH"
    assert prov_ct05["conclusion"] == "FounderRemovableWithEquityLoss"
    assert "R-CT-02" in prov_ct05["depends_on_rules"]
    assert "Standard_4_Year_1_Year_Cliff_Vesting" in prov_ct05["market_norms"]
    assert prov_ct05["norm_relation"] == "deviatesFrom"
    assert "NVCA_Model_Term_Sheet" in prov_ct05["source_docs"]

    # Check conforming rule R-AD-02
    prov_ad02 = get_rule_provenance(graph, "R-AD-02")
    assert prov_ad02["norm_relation"] == "conformsTo"
    assert "Broad_Based_Weighted_Average" in prov_ad02["market_norms"]

    with pytest.raises(KeyError):
        get_rule_provenance(graph, "NON_EXISTENT_RULE")


def test_load_and_validate_rules_v1():
    rules = load_rules_from_yaml()
    assert len(rules) >= 25

    valid, errors = validate_rule_syntax(rules)
    assert valid is True, f"Rule syntax errors: {errors}"
    assert len(errors) == 0

    graph = build_ontology_graph(include_rules=True)
    ont_res = validate_rules_against_ontology(graph, rules)
    assert ont_res["valid_rules"] == len(rules), f"Ontology validation errors: {ont_res['errors']}"
    assert len(ont_res["errors"]) == 0


def test_every_canonical_family_has_rules():
    rules = load_rules_from_yaml()
    for fam in get_canonical_clause_families():
        fam_rules = get_rules_for_family(fam, rules)
        assert len(fam_rules) >= 1, f"Canonical family '{fam}' has no rules in rules_v1.yaml"


def test_validate_rule_syntax_rejects_malformed_rules():
    bad_rules = [
        {"id": "BAD-01", "family": "liquidation_preference"},  # missing fields
        {
            "id": "BAD-02",
            "family": "liquidation_preference",
            "condition": "multiple > 1",
            "severity": "EXTREME_INVALID",
            "message": "Invalid severity",
            "source": "NVCA",
        },
        {
            "id": "BAD-03",
            "family": "liquidation_preference",
            "condition": "multiple >>> === 1",
            "severity": "HIGH",
            "message": "Broken syntax",
            "source": "NVCA",
        },
    ]
    valid, errors = validate_rule_syntax(bad_rules)
    assert valid is False
    assert len(errors) == 3


def test_validate_rules_against_ontology_rejects_unknown_family_and_attribute():
    graph = build_ontology_graph(include_rules=False)
    bad_rules = [
        {
            "id": "BAD-FAM",
            "family": "maritime_salvage_law",
            "condition": "multiple > 1",
            "severity": "HIGH",
            "message": "Unknown family",
            "source": "NVCA",
        },
        {
            "id": "BAD-ATTR",
            "family": "liquidation_preference",
            "condition": "nonexistent_attribute_xyz > 10",
            "severity": "HIGH",
            "message": "Unknown attribute",
            "source": "NVCA",
        },
    ]
    res = validate_rules_against_ontology(graph, bad_rules)
    assert res["valid_rules"] == 0
    assert len(res["errors"]) == 2


def test_severity_weights_and_risk_score_computation():
    weights = load_severity_weights_from_yaml()
    assert weights == DEFAULT_SEVERITY_WEIGHTS
    assert get_severity_weights()[SeverityLevel.CRITICAL] == 40.0
    assert get_severity_weights()[SeverityLevel.HIGH] == 25.0
    assert get_severity_weights()[SeverityLevel.MEDIUM] == 10.0
    assert get_severity_weights()[SeverityLevel.LOW] == 5.0
    assert get_severity_weights()[SeverityLevel.INFO] == 0.0

    # 1 High (25) + 1 Medium (10) + 1 Info (0) = 35.0
    score = compute_weighted_risk_score(
        [SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.INFO]
    )
    assert score == 35.0

    # Score caps at 100.0
    capped = compute_weighted_risk_score(
        [SeverityLevel.CRITICAL, SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    )
    assert capped == 100.0


def test_norm_calibration_on_synthetic_gold_corpus():
    metrics = evaluate_norm_calibration()
    assert metrics["total_documents"] == 30
    assert metrics["standard_documents"] == 15
    assert metrics["adversarial_documents"] == 15
    assert len(CORE_GOLD_RULE_IDS) == 14

    # 100% precision, recall, F1, and exact match on core gold expected_flags
    assert metrics["core_flag_precision"] == 1.0
    assert metrics["core_flag_recall"] == 1.0
    assert metrics["core_flag_f1"] == 1.0
    assert metrics["exact_match_rate"] == 1.0

    # Clean NVCA baseline sheet scores low (<= 10), adversarial sheets score high (100)
    assert metrics["clean_nvca_baseline_score"] <= 10.0
    assert metrics["adversarial_mean_score"] >= 95.0
    assert metrics["baseline_to_adversarial_gap"] >= 85.0
    assert metrics["score_separation_margin"] > 30.0


def test_calibrate_severity_weights_selects_production_profile():
    cal = calibrate_severity_weights()
    assert cal["selected_profile"] == "calibrated_production"
    assert cal["selected_weights"]["critical"] == 40.0
    assert cal["selected_weights"]["high"] == 25.0
    assert "uncalibrated_uniform" in cal["profiles"]
    assert "under_weighted" in cal["profiles"]
