"""Ontology Verification, Rule-Base Audit, and Norm Calibration Benchmark (Stage S4 KR&R).

Evaluates:
1. Typed Ontology Graph structure (nodes: RootConcept, SourceDoc, ClauseFamily, Attribute,
   MarketNorm, Rule; edges: containsClause, hasAttribute, governs, citedIn, appliesTo,
   deviatesFrom, conformsTo, dependsOn).
2. Complete 30-rule YAML rule base syntax, ontology attribute grounding, and multi-step
   chained inference (`R-CT-02` => `InvestorControl` => `R-CT-05`).
3. Norm calibration and severity-weight calibration across the 30 synthetic gold term sheets
   (15 standard vs. 15 adversarial) in `corpus/synthetic/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from termsheet_ai.ontology import (  # noqa: E402
    build_ontology_graph,
    calibrate_severity_weights,
    evaluate_norm_calibration,
    get_canonical_clause_families,
    get_ontology_summary,
    get_rule_provenance,
    get_rules_for_family,
    load_rules_from_yaml,
    validate_rule_syntax,
    validate_rules_against_ontology,
)


def run_calibration_report() -> None:
    """Execute ontology, rule-base, and norm-calibration benchmark."""
    print("=" * 84)
    print("  TERMSHEET AI — KR&R ONTOLOGY, RULE BASE & NORM CALIBRATION BENCHMARK")
    print("=" * 84)

    # 1. Ontology Graph Audit
    graph = build_ontology_graph(include_rules=True)
    summary = get_ontology_summary(graph)
    print("\n1. ONTOLOGY SEMANTIC NETWORK SUMMARY")
    print("-" * 84)
    print(
        f"  Total Nodes: {summary['total_nodes']}  |  Total Directed Edges: {summary['total_edges']}"
    )
    print("  Nodes by Type:")
    for ntype, count in sorted(summary["node_counts_by_type"].items()):
        print(f"    - {ntype:<18}: {count}")
    print("  Edges by Relation:")
    for rel, count in sorted(summary["edge_counts_by_relation"].items()):
        print(f"    - {rel:<18}: {count}")

    # 2. Rule Syntax & Ontology Validation
    rules = load_rules_from_yaml()
    syntax_ok, syntax_errors = validate_rule_syntax(rules)
    ont_check = validate_rules_against_ontology(graph, rules)

    print("\n2. YAML RULE BASE & FAMILY COVERAGE")
    print("-" * 84)
    print(f"  Total Rules Loaded       : {len(rules)}")
    print(f"  AST Syntax Valid         : {syntax_ok} ({len(syntax_errors)} errors)")
    print(
        f"  Ontology Grounding Valid : {ont_check['valid_rules']}/{ont_check['total_rules']} "
        f"({len(ont_check['errors'])} errors)"
    )

    print(f"\n  {'Canonical Clause Family':<28} | {'Rule Count':<10} | {'Rule IDs'}")
    print("  " + "-" * 78)
    for fam in get_canonical_clause_families() + ["dividends"]:
        fam_rules = get_rules_for_family(fam, rules)
        ids = ", ".join(r["id"] for r in fam_rules)
        print(f"  {fam:<28} | {len(fam_rules):<10} | {ids}")

    # 3. Chained Rule Provenance Check
    ct05_prov = get_rule_provenance(graph, "R-CT-05")
    print("\n3. MULTI-STEP CHAINED INFERENCE PROVENANCE (R-CT-02 -> R-CT-05)")
    print("-" * 84)
    print(f"  Rule ID          : {ct05_prov['rule_id']} ({ct05_prov['severity']})")
    print(f"  Condition        : {ct05_prov['condition']}")
    print(f"  Conclusion       : {ct05_prov['conclusion']}")
    print(f"  Depends On Rules : {ct05_prov['depends_on_rules']}")
    print(f"  Deviates From    : {ct05_prov['market_norms']} ({ct05_prov['norm_relation']})")
    print(f"  Cited In         : {ct05_prov['source_docs']} ({ct05_prov['source']})")

    # 4. Gold-Corpus Norm Calibration & Severity Weight Evaluation
    cal_metrics = evaluate_norm_calibration()
    weight_cal = calibrate_severity_weights()

    print("\n4. GOLD-CORPUS NORM CALIBRATION (30 SYNTHETIC TERM SHEETS)")
    print("-" * 84)
    print(
        f"  Documents Evaluated      : {cal_metrics['total_documents']} "
        f"({cal_metrics['standard_documents']} Standard / {cal_metrics['adversarial_documents']} Adversarial)"
    )
    print(f"  Core Gold Flag Precision : {cal_metrics['core_flag_precision'] * 100:.2f}%")
    print(f"  Core Gold Flag Recall    : {cal_metrics['core_flag_recall'] * 100:.2f}%")
    print(f"  Core Gold Flag F1-Score  : {cal_metrics['core_flag_f1']:.4f}")
    print(f"  Exact Document Match     : {cal_metrics['exact_match_rate'] * 100:.2f}%")

    print("\n5. SEVERITY WEIGHT CALIBRATION & RISK SCORE DISTRIBUTION")
    print("-" * 84)
    print(f"  Selected Weight Profile  : {weight_cal['selected_profile']}")
    print(f"  Calibrated Weights       : {weight_cal['selected_weights']}")
    print(
        f"  Standard Sheets Score    : Mean = {cal_metrics['standard_mean_score']:.2f}/100 "
        f"(Range: {cal_metrics['standard_min_score']:.1f} - {cal_metrics['standard_max_score']:.1f})"
    )
    print(
        f"  Adversarial Sheets Score : Mean = {cal_metrics['adversarial_mean_score']:.2f}/100 "
        f"(Range: {cal_metrics['adversarial_min_score']:.1f} - {cal_metrics['adversarial_max_score']:.1f})"
    )
    print(f"  Score Separation Margin  : +{cal_metrics['score_separation_margin']:.2f} points")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    run_calibration_report()
