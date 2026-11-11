"""Ontology, Knowledge Graph, Severity Weights, and Norm Calibration for TermSheet AI.

Builds a typed NetworkX directed semantic network encoding:
- Source Documents (NVCA Model Term Sheet, NVCA Enhanced, YC SAFE, Market Practice Studies)
- Clause Families (the 12 canonical families + auxiliary dividends)
- Clause Attributes (typed ground and derived predicates)
- Market Norms and baseline values citing source documents
- Symbolic Horn Rules with `appliesTo`, `deviatesFrom` / `conformsTo`, `citedIn`, and `dependsOn` edges
- Severity weight loading and norm calibration against gold-labeled evaluation corpora
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import yaml

from termsheet_ai.schemas.contracts import ClauseFamily, SeverityLevel

CANONICAL_FAMILIES: List[str] = [
    ClauseFamily.LIQUIDATION_PREFERENCE.value,
    ClauseFamily.ANTI_DILUTION.value,
    ClauseFamily.BOARD_COMPOSITION.value,
    ClauseFamily.FOUNDER_VESTING.value,
    ClauseFamily.PROTECTIVE_PROVISIONS.value,
    ClauseFamily.INFORMATION_RIGHTS.value,
    ClauseFamily.ROFR_CO_SALE.value,
    ClauseFamily.DRAG_ALONG.value,
    ClauseFamily.OPTION_POOL.value,
    ClauseFamily.EXCLUSIVITY_NO_SHOP.value,
    ClauseFamily.CONVERSION_RIGHTS.value,
    ClauseFamily.REDEMPTION_RIGHTS.value,
]

# Maps common rule/schema family aliases to canonical ClauseFamily values
FAMILY_ALIASES: Dict[str, str] = {
    "liquidation_preference": ClauseFamily.LIQUIDATION_PREFERENCE.value,
    "anti_dilution": ClauseFamily.ANTI_DILUTION.value,
    "board_composition": ClauseFamily.BOARD_COMPOSITION.value,
    "board_of_directors": ClauseFamily.BOARD_COMPOSITION.value,
    "founder_vesting": ClauseFamily.FOUNDER_VESTING.value,
    "vesting": ClauseFamily.FOUNDER_VESTING.value,
    "protective_provisions": ClauseFamily.PROTECTIVE_PROVISIONS.value,
    "information_rights": ClauseFamily.INFORMATION_RIGHTS.value,
    "rofr_co_sale": ClauseFamily.ROFR_CO_SALE.value,
    "drag_along": ClauseFamily.DRAG_ALONG.value,
    "option_pool": ClauseFamily.OPTION_POOL.value,
    "exclusivity_no_shop": ClauseFamily.EXCLUSIVITY_NO_SHOP.value,
    "exclusivity": ClauseFamily.EXCLUSIVITY_NO_SHOP.value,
    "conversion_rights": ClauseFamily.CONVERSION_RIGHTS.value,
    "redemption_rights": ClauseFamily.REDEMPTION_RIGHTS.value,
    "redemption": ClauseFamily.REDEMPTION_RIGHTS.value,
    "dividends": "dividends",
}

DEFAULT_SEVERITY_WEIGHTS: Dict[SeverityLevel, float] = {
    SeverityLevel.CRITICAL: 40.0,
    SeverityLevel.HIGH: 25.0,
    SeverityLevel.MEDIUM: 10.0,
    SeverityLevel.LOW: 5.0,
    SeverityLevel.INFO: 0.0,
}

# Core 14 rules tracked in the synthetic generator's v0 gold labels
CORE_GOLD_RULE_IDS: Set[str] = {
    "R-LP-01",
    "R-LP-02",
    "R-LP-03",
    "R-AD-01",
    "R-AD-02",
    "R-CT-02",
    "R-CT-03",
    "R-VS-01",
    "R-VS-02",
    "R-OP-01",
    "R-EX-01",
    "R-DA-01",
    "R-RD-01",
    "R-DV-01",
}

MARKET_NORMS: Dict[str, Dict[str, Any]] = {
    "liquidation_preference": {
        "norm_id": "1x_Non_Participating",
        "standard_multiple": 1.0,
        "critical_multiple_threshold": 3.0,
        "standard_participation": "non_participating",
        "standard_seniority": "pari_passu",
        "market_source": "NVCA Model Term Sheet §3",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "anti_dilution": {
        "norm_id": "Broad_Based_Weighted_Average",
        "standard_type": "broad_based_weighted_average",
        "market_source": "NVCA Model Term Sheet §4",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "board_composition": {
        "norm_id": "Balanced_Or_Founder_Control_Board",
        "standard_control": "founder_majority_or_parity",
        "min_independent_seats": 1,
        "market_source": "Market Practice Studies (Gompers & Lerner)",
        "source_doc": "Market_Practice_Studies",
    },
    "board_of_directors": {
        "norm_id": "Balanced_Or_Founder_Control_Board",
        "standard_control": "founder_majority_or_parity",
        "min_independent_seats": 1,
        "market_source": "Market Practice Studies (Gompers & Lerner)",
        "source_doc": "Market_Practice_Studies",
    },
    "founder_vesting": {
        "norm_id": "Standard_4_Year_1_Year_Cliff_Vesting",
        "standard_vesting_years": 4.0,
        "standard_vesting_months": 48,
        "standard_cliff_months": 12,
        "standard_acceleration": "double_trigger",
        "standard_vesting_reset": False,
        "market_source": "YC Post-Money SAFE / NVCA Model Term Sheet",
        "source_doc": "YC_SAFE",
    },
    "vesting": {
        "norm_id": "Standard_4_Year_1_Year_Cliff_Vesting",
        "standard_vesting_years": 4.0,
        "standard_vesting_months": 48,
        "standard_cliff_months": 12,
        "standard_acceleration": "double_trigger",
        "standard_vesting_reset": False,
        "market_source": "YC Post-Money SAFE / NVCA Model Term Sheet",
        "source_doc": "YC_SAFE",
    },
    "protective_provisions": {
        "norm_id": "Standard_Major_Action_Protective_Provisions",
        "standard_requires_series_majority": True,
        "standard_veto_day_to_day_operations": False,
        "market_source": "NVCA Model Term Sheet §3 (Protective Provisions)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "information_rights": {
        "norm_id": "Major_Investor_Information_Rights",
        "standard_has_major_investor_threshold": True,
        "standard_monthly_reporting_required": False,
        "annual_financials_days": 120,
        "quarterly_financials_days": 45,
        "market_source": "NVCA Model Term Sheet §4 (Information and Observer Rights)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "rofr_co_sale": {
        "norm_id": "Standard_Company_Then_Investor_ROFR",
        "standard_investor_co_sale": True,
        "standard_full_founder_transfer_lockup": False,
        "market_source": "NVCA Model Term Sheet §4 (Right of First Refusal / Co-Sale)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "drag_along": {
        "norm_id": "Supermajority_And_Board_Drag_Along",
        "min_approval_threshold": 50.0,
        "standard_approval_threshold": 60.0,
        "standard_requires_founder_consent": True,
        "market_source": "NVCA Model Term Sheet §4 (Drag-Along)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "option_pool": {
        "norm_id": "Standard_10_To_20_Pct_Option_Pool",
        "typical_range_min": 10.0,
        "typical_range_max": 20.0,
        "max_pre_money_pool": 15.0,
        "market_source": "NVCA Enhanced Model Term Sheet / Market Norm",
        "source_doc": "NVCA_Enhanced_Model_Term_Sheet",
    },
    "exclusivity_no_shop": {
        "norm_id": "30_To_45_Day_No_Shop",
        "min_standard_days": 30,
        "max_standard_days": 45,
        "market_source": "NVCA Model Term Sheet §6 (No-Shop)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "exclusivity": {
        "norm_id": "30_To_45_Day_No_Shop",
        "min_standard_days": 30,
        "max_standard_days": 45,
        "market_source": "Market Norm",
        "source_doc": "Market_Practice_Studies",
    },
    "conversion_rights": {
        "norm_id": "Standard_1_To_1_Conversion_Rights",
        "standard_conversion_ratio": 1.0,
        "max_qualified_ipo_threshold": 100.0,
        "market_source": "NVCA Model Term Sheet §4 (Optional and Automatic Conversion)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "redemption_rights": {
        "norm_id": "No_Early_Mandatory_Redemption",
        "min_redemption_years": 5,
        "max_interest_rate": 10.0,
        "market_source": "NVCA Model Term Sheet §4 (Redemption Rights)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "redemption": {
        "norm_id": "No_Early_Mandatory_Redemption",
        "min_redemption_years": 5,
        "max_interest_rate": 10.0,
        "market_source": "NVCA Model Term Sheet §4 (Redemption Rights)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
    "dividends": {
        "norm_id": "No_Cumulative_Dividends",
        "standard_cumulative": False,
        "typical_non_cumulative_rate": 8.0,
        "market_source": "NVCA Model Term Sheet §3 (Dividends)",
        "source_doc": "NVCA_Model_Term_Sheet",
    },
}

FAMILY_ATTRIBUTES: Dict[str, List[str]] = {
    "liquidation_preference": ["multiple", "participation", "cap", "seniority"],
    "anti_dilution": ["type", "carveouts"],
    "board_composition": [
        "founder_seats",
        "investor_seats",
        "independent_seats",
        "total_seats",
        "InvestorControl",
    ],
    "board_of_directors": [
        "founder_seats",
        "investor_seats",
        "independent_seats",
        "total_seats",
        "InvestorControl",
    ],
    "protective_provisions": [
        "veto_day_to_day_operations",
        "requires_series_majority",
        "negative_covenants",
    ],
    "founder_vesting": [
        "vesting_months",
        "cliff_months",
        "acceleration",
        "VestingReset",
    ],
    "vesting": [
        "vesting_months",
        "cliff_months",
        "acceleration",
        "VestingReset",
    ],
    "information_rights": [
        "has_major_investor_threshold",
        "monthly_reporting_required",
        "annual_financials_days",
        "quarterly_financials_days",
    ],
    "rofr_co_sale": [
        "investor_co_sale",
        "full_founder_transfer_lockup",
        "rofr_applies_to_investors",
    ],
    "drag_along": [
        "approval_threshold",
        "requires_founder_consent",
    ],
    "option_pool": ["pool_percentage", "pre_money_pool"],
    "exclusivity_no_shop": ["duration_days", "fiduciary_out"],
    "exclusivity": ["duration_days", "fiduciary_out"],
    "conversion_rights": [
        "optional_conversion_ratio",
        "automatic_conversion_ipo_threshold",
    ],
    "redemption_rights": ["redemption_years", "interest_rate"],
    "redemption": ["redemption_years", "interest_rate"],
    "dividends": ["cumulative", "annual_rate"],
}


def normalize_family_name(family: str) -> str:
    """Normalizes a clause family or rule family alias to its canonical ClauseFamily string."""
    clean = (family or "").strip().lower()
    return FAMILY_ALIASES.get(clean, clean)


def get_canonical_clause_families() -> List[str]:
    """Returns the list of 12 canonical clause families."""
    return list(CANONICAL_FAMILIES)


def get_market_norms() -> Dict[str, Dict[str, Any]]:
    """Returns the baseline market norms dictionary."""
    return dict(MARKET_NORMS)


def get_family_norm(family: str) -> Optional[Dict[str, Any]]:
    """Returns the calibrated market norm entry for a given clause family or alias."""
    clean = (family or "").strip().lower()
    if clean in MARKET_NORMS:
        return dict(MARKET_NORMS[clean])
    norm_fam = normalize_family_name(clean)
    if norm_fam in MARKET_NORMS:
        return dict(MARKET_NORMS[norm_fam])
    return None


def _resolve_source_doc_id(source_str: str) -> str:
    """Maps a human-readable source string to its canonical SourceDoc node ID."""
    lower = (source_str or "").lower()
    if "enhanced" in lower:
        return "NVCA_Enhanced_Model_Term_Sheet"
    if "yc" in lower or "safe" in lower:
        return "YC_SAFE"
    if "nvca" in lower:
        return "NVCA_Model_Term_Sheet"
    return "Market_Practice_Studies"


def build_ontology_graph(
    include_rules: bool = True,
    yaml_path: Optional[Path] = None,
) -> nx.DiGraph:
    """Constructs the comprehensive typed ontology semantic network for term sheet evaluation.

    Includes typed nodes (`RootConcept`, `SourceDoc`, `ClauseFamily`, `Attribute`,
    `MarketNorm`, `Rule`) and semantic edges (`containsClause`, `hasAttribute`,
    `governs`, `citedIn`, `appliesTo`, `deviatesFrom`, `conformsTo`, `dependsOn`).
    """
    G = nx.DiGraph()

    # Root Term Sheet concept
    G.add_node("TermSheet", type="RootConcept", description="Venture Capital Term Sheet")

    # 1. Source Documents
    source_docs = [
        ("NVCA_Model_Term_Sheet", "NVCA Model Legal Documents (2020 revision)"),
        (
            "NVCA_Enhanced_Model_Term_Sheet",
            "NVCA Enhanced Model Term Sheet with Deal Term Frequency",
        ),
        ("YC_SAFE", "Y Combinator Standard Post-Money SAFE (2018)"),
        (
            "Market_Practice_Studies",
            "Empirical VC Contract Studies (Kaplan & Strömberg, Gompers & Lerner)",
        ),
    ]
    for doc_id, desc in source_docs:
        G.add_node(doc_id, type="SourceDoc", description=desc)

    # 2. Canonical Clause Families
    for fam in CANONICAL_FAMILIES:
        G.add_node(fam, type="ClauseFamily")
        G.add_edge("TermSheet", fam, relation="containsClause")

    # Family attributes across all canonical families and aliases
    for fam, attributes in FAMILY_ATTRIBUTES.items():
        if not G.has_node(fam):
            G.add_node(fam, type="ClauseFamily")
            G.add_edge("TermSheet", fam, relation="containsClause")
        for attr in attributes:
            G.add_node(attr, type="Attribute")
            G.add_edge(fam, attr, relation="hasAttribute")

    # 3. Market Norms and Citations covering all 12 canonical families + dividends
    market_norms = [
        (
            "1x_Non_Participating",
            "1.0x non-participating preferred is standard market practice",
            "liquidation_preference",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "Broad_Based_Weighted_Average",
            "Broad-based weighted average anti-dilution is market standard",
            "anti_dilution",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "Balanced_Or_Founder_Control_Board",
            "Founders retain parity or majority in early-stage rounds with independent representation",
            "board_composition",
            "Market_Practice_Studies",
        ),
        (
            "Standard_4_Year_1_Year_Cliff_Vesting",
            "48 months total vesting with 12 months cliff and double-trigger acceleration",
            "founder_vesting",
            "YC_SAFE",
        ),
        (
            "Standard_Major_Action_Protective_Provisions",
            "Preferred majority consent required only for major structural corporate actions, not day-to-day operations",
            "protective_provisions",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "Major_Investor_Information_Rights",
            "Annual audited and quarterly unaudited financials provided to Major Investors",
            "information_rights",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "Standard_Company_Then_Investor_ROFR",
            "Primary Company ROFR followed by Investor secondary ROFR and pro-rata co-sale rights",
            "rofr_co_sale",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "Supermajority_And_Board_Drag_Along",
            "Drag-along requires majority/supermajority preferred and common/board approval",
            "drag_along",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "Standard_10_To_20_Pct_Option_Pool",
            "Employee option pool typically ranges between 10% and 20% of post-money capitalization",
            "option_pool",
            "NVCA_Enhanced_Model_Term_Sheet",
        ),
        (
            "30_To_45_Day_No_Shop",
            "Exclusivity (no-shop) period should not exceed 30-45 calendar days",
            "exclusivity_no_shop",
            "Market_Practice_Studies",
        ),
        (
            "Standard_1_To_1_Conversion_Rights",
            "1:1 optional conversion into Common Stock and automatic conversion on Qualified IPO",
            "conversion_rights",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "No_Early_Mandatory_Redemption",
            "No mandatory redemption or exercisable only after 5+ years from closing",
            "redemption_rights",
            "NVCA_Model_Term_Sheet",
        ),
        (
            "No_Cumulative_Dividends",
            "Dividends payable only when and if declared by the Board (non-cumulative)",
            "dividends",
            "NVCA_Model_Term_Sheet",
        ),
    ]

    for norm_id, description, family, source_doc in market_norms:
        G.add_node(norm_id, type="MarketNorm", description=description)
        G.add_edge(norm_id, family, relation="governs")
        G.add_edge(norm_id, source_doc, relation="citedIn")
        # Also link rule-family alias nodes if distinct from canonical name
        for alias, canon in FAMILY_ALIASES.items():
            if canon == family and alias != family and G.has_node(alias):
                G.add_edge(norm_id, alias, relation="governs")

    # 4. Symbolic Rules and Rule-to-Norm / Chaining Edges
    if include_rules:
        try:
            rules = load_rules_from_yaml(yaml_path)
        except FileNotFoundError:
            rules = []

        conclusion_to_rule: Dict[str, str] = {}
        for r in rules:
            rid = r.get("id")
            concl = r.get("conclusion")
            if rid and concl:
                conclusion_to_rule[concl] = rid

        for r in rules:
            rid = r.get("id")
            if not rid:
                continue
            raw_fam = r.get("family", "other").lower()
            canon_fam = normalize_family_name(raw_fam)
            sev = str(r.get("severity", "MEDIUM")).upper()
            cond = str(r.get("condition", ""))
            msg = str(r.get("message", ""))
            src = str(r.get("source", "NVCA Model Term Sheet"))
            concl = r.get("conclusion", f"Flag: {msg}")
            norm_id = r.get("norm_id") or MARKET_NORMS.get(canon_fam, {}).get("norm_id")
            source_doc_id = _resolve_source_doc_id(src)

            G.add_node(
                rid,
                type="Rule",
                family=canon_fam,
                raw_family=raw_fam,
                condition=cond,
                severity=sev,
                message=msg,
                source=src,
                conclusion=concl,
                norm_id=norm_id,
            )

            if G.has_node(canon_fam):
                G.add_edge(rid, canon_fam, relation="appliesTo")
            if raw_fam != canon_fam and G.has_node(raw_fam):
                G.add_edge(rid, raw_fam, relation="appliesTo")

            if norm_id and G.has_node(norm_id):
                rel = "conformsTo" if sev in ("INFO", "LOW") else "deviatesFrom"
                G.add_edge(rid, norm_id, relation=rel)

            if G.has_node(source_doc_id):
                G.add_edge(rid, source_doc_id, relation="citedIn")

            # Link chained rules if condition references another rule's conclusion predicate
            for derived_pred, parent_rule_id in conclusion_to_rule.items():
                if parent_rule_id != rid and derived_pred in cond:
                    G.add_edge(rid, parent_rule_id, relation="dependsOn")

    return G


def load_rules_from_yaml(yaml_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads and returns the list of rule dictionaries from rules YAML file."""
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent.parent.parent / "docs" / "rules_v1.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Rules YAML not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    rules = data.get("rules", [])
    return rules


def load_severity_weights_from_yaml(
    yaml_path: Optional[Path] = None,
) -> Dict[SeverityLevel, float]:
    """Loads calibrated severity weights from rules_v1.yaml, falling back to defaults."""
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent.parent.parent / "docs" / "rules_v1.yaml"

    weights = dict(DEFAULT_SEVERITY_WEIGHTS)
    if not yaml_path.exists():
        return weights

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_weights = data.get("severity_weights", {})
    if isinstance(raw_weights, dict):
        for k, v in raw_weights.items():
            try:
                sev = SeverityLevel(str(k).lower())
                weights[sev] = float(v)
            except (ValueError, TypeError):
                continue

    return weights


def get_severity_weights(yaml_path: Optional[Path] = None) -> Dict[SeverityLevel, float]:
    """Returns the calibrated severity weight mapping."""
    return load_severity_weights_from_yaml(yaml_path)


def compute_weighted_risk_score(
    severities: List[SeverityLevel],
    weights: Optional[Dict[SeverityLevel, float]] = None,
) -> float:
    """Computes the normalized [0, 100] document risk score from triggered flag severities."""
    effective_weights = weights or DEFAULT_SEVERITY_WEIGHTS
    raw_total = sum(effective_weights.get(s, 0.0) for s in severities)
    return min(100.0, max(0.0, raw_total))


def validate_rule_syntax(rules: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validates schema, fields, and condition syntax for a list of rule dictionaries."""
    errors: List[str] = []
    required_fields = {"id", "family", "condition", "severity", "message", "source"}

    seen_ids: Set[str] = set()
    for idx, rule in enumerate(rules, start=1):
        missing = required_fields - set(rule.keys())
        if missing:
            errors.append(f"Rule #{idx} ({rule.get('id', 'unknown')}) missing fields: {missing}")
            continue

        rule_id = str(rule["id"])
        if rule_id in seen_ids:
            errors.append(f"Duplicate rule ID: {rule_id}")
        seen_ids.add(rule_id)

        # Severity validation
        try:
            SeverityLevel(str(rule["severity"]).lower())
        except ValueError:
            errors.append(f"Rule {rule_id} invalid severity: {rule['severity']}")

        # Condition syntax validation
        cond = (
            str(rule["condition"])
            .replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        try:
            ast.parse(cond, mode="eval")
        except SyntaxError as e:
            errors.append(f"Rule {rule_id} condition syntax error: {e}")

    return len(errors) == 0, errors


def validate_rules_against_ontology(
    graph: nx.DiGraph, rules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validates that loaded rules refer to valid families and ontology attributes."""
    valid_families = {
        n.lower() for n, d in graph.nodes(data=True) if d.get("type") == "ClauseFamily"
    }
    valid_families.update(FAMILY_ALIASES.keys())

    valid_attributes = {n for n, d in graph.nodes(data=True) if d.get("type") == "Attribute"}
    # Also permit derived rule conclusions as valid condition identifiers
    for r in rules:
        if r.get("conclusion"):
            valid_attributes.add(str(r["conclusion"]))

    results: Dict[str, Any] = {
        "total_rules": len(rules),
        "valid_rules": 0,
        "errors": [],
    }

    for r in rules:
        rule_id = r.get("id")
        family = r.get("family", "").lower()
        if not rule_id or not family:
            results["errors"].append(f"Rule missing ID or family: {r}")
            continue

        if family not in valid_families:
            results["errors"].append(f"Rule {rule_id} specifies unknown family: {family}")
            continue

        # Check AST condition variable names against ontology attributes
        cond = (
            str(r.get("condition", ""))
            .replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        try:
            parsed = ast.parse(cond, mode="eval")
            unknown_vars = [
                sub.id
                for sub in ast.walk(parsed)
                if isinstance(sub, ast.Name)
                and sub.id not in ("True", "False", "None")
                and sub.id not in valid_attributes
            ]
            if unknown_vars:
                results["errors"].append(
                    f"Rule {rule_id} references unknown attribute(s): {unknown_vars}"
                )
                continue
        except SyntaxError as e:
            results["errors"].append(f"Rule {rule_id} syntax error: {e}")
            continue

        results["valid_rules"] += 1

    return results


def get_rules_for_family(
    family: str,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Returns all rules targeting a given clause family (matching aliases transparently)."""
    all_rules = rules if rules is not None else load_rules_from_yaml()
    target_canon = normalize_family_name(family)
    return [r for r in all_rules if normalize_family_name(r.get("family", "")) == target_canon]


def get_rule_provenance(graph: nx.DiGraph, rule_id: str) -> Dict[str, Any]:
    """Queries the ontology graph for full provenance of a rule node."""
    if not graph.has_node(rule_id):
        raise KeyError(f"Rule node '{rule_id}' not found in ontology graph.")

    node_data = dict(graph.nodes[rule_id])
    norms: List[str] = []
    source_docs: List[str] = []
    applies_to: List[str] = []
    depends_on: List[str] = []
    relation_to_norm: Optional[str] = None

    for _, target, edge_data in graph.out_edges(rule_id, data=True):
        rel = edge_data.get("relation")
        if rel in ("deviatesFrom", "conformsTo"):
            norms.append(target)
            relation_to_norm = rel
        elif rel == "citedIn":
            source_docs.append(target)
        elif rel == "appliesTo":
            applies_to.append(target)
        elif rel == "dependsOn":
            depends_on.append(target)

    return {
        "rule_id": rule_id,
        "severity": node_data.get("severity"),
        "condition": node_data.get("condition"),
        "conclusion": node_data.get("conclusion"),
        "message": node_data.get("message"),
        "source": node_data.get("source"),
        "applies_to_families": applies_to,
        "market_norms": norms,
        "norm_relation": relation_to_norm,
        "source_docs": source_docs,
        "depends_on_rules": depends_on,
    }


def get_ontology_summary(graph: Optional[nx.DiGraph] = None) -> Dict[str, Any]:
    """Computes node and edge statistics for the ontology semantic network."""
    G = graph if graph is not None else build_ontology_graph()
    node_counts: Dict[str, int] = {}
    for _, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        node_counts[ntype] = node_counts.get(ntype, 0) + 1

    edge_counts: Dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "Unknown")
        edge_counts[rel] = edge_counts.get(rel, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_counts_by_type": node_counts,
        "edge_counts_by_relation": edge_counts,
    }


def evaluate_norm_calibration(
    synthetic_dir: Optional[Path] = None,
    weights: Optional[Dict[SeverityLevel, float]] = None,
) -> Dict[str, Any]:
    """Evaluates rule firing accuracy and risk-score calibration across synthetic gold sheets.

    Compares the forward-chaining engine's fired rules against the ground-truth
    `expected_flags` in `corpus/synthetic/*.gold.json` and measures the risk-score
    separation between standard market-norm term sheets and adversarial term sheets.
    """
    from termsheet_ai.reasoning.engine import ForwardChainingEngine, WorkingMemory

    if synthetic_dir is None:
        synthetic_dir = Path(__file__).resolve().parent.parent.parent / "corpus" / "synthetic"

    gold_files = sorted(synthetic_dir.glob("ts_synth_*.gold.json"))
    if not gold_files:
        raise FileNotFoundError(f"No synthetic gold JSON files found in {synthetic_dir}")

    effective_weights = weights or load_severity_weights_from_yaml()
    engine = ForwardChainingEngine(severity_weights=effective_weights)

    tp_total = 0
    fp_total = 0
    fn_total = 0
    exact_matches = 0
    standard_scores: List[float] = []
    adversarial_scores: List[float] = []
    per_rule_counts: Dict[str, Dict[str, int]] = {
        r.rule_id: {"standard": 0, "adversarial": 0, "total": 0} for r in engine.rules
    }

    for gfile in gold_files:
        gold = json.loads(gfile.read_text(encoding="utf-8"))
        doc_id = gold.get("doc_id", gfile.stem)
        is_adv = bool(gold.get("is_adversarial", False))
        expected_set = set(gold.get("expected_flags", []))

        wm = WorkingMemory()
        wm.assert_fact("multiple", gold["liq_multiple"], "c_lp")
        wm.assert_fact(
            "participation",
            gold["liq_participation"] in ("fully_participating", "capped_participating"),
            "c_lp",
        )
        wm.assert_fact("cap", gold.get("liq_cap"), "c_lp")
        wm.assert_fact("seniority", gold["liq_seniority"], "c_lp")
        wm.assert_fact("type", gold["anti_dilution_type"], "c_ad")
        wm.assert_fact("investor_seats", gold["board_investor_seats"], "c_bd")
        wm.assert_fact("founder_seats", gold["board_founder_seats"], "c_bd")
        wm.assert_fact("independent_seats", gold["board_independent_seats"], "c_bd")
        wm.assert_fact(
            "total_seats",
            gold["board_investor_seats"]
            + gold["board_founder_seats"]
            + gold["board_independent_seats"],
            "c_bd",
        )
        wm.assert_fact("vesting_months", gold["vesting_months"], "c_vs")
        wm.assert_fact("cliff_months", gold["cliff_months"], "c_vs")
        wm.assert_fact("acceleration", gold["acceleration"], "c_vs")
        wm.assert_fact("pool_percentage", gold["option_pool_pct"], "c_op")
        wm.assert_fact("duration_days", gold["exclusivity_days"], "c_ex")
        wm.assert_fact("approval_threshold", gold["drag_threshold_pct"], "c_da")
        wm.assert_fact("redemption_years", gold["redemption_years"], "c_rd")
        wm.assert_fact("cumulative", gold["dividends_cumulative"], "c_dv")

        trace = engine.infer(wm, doc_id=doc_id)
        fired_all = {rec.rule_id for rec in trace.rule_traces}
        fired_core = fired_all & CORE_GOLD_RULE_IDS

        tp = len(fired_core & expected_set)
        fp = len(fired_core - expected_set)
        fn = len(expected_set - fired_core)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        if fired_core == expected_set:
            exact_matches += 1

        score = trace.risk_score.overall_score
        if is_adv:
            adversarial_scores.append(score)
        else:
            standard_scores.append(score)

        for rid in fired_all:
            if rid in per_rule_counts:
                bucket = "adversarial" if is_adv else "standard"
                per_rule_counts[rid][bucket] += 1
                per_rule_counts[rid]["total"] += 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 1.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    std_mean = sum(standard_scores) / len(standard_scores) if standard_scores else 0.0
    adv_mean = sum(adversarial_scores) / len(adversarial_scores) if adversarial_scores else 0.0

    # Evaluate a canonical clean NVCA market-standard baseline sheet
    clean_wm = WorkingMemory()
    clean_wm.assert_fact("multiple", 1.0, "c_lp")
    clean_wm.assert_fact("participation", False, "c_lp")
    clean_wm.assert_fact("cap", None, "c_lp")
    clean_wm.assert_fact("seniority", "pari_passu", "c_lp")
    clean_wm.assert_fact("type", "broad_based_weighted_average", "c_ad")
    clean_wm.assert_fact("investor_seats", 1, "c_bd")
    clean_wm.assert_fact("founder_seats", 2, "c_bd")
    clean_wm.assert_fact("independent_seats", 1, "c_bd")
    clean_wm.assert_fact("total_seats", 4, "c_bd")
    clean_wm.assert_fact("vesting_months", 48, "c_vs")
    clean_wm.assert_fact("cliff_months", 12, "c_vs")
    clean_wm.assert_fact("acceleration", "double_trigger", "c_vs")
    clean_wm.assert_fact("pool_percentage", 15, "c_op")
    clean_wm.assert_fact("duration_days", 30, "c_ex")
    clean_wm.assert_fact("approval_threshold", 67, "c_da")
    clean_wm.assert_fact("redemption_years", 5, "c_rd")
    clean_wm.assert_fact("cumulative", False, "c_dv")
    clean_trace = engine.infer(clean_wm, doc_id="clean_nvca_baseline")
    clean_score = clean_trace.risk_score.overall_score

    return {
        "total_documents": len(gold_files),
        "standard_documents": len(standard_scores),
        "adversarial_documents": len(adversarial_scores),
        "core_flag_precision": precision,
        "core_flag_recall": recall,
        "core_flag_f1": f1,
        "exact_match_rate": exact_matches / len(gold_files),
        "clean_nvca_baseline_score": clean_score,
        "standard_mean_score": std_mean,
        "standard_min_score": min(standard_scores) if standard_scores else 0.0,
        "standard_max_score": max(standard_scores) if standard_scores else 0.0,
        "adversarial_mean_score": adv_mean,
        "adversarial_min_score": min(adversarial_scores) if adversarial_scores else 0.0,
        "adversarial_max_score": max(adversarial_scores) if adversarial_scores else 0.0,
        "score_separation_margin": adv_mean - std_mean,
        "baseline_to_adversarial_gap": adv_mean - clean_score,
        "per_rule_counts": per_rule_counts,
    }


def calibrate_severity_weights(
    synthetic_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Calibrates and compares candidate severity weight profiles on the synthetic corpus."""
    candidates: Dict[str, Dict[SeverityLevel, float]] = {
        "calibrated_production": {
            SeverityLevel.CRITICAL: 40.0,
            SeverityLevel.HIGH: 25.0,
            SeverityLevel.MEDIUM: 10.0,
            SeverityLevel.LOW: 5.0,
            SeverityLevel.INFO: 0.0,
        },
        "uncalibrated_uniform": {
            SeverityLevel.CRITICAL: 10.0,
            SeverityLevel.HIGH: 10.0,
            SeverityLevel.MEDIUM: 10.0,
            SeverityLevel.LOW: 10.0,
            SeverityLevel.INFO: 10.0,
        },
        "under_weighted": {
            SeverityLevel.CRITICAL: 15.0,
            SeverityLevel.HIGH: 10.0,
            SeverityLevel.MEDIUM: 8.0,
            SeverityLevel.LOW: 5.0,
            SeverityLevel.INFO: 2.0,
        },
    }

    profile_results: Dict[str, Dict[str, Any]] = {}
    best_profile = "calibrated_production"
    best_gap = -1.0

    for name, weights in candidates.items():
        metrics = evaluate_norm_calibration(synthetic_dir=synthetic_dir, weights=weights)
        gap = metrics["baseline_to_adversarial_gap"]
        profile_results[name] = {
            "weights": {k.value: v for k, v in weights.items()},
            "clean_nvca_baseline_score": metrics["clean_nvca_baseline_score"],
            "standard_mean_score": metrics["standard_mean_score"],
            "adversarial_mean_score": metrics["adversarial_mean_score"],
            "score_separation_margin": metrics["score_separation_margin"],
            "baseline_to_adversarial_gap": gap,
            "core_flag_f1": metrics["core_flag_f1"],
        }
        if gap > best_gap:
            best_gap = gap
            best_profile = name

    return {
        "selected_profile": best_profile,
        "selected_weights": profile_results[best_profile]["weights"],
        "profiles": profile_results,
    }
