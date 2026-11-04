"""Ontology and Knowledge Graph definition for TermSheet AI.

Builds a NetworkX directed graph encoding:
- Source Documents (NVCA Model Term Sheet, YC SAFE, Market Practice Studies)
- Clause Families (the 12 canonical families + other)
- Clause Attributes
- Market Norms and baseline values citing source documents
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import yaml

from termsheet_ai.schemas.contracts import ClauseFamily, SeverityLevel

CANONICAL_FAMILIES = [
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

MARKET_NORMS = {
    "liquidation_preference": {
        "standard_multiple": 1.0,
        "standard_participation": "non_participating",
        "market_source": "NVCA Model Term Sheet §3",
    },
    "anti_dilution": {
        "standard_type": "broad_based_weighted_average",
        "market_source": "NVCA Model Term Sheet §4",
    },
    "board_composition": {
        "standard_control": "founder_majority_or_parity",
        "market_source": "Market Practice Studies (Gompers & Lerner)",
    },
    "founder_vesting": {
        "standard_vesting_years": 4.0,
        "standard_cliff_months": 12,
        "standard_acceleration": "double_trigger",
        "market_source": "YC Post-Money SAFE / NVCA Model Term Sheet",
    },
    "option_pool": {
        "typical_range_min": 10.0,
        "typical_range_max": 20.0,
        "market_source": "Market Norm",
    },
    "exclusivity": {
        "max_standard_days": 45,
        "market_source": "Market Norm",
    },
    "dividends": {
        "standard_cumulative": False,
        "market_source": "NVCA Model Term Sheet",
    },
}


def get_canonical_clause_families() -> List[str]:
    """Returns the list of 12 canonical clause families."""
    return list(CANONICAL_FAMILIES)


def get_market_norms() -> Dict[str, Dict[str, Any]]:
    """Returns the baseline market norms dictionary."""
    return dict(MARKET_NORMS)


def build_ontology_graph() -> nx.DiGraph:
    """Constructs the comprehensive typed ontology semantic network for term sheet evaluation."""
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
        ("Market_Practice_Studies", "Empirical VC Contract Studies (Gompers, Lerner et al.)"),
    ]
    for doc_id, desc in source_docs:
        G.add_node(doc_id, type="SourceDoc", description=desc)

    # 2. Canonical Clause Families
    for fam in CANONICAL_FAMILIES:
        G.add_node(fam, type="ClauseFamily")
        G.add_edge("TermSheet", fam, relation="containsClause")

    # Family attributes
    family_attributes: Dict[str, List[str]] = {
        "liquidation_preference": ["multiple", "participation", "cap", "seniority"],
        "anti_dilution": ["type", "carveouts"],
        "board_of_directors": [
            "founder_seats",
            "investor_seats",
            "independent_seats",
            "total_seats",
        ],
        "protective_provisions": [
            "veto_day_to_day_operations",
            "requires_series_majority",
            "negative_covenants",
        ],
        "founder_vesting": ["vesting_months", "cliff_months", "acceleration"],
        "option_pool": ["pool_percentage", "pre_money_pool"],
        "exclusivity": ["duration_days", "fiduciary_out"],
        "conversion_rights": ["optional_conversion_ratio", "automatic_conversion_ipo_threshold"],
        "redemption_rights": ["redemption_years", "interest_rate"],
        "dividends": ["cumulative", "annual_rate"],
    }

    for fam, attributes in family_attributes.items():
        if not G.has_node(fam):
            G.add_node(fam, type="ClauseFamily")
        for attr in attributes:
            G.add_node(attr, type="Attribute")
            G.add_edge(fam, attr, relation="hasAttribute")

    # 3. Market Norms and Citations
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
            "Founders retain parity or majority in early-stage rounds",
            "board_of_directors",
            "Market_Practice_Studies",
        ),
        (
            "Standard_4_Year_1_Year_Cliff_Vesting",
            "48 months total vesting with 12 months cliff",
            "founder_vesting",
            "YC_SAFE",
        ),
        (
            "30_To_45_Day_No_Shop",
            "Exclusivity should not exceed 30-45 calendar days",
            "exclusivity",
            "Market_Practice_Studies",
        ),
        (
            "No_Cumulative_Dividends",
            "Dividends payable only when declared (non-cumulative)",
            "dividends",
            "NVCA_Model_Term_Sheet",
        ),
    ]

    for norm_id, description, family, source_doc in market_norms:
        G.add_node(norm_id, type="MarketNorm", description=description)
        G.add_edge(norm_id, family, relation="governs")
        G.add_edge(norm_id, source_doc, relation="citedIn")

    return G


def load_rules_from_yaml(yaml_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Loads and returns the list of rule dictionaries from rules YAML file."""
    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent.parent.parent / "docs" / "rules_v1.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Rules YAML not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = data.get("rules", [])
    return rules


def validate_rule_syntax(rules: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validates schema, fields, and condition syntax for a list of rule dictionaries."""
    errors: List[str] = []
    required_fields = {"id", "family", "condition", "severity", "message", "source"}

    for idx, rule in enumerate(rules, start=1):
        missing = required_fields - set(rule.keys())
        if missing:
            errors.append(f"Rule #{idx} ({rule.get('id', 'unknown')}) missing fields: {missing}")
            continue

        # Severity validation
        try:
            SeverityLevel(rule["severity"].lower())
        except ValueError:
            errors.append(f"Rule {rule['id']} invalid severity: {rule['severity']}")

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
            errors.append(f"Rule {rule['id']} condition syntax error: {e}")

    return len(errors) == 0, errors


def validate_rules_against_ontology(
    graph: nx.DiGraph, rules: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validates that loaded rules refer to valid families and ontology attributes."""
    valid_families = {
        n.lower() for n, d in graph.nodes(data=True) if d.get("type") == "ClauseFamily"
    }
    # Common aliases
    valid_families.update(
        {
            "board_of_directors",
            "board_composition",
            "liquidation_preference",
            "anti_dilution",
            "protective_provisions",
            "option_pool",
            "exclusivity",
            "redemption",
            "redemption_rights",
            "vesting",
            "founder_vesting",
            "dividends",
            "drag_along",
            "information_rights",
            "rofr_co_sale",
            "conversion_rights",
        }
    )

    results = {
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

        results["valid_rules"] += 1

    return results
