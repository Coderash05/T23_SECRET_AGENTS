"""Deterministic rule-based reasoning for TermSheet AI.

Supports existing YAML risk rules, optional derived facts, forward chaining,
working-memory provenance, and audit records.
"""

import ast
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from termsheet_ai.ontology import load_rules_from_yaml
from termsheet_ai.pipeline.base import BaseReasoner
from termsheet_ai.schemas.contracts import (
    AuditTrace,
    DocumentRiskScore,
    ExtractedFacts,
    JustificationRecord,
    RiskFlag,
    SeverityLevel,
)


_COMPARISONS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
}
_MISSING = object()
_LITERALS = {"true": True, "false": False, "null": None, "none": None}


def _safe_eval_node(node: ast.AST, context: Dict[str, Any]) -> Any:
    """Evaluate only a small, explicitly permitted subset of Python syntax."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        return _MISSING

    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        return _LITERALS.get(node.id.lower(), _MISSING)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_safe_eval_node(v, context) is True for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_safe_eval_node(v, context) is True for v in node.values)
        return False

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _safe_eval_node(node.operand, context)
        return False if value is _MISSING else not bool(value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left = _safe_eval_node(node.left, context)
        right = _safe_eval_node(node.right, context)
        if (type(left) not in (int, float)) or (type(right) not in (int, float)):
            return _MISSING
        return _ARITHMETIC[type(node.op)](left, right)

    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval_node(comparator, context)
            if left is _MISSING or right is _MISSING:
                return False
            fn = _COMPARISONS.get(type(op))
            if fn is None:
                return False
            if type(op) in (ast.Gt, ast.GtE, ast.Lt, ast.LtE):
                comparable = (
                    (type(left) in (int, float) and type(right) in (int, float))
                    or (type(left) is str and type(right) is str)
                )
                if not comparable:
                    return False
            elif type(left) is not type(right):
                if not (type(left) in (int, float) and type(right) in (int, float)):
                    return False
            try:
                if not fn(left, right):
                    return False
            except (TypeError, ValueError):
                return False
            left = right
        return True

    return _MISSING


def safe_eval_condition(
    condition_str: str, context: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Return whether a condition matches and the values of its referenced facts."""
    try:
        expression = ast.parse(condition_str, mode="eval")
        matched = _safe_eval_node(expression.body, context) is True
        premises = {
            name.id: context[name.id]
            for name in ast.walk(expression)
            if isinstance(name, ast.Name) and name.id in context
        }
        return matched, premises
    except (SyntaxError, TypeError, ValueError, ArithmeticError):
        return False, {}


class HornRule:
    """A conditional rule that may derive facts and/or raise a risk flag."""

    def __init__(
        self,
        rule_id: str,
        family: str,
        condition: str,
        severity: Optional[SeverityLevel],
        message: str,
        source: str,
        derives: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rule_id = rule_id
        self.family = family.lower()
        self.condition = condition
        self.severity = severity
        self.message = message
        self.source = source
        self.derives = dict(derives or {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HornRule":
        raw_severity = data.get("severity", "MEDIUM" if not data.get("derives") else None)
        if raw_severity is None:
            severity = None
        else:
            try:
                severity = SeverityLevel(str(raw_severity).lower())
            except ValueError:
                severity = SeverityLevel.MEDIUM
        return cls(
            rule_id=data.get("id", "UNKNOWN_RULE"),
            family=data.get("family", "other"),
            condition=data.get("condition", "False"),
            severity=severity,
            message=data.get("message", "Rule triggered."),
            source=data.get("source", "NVCA Model Term Sheet"),
            derives=data.get("derives", {}),
        )


class WorkingMemory:
    """Store asserted and inferred facts together with their clause provenance."""

    def __init__(self) -> None:
        self._facts: Dict[str, Any] = {}
        self._clause_associations: Dict[str, str] = {}
        self._source_spans: Dict[str, str] = {}
        self._derivations: Dict[str, JustificationRecord] = {}

    def assert_fact(
        self,
        key: str,
        value: Any,
        clause_id: Optional[str] = None,
        source_span: Optional[str] = None,
    ) -> None:
        self._facts[key] = value
        if clause_id:
            self._clause_associations[key] = clause_id
            if "." in key:
                self._clause_associations[key.split(".", 1)[1]] = clause_id
        if source_span:
            self._source_spans[key] = source_span
            if "." in key:
                self._source_spans[key.split(".", 1)[1]] = source_span

    def get(self, key: str, default: Any = None) -> Any:
        return self._facts.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._facts

    def to_context(self, family: Optional[str] = None) -> Dict[str, Any]:
        context = dict(self._facts)
        if family:
            prefix = f"{family}."
            for key, value in self._facts.items():
                if key.startswith(prefix):
                    context[key[len(prefix):]] = value
        return context

    def get_clause_for_key(self, key: str) -> Optional[str]:
        return self._clause_associations.get(key)

    def get_derivation(self, key: str) -> Optional[JustificationRecord]:
        return self._derivations.get(key)

    
    def build_fact_proof(
        self,
        fact_name: str,
        visited: Optional[set] = None,
    ) -> Dict[str, Any]:
        """Recursively explain how a fact was established."""

        if visited is None:
            visited = set()

        if fact_name in visited:
            return {
                "fact": fact_name,
                "cycle_detected": True,
            }

        visited = visited | {fact_name}

        value = self.get(fact_name)

        derivation = self.get_derivation(fact_name)

        # Base case: fact came directly from extraction.
        if derivation is None:
            return {
                "fact": fact_name,
                "value": value,
                "origin": "extracted",
            }

        # Recursive case: fact was produced by another rule.
        return {
            "fact": fact_name,
            "value": value,
            "origin": "derived",
            "rule_id": derivation.rule_id,
            "source": derivation.market_norm_cite,
            "premises": [
                self.build_fact_proof(name, visited)
                for name in derivation.premise_facts
            ],
        }


class ForwardChainingEngine(BaseReasoner):
    """Run rules to a fixed point, then return deterministic flags and traces."""

    SEVERITY_WEIGHTS = {
        SeverityLevel.CRITICAL: 40.0,
        SeverityLevel.HIGH: 25.0,
        SeverityLevel.MEDIUM: 10.0,
        SeverityLevel.LOW: 5.0,
        SeverityLevel.INFO: 0.0,
    }

    def __init__(
        self,
        rules: Optional[List[HornRule]] = None,
        rules_yaml_path: Optional[Path] = None,
    ) -> None:
        if rules is not None:
            self.rules = rules
        else:
            self.rules = [
                HornRule.from_dict(item)
                for item in load_rules_from_yaml(rules_yaml_path)
            ]

    def build_working_memory_from_facts(self, facts: ExtractedFacts) -> WorkingMemory:
        wm = WorkingMemory()

        if facts.liquidation_preference:
            lp = facts.liquidation_preference
            span = lp.source_span.text if lp.source_span else ""
            cid = lp.clause_id
            is_participating = lp.participation.value in (
                "fully_participating", "capped_participating"
            )
            fields = {
                "multiple": lp.multiple,
                "participation": is_participating,
                "cap": lp.cap_multiple,
                "seniority": lp.seniority,
            }
            for key, value in fields.items():
                if key == "seniority" and not value:
                    continue
                if key == "participation" and lp.participation.value == "unknown":
                    continue
                wm.assert_fact(f"liquidation_preference.{key}", value, cid, span)
                wm.assert_fact(key, value, cid, span)

        if facts.anti_dilution:
            ad = facts.anti_dilution
            span = ad.source_span.text if ad.source_span else ""
            value = ad.anti_dilution_type.value
            wm.assert_fact("anti_dilution.type", value, ad.clause_id, span)
            wm.assert_fact("type", value, ad.clause_id, span)

        if facts.board_composition:
            bd = facts.board_composition
            span = bd.source_span.text if bd.source_span else ""
            fields = {
                "founder_seats": bd.founder_seats,
                "investor_seats": bd.investor_seats,
                "independent_seats": bd.independent_seats,
                "total_seats": bd.total_seats,
            }
            for key, value in fields.items():
                wm.assert_fact(f"board_of_directors.{key}", value, bd.clause_id, span)
                wm.assert_fact(key, value, bd.clause_id, span)
            if bd.total_seats > 0:
                majority = bd.investor_seats * 2 > bd.total_seats
                wm.assert_fact("board_of_directors.investor_majority", majority, bd.clause_id, span)
                wm.assert_fact("investor_majority", majority, bd.clause_id, span)

        if facts.founder_vesting:
            fv = facts.founder_vesting
            span = fv.source_span.text if fv.source_span else ""
            fields = {
                "vesting_months": int(fv.total_vesting_years * 12),
                "cliff_months": fv.cliff_months,
                "acceleration": fv.acceleration_type,
            }
            for key, value in fields.items():
                wm.assert_fact(f"vesting.{key}", value, fv.clause_id, span)
                wm.assert_fact(key, value, fv.clause_id, span)

        if facts.protective_provisions:
            pp = facts.protective_provisions
            span = pp.source_span.text if pp.source_span else ""
            wm.assert_fact("protective_provisions.requires_series_majority", pp.requires_series_majority, pp.clause_id, span)
            wm.assert_fact("requires_series_majority", pp.requires_series_majority, pp.clause_id, span)

        for key, value in facts.custom_facts.items():
            wm.assert_fact(key, value)
        return wm

    def infer(self, wm: WorkingMemory, doc_id: str) -> AuditTrace:
        """Execute each rule once per run when its conditions become true.

        The repeated passes make execution independent of rule ordering. A
        conclusion is asserted only if it does not conflict with a known fact.
        """
        flags: List[RiskFlag] = []
        traces: List[JustificationRecord] = []
        fired_rules = set()
        total_risk_score = 0.0

        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.rule_id in fired_rules:
                    continue
                matched, premises = safe_eval_condition(
                    rule.condition, wm.to_context(family=rule.family)
                )
                if not matched:
                    continue

                fired_rules.add(rule.rule_id)
                clause_id = "general"
                for key in premises:
                    clause_id = (
                        wm.get_clause_for_key(f"{rule.family}.{key}")
                        or wm.get_clause_for_key(key)
                        or clause_id
                    )
                    if clause_id != "general":
                        break

                conclusion = (
                    ", ".join(f"{key}={value!r}" for key, value in rule.derives.items())
                    if rule.derives else f"Flag: {rule.message}"
                )
                record = JustificationRecord(
                    rule_id=rule.rule_id,
                    rule_name=rule.message,
                    market_norm_cite=rule.source,
                    premise_facts=premises,
                    conclusion_flag=conclusion,
                )
                traces.append(record)

                for key, value in rule.derives.items():
                    if not wm.has(key):
                        wm.assert_fact(key, value, clause_id=clause_id)
                        wm._derivations[key] = record
                        changed = True
                    elif wm.get(key) != value:
                        # Never silently overwrite contradictory ground facts.
                        continue

                if rule.severity is None:
                    continue

                # Build a complete explanation tree for this rule.
                record.proof_tree = {
                    "rule_id": rule.rule_id,
                    "conclusion": rule.message,
                    "source": rule.source,
                    "premises": [
                        wm.build_fact_proof(fact_name)
                        for fact_name in premises
                    ],
                }

                dependencies = [
                    wm.get_derivation(key).rule_id
                    for key in premises
                    if wm.get_derivation(key) is not None
                ]
                explanation = (
                    f"Condition '{rule.condition}' matched facts {premises}."
                )
                if dependencies:
                    explanation += f" Depends on rule(s): {', '.join(dict.fromkeys(dependencies))}."

                flags.append(
                    RiskFlag(
                        flag_id=f"flag_{rule.rule_id}",
                        clause_id=clause_id,
                        title=rule.message,
                        explanation=explanation,
                        severity=rule.severity,
                        market_standard_source=rule.source,
                        justification=record,
                    )
                )
                total_risk_score += self.SEVERITY_WEIGHTS.get(rule.severity, 0.0)

        score = DocumentRiskScore(
            overall_score=min(100.0, total_risk_score),
            critical_count=sum(f.severity == SeverityLevel.CRITICAL for f in flags),
            high_count=sum(f.severity == SeverityLevel.HIGH for f in flags),
            medium_count=sum(f.severity == SeverityLevel.MEDIUM for f in flags),
            low_count=sum(f.severity == SeverityLevel.LOW for f in flags),
            info_count=sum(f.severity == SeverityLevel.INFO for f in flags),
        )
        return AuditTrace(doc_id=doc_id, risk_score=score, flags=flags, rule_traces=traces)

    def reason(self, facts: ExtractedFacts) -> AuditTrace:
        """BaseReasoner entrypoint used by the full pipeline."""
        wm = self.build_working_memory_from_facts(facts)
        return self.infer(wm, facts.doc_id)