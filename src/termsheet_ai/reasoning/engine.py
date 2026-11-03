"""Custom forward-chaining inference engine with justification records.

Implements forward chaining over definite Horn clauses loaded from rules_v1.yaml,
maintaining an agenda, working memory, and proof-tree justification records
as specified in the TermSheet AI KR&R architecture.
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

# Safe AST operator mappings
OPERATORS = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _safe_eval_node(node: ast.AST, context: Dict[str, Any]) -> Any:
    """Safely evaluates an AST node in the given variable context."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        if node.id.lower() in ("true", "false", "none"):
            return {"true": True, "false": False, "none": None}[node.id.lower()]
        return None
    elif isinstance(node, ast.Compare):
        left_val = _safe_eval_node(node.left, context)
        for op_type, right_node in zip(node.ops, node.comparators, strict=False):
            right_val = _safe_eval_node(right_node, context)
            if type(op_type) not in OPERATORS:
                return False
            if left_val is None or right_val is None:
                # Support `x == None` comparison
                if type(op_type) is ast.Eq:
                    return left_val is right_val
                elif type(op_type) is ast.NotEq:
                    return left_val is not right_val
                return False
            # Numeric conversion if needed
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                if not OPERATORS[type(op_type)](left_val, right_val):
                    return False
            elif type(left_val) is type(right_val):
                if not OPERATORS[type(op_type)](left_val, right_val):
                    return False
            else:
                return False
            left_val = right_val
        return True
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_safe_eval_node(v, context) for v in node.values)
        elif isinstance(node.op, ast.Or):
            return any(_safe_eval_node(v, context) for v in node.values)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _safe_eval_node(node.operand, context)

    return False


def safe_eval_condition(condition_str: str, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Evaluates a rule condition string safely, returning (matched, premise_variables)."""
    clean_str = (
        condition_str.replace("true", "True").replace("false", "False").replace("null", "None")
    )
    try:
        parsed = ast.parse(clean_str, mode="eval")
        result = bool(_safe_eval_node(parsed.body, context))

        used_vars = {}
        for sub in ast.walk(parsed):
            if isinstance(sub, ast.Name) and sub.id in context:
                used_vars[sub.id] = context[sub.id]

        return result, used_vars
    except Exception:
        return False, {}


class HornRule:
    """A first-order definite clause with antecedents and conclusion."""

    def __init__(
        self,
        rule_id: str,
        family: str,
        condition: str,
        severity: SeverityLevel,
        message: str,
        source: str,
    ):
        self.rule_id = rule_id
        self.family = family.lower()
        self.condition = condition
        self.severity = severity
        self.message = message
        self.source = source

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HornRule":
        raw_sev = str(data.get("severity", "MEDIUM")).lower()
        try:
            severity = SeverityLevel(raw_sev)
        except ValueError:
            severity = SeverityLevel.MEDIUM

        return cls(
            rule_id=data.get("id", "UNKNOWN_RULE"),
            family=data.get("family", "other"),
            condition=data.get("condition", "False"),
            severity=severity,
            message=data.get("message", "Rule triggered."),
            source=data.get("source", "NVCA Model Term Sheet"),
        )


class WorkingMemory:
    """Working memory holding asserted ground facts and derived facts."""

    def __init__(self):
        self._facts: Dict[str, Any] = {}
        self._clause_associations: Dict[str, str] = {}
        self._source_spans: Dict[str, str] = {}

    def assert_fact(
        self,
        key: str,
        value: Any,
        clause_id: Optional[str] = None,
        source_span: Optional[str] = None,
    ) -> None:
        """Asserts a ground fact into working memory."""
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

    def to_context(self, family: Optional[str] = None) -> Dict[str, Any]:
        """Provides flat context dictionary for condition evaluation."""
        context = dict(self._facts)
        if family:
            prefix = f"{family}."
            for k, v in self._facts.items():
                if k.startswith(prefix):
                    context[k[len(prefix) :]] = v
        return context

    def get_clause_for_key(self, key: str) -> Optional[str]:
        return self._clause_associations.get(key)


class ForwardChainingEngine(BaseReasoner):
    """The symbolic forward-chaining inference engine with justification traces."""

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
    ):
        if rules is not None:
            self.rules = rules
        else:
            raw_rules = load_rules_from_yaml(rules_yaml_path)
            self.rules = [HornRule.from_dict(r) for r in raw_rules]

    def build_working_memory_from_facts(self, facts: ExtractedFacts) -> WorkingMemory:
        """Translates Pydantic ExtractedFacts into working memory ground facts."""
        wm = WorkingMemory()

        # Liquidation Preference
        if facts.liquidation_preference:
            lp = facts.liquidation_preference
            cid = lp.clause_id
            span = lp.source_span.text if lp.source_span else ""
            wm.assert_fact("liquidation_preference.multiple", lp.multiple, cid, span)
            wm.assert_fact("multiple", lp.multiple, cid, span)

            is_part = lp.participation.value in ("fully_participating", "capped_participating")
            wm.assert_fact("liquidation_preference.participation", is_part, cid, span)
            wm.assert_fact("participation", is_part, cid, span)

            wm.assert_fact("liquidation_preference.cap", lp.cap_multiple, cid, span)
            wm.assert_fact("cap", lp.cap_multiple, cid, span)

            if lp.seniority:
                wm.assert_fact("liquidation_preference.seniority", lp.seniority, cid, span)
                wm.assert_fact("seniority", lp.seniority, cid, span)

        # Anti-Dilution
        if facts.anti_dilution:
            ad = facts.anti_dilution
            cid = ad.clause_id
            span = ad.source_span.text if ad.source_span else ""
            wm.assert_fact("anti_dilution.type", ad.anti_dilution_type.value, cid, span)
            wm.assert_fact("type", ad.anti_dilution_type.value, cid, span)

        # Board of Directors
        if facts.board_composition:
            bd = facts.board_composition
            cid = bd.clause_id
            span = bd.source_span.text if bd.source_span else ""
            wm.assert_fact("board_of_directors.founder_seats", bd.founder_seats, cid, span)
            wm.assert_fact("founder_seats", bd.founder_seats, cid, span)
            wm.assert_fact("board_of_directors.investor_seats", bd.investor_seats, cid, span)
            wm.assert_fact("investor_seats", bd.investor_seats, cid, span)
            wm.assert_fact("board_of_directors.independent_seats", bd.independent_seats, cid, span)
            wm.assert_fact("independent_seats", bd.independent_seats, cid, span)
            wm.assert_fact("board_of_directors.total_seats", bd.total_seats, cid, span)
            wm.assert_fact("total_seats", bd.total_seats, cid, span)

        # Vesting
        if facts.founder_vesting:
            fv = facts.founder_vesting
            cid = fv.clause_id
            span = fv.source_span.text if fv.source_span else ""
            months = int(fv.total_vesting_years * 12)
            wm.assert_fact("vesting.vesting_months", months, cid, span)
            wm.assert_fact("vesting_months", months, cid, span)
            wm.assert_fact("vesting.cliff_months", fv.cliff_months, cid, span)
            wm.assert_fact("cliff_months", fv.cliff_months, cid, span)
            wm.assert_fact("vesting.acceleration", fv.acceleration_type, cid, span)
            wm.assert_fact("acceleration", fv.acceleration_type, cid, span)

        # Custom / Extended facts
        for k, v in facts.custom_facts.items():
            wm.assert_fact(k, v)

        return wm

    def infer(self, wm: WorkingMemory, doc_id: str) -> AuditTrace:
        """Executes forward chaining over working memory using rules on the agenda."""
        agenda = list(self.rules)
        flags: List[RiskFlag] = []
        rule_traces: List[JustificationRecord] = []
        total_risk_score = 0.0

        for rule in agenda:
            context = wm.to_context(family=rule.family)
            matched, premises = safe_eval_condition(rule.condition, context)

            if matched:
                clause_id = "general"
                for prem_var in premises.keys():
                    cid = wm.get_clause_for_key(prem_var) or wm.get_clause_for_key(
                        f"{rule.family}.{prem_var}"
                    )
                    if cid:
                        clause_id = cid
                        break

                rec = JustificationRecord(
                    rule_id=rule.rule_id,
                    rule_name=rule.message,
                    market_norm_cite=rule.source,
                    premise_facts=premises,
                    conclusion_flag=f"Flag: {rule.message}",
                )

                flag = RiskFlag(
                    flag_id=f"flag_{rule.rule_id}",
                    clause_id=clause_id,
                    title=rule.message,
                    explanation=f"Evaluated condition '{rule.condition}' with ground premises: {premises}.",
                    severity=rule.severity,
                    market_standard_source=rule.source,
                    justification=rec,
                )

                flags.append(flag)
                rule_traces.append(rec)
                total_risk_score += self.SEVERITY_WEIGHTS.get(rule.severity, 0.0)

        risk_score = DocumentRiskScore(
            overall_score=min(100.0, total_risk_score),
            critical_count=sum(1 for f in flags if f.severity == SeverityLevel.CRITICAL),
            high_count=sum(1 for f in flags if f.severity == SeverityLevel.HIGH),
            medium_count=sum(1 for f in flags if f.severity == SeverityLevel.MEDIUM),
            low_count=sum(1 for f in flags if f.severity == SeverityLevel.LOW),
            info_count=sum(1 for f in flags if f.severity == SeverityLevel.INFO),
        )

        return AuditTrace(
            doc_id=doc_id,
            risk_score=risk_score,
            flags=flags,
            rule_traces=rule_traces,
        )

    def reason(self, facts: ExtractedFacts) -> AuditTrace:
        """BaseReasoner entrypoint."""
        wm = self.build_working_memory_from_facts(facts)
        return self.infer(wm, facts.doc_id)
