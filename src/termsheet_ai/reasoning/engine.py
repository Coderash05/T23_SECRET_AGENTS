"""Custom forward-chaining inference engine with justification records.

Implements forward chaining over definite Horn clauses loaded from rules_v1.yaml,
maintaining an agenda, working memory, derived predicate chaining, and proof-tree
justification records as specified in the TermSheet AI KR&R architecture.
"""

import ast
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from termsheet_ai.ontology import (
    DEFAULT_SEVERITY_WEIGHTS,
    load_rules_from_yaml,
    load_severity_weights_from_yaml,
)
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
            # Numeric conversion if needed (excluding booleans which are int subclasses in Python)
            if (
                isinstance(left_val, (int, float))
                and not isinstance(left_val, bool)
                and isinstance(right_val, (int, float))
                and not isinstance(right_val, bool)
            ):
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
        conclusion: Optional[str] = None,
        norm_id: Optional[str] = None,
    ):
        self.rule_id = rule_id
        self.family = family.lower()
        self.condition = condition
        self.severity = severity
        self.message = message
        self.source = source
        self.conclusion = conclusion
        self.norm_id = norm_id

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
            conclusion=data.get("conclusion"),
            norm_id=data.get("norm_id"),
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
        """Asserts a ground or derived fact into working memory."""
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

    SEVERITY_WEIGHTS = dict(DEFAULT_SEVERITY_WEIGHTS)

    def __init__(
        self,
        rules: Optional[List[HornRule]] = None,
        rules_yaml_path: Optional[Path] = None,
        severity_weights: Optional[Dict[SeverityLevel, float]] = None,
    ):
        if rules is not None:
            self.rules = rules
        else:
            raw_rules = load_rules_from_yaml(rules_yaml_path)
            self.rules = [HornRule.from_dict(r) for r in raw_rules]

        if severity_weights is not None:
            self.severity_weights = dict(severity_weights)
        else:
            self.severity_weights = load_severity_weights_from_yaml(rules_yaml_path)

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
        vesting_cid: Optional[str] = None
        if facts.founder_vesting:
            fv = facts.founder_vesting
            cid = fv.clause_id
            vesting_cid = cid
            span = fv.source_span.text if fv.source_span else ""
            months = int(fv.total_vesting_years * 12)
            wm.assert_fact("vesting.vesting_months", months, cid, span)
            wm.assert_fact("vesting_months", months, cid, span)
            wm.assert_fact("vesting.cliff_months", fv.cliff_months, cid, span)
            wm.assert_fact("cliff_months", fv.cliff_months, cid, span)
            wm.assert_fact("vesting.acceleration", fv.acceleration_type, cid, span)
            wm.assert_fact("acceleration", fv.acceleration_type, cid, span)

            lower_span = span.lower()
            if any(tok in lower_span for tok in ("re-vest", "revest", "vesting reset")):
                wm.assert_fact("vesting.VestingReset", True, cid, span)
                wm.assert_fact("VestingReset", True, cid, span)

        # Protective Provisions
        if facts.protective_provisions:
            pp = facts.protective_provisions
            cid = pp.clause_id
            span = pp.source_span.text if pp.source_span else ""
            wm.assert_fact(
                "protective_provisions.requires_series_majority",
                pp.requires_series_majority,
                cid,
                span,
            )
            wm.assert_fact("requires_series_majority", pp.requires_series_majority, cid, span)
            lower_actions = " ".join(pp.covered_actions).lower() + " " + span.lower()
            if "day-to-day" in lower_actions or "day_to_day" in lower_actions:
                wm.assert_fact("protective_provisions.veto_day_to_day_operations", True, cid, span)
                wm.assert_fact("veto_day_to_day_operations", True, cid, span)

        # Custom / Extended facts
        for k, v in facts.custom_facts.items():
            wm.assert_fact(k, v)
            if k == "vesting_reset" and bool(v):
                wm.assert_fact("vesting.VestingReset", True, vesting_cid)
                wm.assert_fact("VestingReset", True, vesting_cid)
            elif k == "no_shop_days":
                wm.assert_fact("exclusivity.duration_days", v)
                wm.assert_fact("duration_days", v)

        return wm

    def infer(self, wm: WorkingMemory, doc_id: str) -> AuditTrace:
        """Executes multi-pass forward chaining over working memory until fixpoint."""
        flags: List[RiskFlag] = []
        rule_traces: List[JustificationRecord] = []
        total_risk_score = 0.0
        fired_rules: Set[str] = set()

        while True:
            new_fired = False
            for rule in self.rules:
                if rule.rule_id in fired_rules:
                    continue

                context = wm.to_context(family=rule.family)
                matched, premises = safe_eval_condition(rule.condition, context)

                if matched:
                    fired_rules.add(rule.rule_id)
                    new_fired = True

                    # Prefer associating the flag with the clause matching the rule's own family
                    clause_id = "general"
                    for prem_var in premises.keys():
                        cid = wm.get_clause_for_key(f"{rule.family}.{prem_var}")
                        if cid:
                            clause_id = cid
                            break
                    if clause_id == "general":
                        for prem_var in premises.keys():
                            cid = wm.get_clause_for_key(prem_var)
                            if cid:
                                clause_id = cid
                                break

                    conclusion_label = (
                        rule.conclusion if rule.conclusion else f"Flag: {rule.message}"
                    )
                    if rule.conclusion:
                        wm.assert_fact(rule.conclusion, True, clause_id=clause_id)
                        wm.assert_fact(
                            f"{rule.family}.{rule.conclusion}", True, clause_id=clause_id
                        )

                    rec = JustificationRecord(
                        rule_id=rule.rule_id,
                        rule_name=rule.message,
                        market_norm_cite=rule.source,
                        premise_facts=premises,
                        conclusion_flag=conclusion_label,
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
                    total_risk_score += self.severity_weights.get(rule.severity, 0.0)

            if not new_fired:
                break

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
