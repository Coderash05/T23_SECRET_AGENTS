"""Turn flat JustificationRecords into a proof tree (no Streamlit import).

The contract has no explicit "derived from" link, so chaining is inferred by name:
if a record's ``premise_facts`` contains a key equal to another record's
``conclusion_flag`` (e.g. R-CT-05 needs ``InvestorControl``, which R-CT-02 concludes),
that other record becomes a child node. Everything else is a ground fact.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from termsheet_ai.schemas.contracts import AuditTrace, JustificationRecord


@dataclass
class TraceNode:
    rule_id: str
    rule_name: str
    cite: str
    conclusion: str
    ground_facts: Dict[str, Any] = field(default_factory=dict)
    children: List["TraceNode"] = field(default_factory=list)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)


def all_trace_records(trace: Optional[AuditTrace]) -> List[JustificationRecord]:
    """rule_traces plus every flag's justification, de-duplicated by rule_id."""
    if trace is None:
        return []
    seen, records = set(), []
    for rec in [*trace.rule_traces, *(f.justification for f in trace.flags)]:
        if rec.rule_id not in seen:
            seen.add(rec.rule_id)
            records.append(rec)
    return records


def build_trace_tree(
    record: JustificationRecord,
    records: Iterable[JustificationRecord],
    _visited: FrozenSet[str] = frozenset(),
) -> TraceNode:
    records = list(records)
    visited = _visited | {record.rule_id}
    node = TraceNode(
        rule_id=record.rule_id,
        rule_name=record.rule_name,
        cite=record.market_norm_cite,
        conclusion=record.conclusion_flag,
    )
    for key, value in record.premise_facts.items():
        parent = next(
            (r for r in records if r.conclusion_flag == key and r.rule_id not in visited),
            None,
        )
        if parent is not None:
            node.children.append(build_trace_tree(parent, records, visited))
        else:
            node.ground_facts[key] = value
    return node


def _body_lines(node: TraceNode, prefix: str, lines: List[str]) -> None:
    entries = [("norm", f"market norm: {node.cite}", None)]
    entries += [("fact", f"{k} = {v}", None) for k, v in node.ground_facts.items()]
    entries += [("rule", "", child) for child in node.children]
    for i, (kind, text, child) in enumerate(entries):
        last = i == len(entries) - 1
        branch = "└─ " if last else "├─ "
        extension = "   " if last else "│  "
        if kind == "rule":
            lines.append(f"{prefix}{branch}derived by {_headline(child)}")
            _body_lines(child, prefix + extension, lines)
        else:
            lines.append(f"{prefix}{branch}{text}")


def _headline(node: TraceNode) -> str:
    return f"{node.rule_id} ({node.rule_name})  =>  {node.conclusion}"


def format_trace_tree(node: TraceNode) -> List[str]:
    """Plain-text proof tree, one string per line."""
    lines = [_headline(node)]
    _body_lines(node, "", lines)
    return lines
