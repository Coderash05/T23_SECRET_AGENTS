"""Streamlit rendering functions. Each takes a PipelinePayload (or view-model rows)."""

from html import escape
from typing import Iterable, List, Sequence

import streamlit as st

from termsheet_ai.schemas.contracts import PipelinePayload
from termsheet_ai.ui.theme import (
    NEUTRAL_COLOUR,
    SEVERITY_ORDER,
    chip_html,
    family_label,
    md_safe,
    score_band,
    sev_colour,
    sev_icon,
    sev_label,
    sev_value,
)
from termsheet_ai.ui.trace import all_trace_records, build_trace_tree, format_trace_tree
from termsheet_ai.ui.viewmodel import ClauseRow, clause_label, sorted_flags

FACT_FIELDS = (
    "liquidation_preference",
    "anti_dilution",
    "board_composition",
    "founder_vesting",
    "protective_provisions",
)


def render_header(payload: PipelinePayload) -> None:
    meta, trace = payload.metadata, payload.audit_trace
    left, right = st.columns([1, 2])
    with left:
        if trace is None:
            st.error("The reasoning stage did not produce a result. See the Debug tab.")
        else:
            score = trace.risk_score.overall_score
            label, colour = score_band(score)
            st.markdown(
                '<div class="ts-label">Overall risk score</div>'
                f'<div class="ts-score" style="color:{colour}">{score:.0f}<small> / 100</small></div>'
                f"<div>{escape(label)}</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(score / 100.0, 0.0), 1.0))
    with right:
        if trace is not None:
            chips = "".join(
                chip_html(
                    f"{getattr(trace.risk_score, sev + '_count')} {sev_label(sev)}",
                    sev_colour(sev) if getattr(trace.risk_score, sev + "_count") else NEUTRAL_COLOUR,
                )
                for sev in SEVERITY_ORDER
            )
            st.markdown(chips, unsafe_allow_html=True)
        st.caption(
            f"{meta.filename} ({meta.file_type.value.upper()}), "
            f"{len(payload.clauses)} clauses, "
            f"{sum(1 for c in payload.clauses if c.is_unclassified)} need manual review"
        )
        st.caption("Educational tool, not legal advice. Every flag below cites the rule that raised it.")


def render_clauses_tab(
    rows: Sequence[ClauseRow], severities: Iterable[str], only_flagged: bool
) -> None:
    selected = set(severities)
    shown = 0
    for row in rows:
        if only_flagged and not row.flags:
            continue
        if row.flags and not any(sev_value(f.severity) in selected for f in row.flags):
            continue
        shown += 1
        _render_clause_card(row)
    if shown == 0:
        st.info("No clauses match the current filters.")


def _render_clause_card(row: ClauseRow) -> None:
    clause = row.clause
    colour = sev_colour(row.worst_severity) if row.worst_severity else NEUTRAL_COLOUR
    header = escape(clause.section_header or clause.clause_id)
    chips = chip_html(family_label(clause.detected_family), outline=True)
    chips += chip_html(f"confidence {clause.classification_confidence:.0%}", outline=True)
    if clause.is_unclassified:
        chips += chip_html("needs manual review", "#A8741A")
    for flag in row.flags:
        chips += chip_html(sev_label(flag.severity), sev_colour(flag.severity))
    st.markdown(
        f'<div class="ts-card" style="border-left-color:{colour}"><b>{header}</b> &nbsp;{chips}</div>',
        unsafe_allow_html=True,
    )
    original, plain = st.columns(2)
    with original:
        st.markdown('<div class="ts-label">Original clause</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ts-clause">{escape(clause.raw_text)}</div>', unsafe_allow_html=True)
        st.caption(f"Page {clause.page_number}")
    with plain:
        st.markdown('<div class="ts-label">In plain English</div>', unsafe_allow_html=True)
        if clause.is_unclassified:
            st.warning("This clause could not be classified confidently, so it was not analysed.")
        if row.summary is None:
            st.caption("No summary available yet.")
        else:
            st.markdown(md_safe(row.summary.summary_text))
            if not row.summary.numbers_verified:
                st.caption("⚠️ Numbers in this summary were not verified against the extracted facts. Rely on the original clause.")
        for flag in row.flags:
            st.markdown(f"{sev_icon(flag.severity)} **{flag.title}**")


def render_flags_tab(payload: PipelinePayload) -> None:
    flags = sorted_flags(payload)
    if not flags:
        st.success("No risk flags were raised for this document.")
        return
    records = all_trace_records(payload.audit_trace)
    for flag in flags:
        label = f"{sev_icon(flag.severity)} {sev_label(flag.severity)}: {flag.title}"
        with st.expander(label):
            st.markdown(md_safe(flag.explanation))
            st.markdown(
                f"**Clause:** {clause_label(payload, flag.clause_id)}  \n"
                f"**Market standard:** {md_safe(flag.market_standard_source)}"
            )
            st.markdown("**Proof trace** (how the rule engine reached this flag)")
            tree = build_trace_tree(flag.justification, records)
            st.code("\n".join(format_trace_tree(tree)), language="text")


def render_facts_tab(payload: PipelinePayload) -> None:
    facts = payload.facts
    if facts is None:
        st.warning("No facts were extracted.")
        return
    for name in FACT_FIELDS:
        fact = getattr(facts, name)
        st.subheader(family_label(name))
        if fact is None:
            st.caption("Not found in this document.")
            continue
        data = fact.model_dump(mode="json", exclude={"source_span", "clause_id", "is_grounded"})
        st.dataframe(
            [{"attribute": k, "value": str(v)} for k, v in data.items()], hide_index=True
        )
        grounded = "verified against the clause text" if fact.is_grounded else "NOT grounded in the clause text"
        st.markdown(f"> {md_safe(fact.source_span.text)}")
        st.caption(f"Source quote, page {fact.source_span.page_number}, {grounded}.")
    if facts.custom_facts:
        st.subheader("Other facts")
        st.json(facts.custom_facts)


def render_investor_tab() -> None:
    st.info("Investor lookup (SEC EDGAR Form D) is a stretch feature planned for Week 4. This panel is a placeholder.")
    st.text_input("Investor or fund name", disabled=True, placeholder="e.g. Example Ventures LP")
    st.button("Search Form D records", disabled=True)


def render_debug_tab(payload: PipelinePayload) -> None:
    if payload.execution_errors:
        for err in payload.execution_errors:
            st.error(err)
    else:
        st.success("No pipeline errors.")
    st.json(payload.model_dump(mode="json"), expanded=False)
