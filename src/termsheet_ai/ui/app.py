"""TermSheet AI Streamlit entry point.  Run:  streamlit run app.py  (repo root)."""

import streamlit as st

from termsheet_ai.ui import views
from termsheet_ai.ui.data_source import (
    build_pipeline,
    load_mock_payload,
    run_pipeline_on_upload,
    uses_plaintext_ingestor,
)
from termsheet_ai.ui.theme import CSS, SEVERITY_ORDER, sev_label
from termsheet_ai.ui.viewmodel import build_clause_rows

DEMO = "Demo result (mock data)"
UPLOAD = "Upload a term sheet"


def _sidebar():
    st.sidebar.header("Term sheet")
    mode = st.sidebar.radio("Source", [DEMO, UPLOAD], label_visibility="collapsed")
    payload = None
    if mode == DEMO:
        payload = load_mock_payload()
    else:
        upload = st.sidebar.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
        if (
            upload is not None
            and uses_plaintext_ingestor(build_pipeline())
            and not upload.name.lower().endswith(".txt")
        ):
            st.sidebar.warning(
                "The current ingestor reads plain text only. PDF/DOCX parsing arrives with the real S1 stage."
            )
        if upload is not None and st.sidebar.button("Analyse", type="primary"):
            with st.spinner("Running the pipeline..."):
                try:
                    st.session_state["payload"] = run_pipeline_on_upload(
                        upload.name, upload.getvalue()
                    )
                except Exception as exc:  # surface any pipeline failure instead of a stack trace
                    st.session_state.pop("payload", None)
                    st.sidebar.error(f"Pipeline failed: {exc}")
        payload = st.session_state.get("payload")

    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    severities = st.sidebar.multiselect(
        "Severities to show", SEVERITY_ORDER, default=list(SEVERITY_ORDER), format_func=sev_label
    )
    only_flagged = st.sidebar.checkbox("Only clauses with flags", value=False)
    return mode, payload, severities, only_flagged


def main() -> None:
    st.set_page_config(page_title="TermSheet AI", page_icon="📑", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("TermSheet AI")
    st.caption("Plain-English term sheet review with an auditable rule trace.")

    mode, payload, severities, only_flagged = _sidebar()
    if payload is None:
        st.info(
            "Upload a term sheet in the sidebar and press Analyse, or switch to the demo result."
        )
        return
    if mode == DEMO:
        st.info("You are viewing bundled mock data. Nothing here came from the live pipeline.")

    views.render_header(payload)
    clauses_tab, flags_tab, facts_tab, investor_tab, debug_tab = st.tabs(
        ["Clauses", "Flags and audit trace", "Extracted facts", "Investor lookup", "Debug"]
    )
    with clauses_tab:
        views.render_clauses_tab(build_clause_rows(payload), severities, only_flagged)
    with flags_tab:
        views.render_flags_tab(payload)
    with facts_tab:
        views.render_facts_tab(payload)
    with investor_tab:
        views.render_investor_tab()
    with debug_tab:
        views.render_debug_tab(payload)


if __name__ == "__main__":
    main()
