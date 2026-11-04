"""Severity palette, labels and CSS shared by all UI views (no Streamlit import)."""

from html import escape
from typing import Any, Tuple

SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")

SEVERITY_STYLE = {
    "critical": {"label": "Critical", "colour": "#8E1B2E", "icon": "🔴"},
    "high": {"label": "High", "colour": "#C4491C", "icon": "🟠"},
    "medium": {"label": "Medium", "colour": "#A8741A", "icon": "🟡"},
    "low": {"label": "Low", "colour": "#2F7D57", "icon": "🟢"},
    "info": {"label": "Info", "colour": "#4F6D8A", "icon": "🔵"},
}

NEUTRAL_COLOUR = "#7A7F87"


def sev_value(severity: Any) -> str:
    """Accept a SeverityLevel enum or a plain string and return the lowercase value."""
    return str(getattr(severity, "value", severity)).lower()


def sev_rank(severity: Any) -> int:
    """Lower is worse. Unknown severities sort last."""
    value = sev_value(severity)
    return SEVERITY_ORDER.index(value) if value in SEVERITY_ORDER else len(SEVERITY_ORDER)


def sev_colour(severity: Any) -> str:
    return SEVERITY_STYLE.get(sev_value(severity), {}).get("colour", NEUTRAL_COLOUR)


def sev_label(severity: Any) -> str:
    return SEVERITY_STYLE.get(sev_value(severity), {}).get("label", sev_value(severity).title())


def sev_icon(severity: Any) -> str:
    return SEVERITY_STYLE.get(sev_value(severity), {}).get("icon", "⚪")


def score_band(score: float) -> Tuple[str, str]:
    """Map the 0-100 document score to a label and colour (UI-only thresholds)."""
    if score < 25:
        return "Low risk", SEVERITY_STYLE["low"]["colour"]
    if score < 50:
        return "Moderate risk", SEVERITY_STYLE["medium"]["colour"]
    if score < 75:
        return "High risk", SEVERITY_STYLE["high"]["colour"]
    return "Severe risk", SEVERITY_STYLE["critical"]["colour"]


def family_label(family: Any) -> str:
    return str(getattr(family, "value", family)).replace("_", " ").title()


def chip_html(text: str, colour: str = NEUTRAL_COLOUR, outline: bool = False) -> str:
    cls = "ts-chip ts-chip-outline" if outline else "ts-chip"
    style = "" if outline else f' style="background:{colour}"'
    return f'<span class="{cls}"{style}>{escape(text)}</span>'


def md_safe(text: str) -> str:
    """Stop '$' from being parsed as LaTeX in st.markdown."""
    return text.replace("$", "\\$")


CSS = """
<style>
.ts-card{border-left:5px solid #7A7F87;background:rgba(128,128,128,.07);
  padding:.6rem .9rem;border-radius:4px;margin:.6rem 0 .4rem}
.ts-card b{font-size:1.02rem}
.ts-chip{display:inline-block;padding:.08rem .6rem;border-radius:999px;font-size:.78rem;
  font-weight:600;color:#fff;margin-right:.35rem}
.ts-chip-outline{background:transparent;border:1px solid rgba(128,128,128,.7);
  color:inherit;font-weight:500}
.ts-clause{font-family:Georgia,'Times New Roman',serif;line-height:1.55;white-space:pre-wrap;
  border-left:2px solid rgba(128,128,128,.4);padding-left:.8rem}
.ts-label{font-size:.85rem;opacity:.7;margin-bottom:.25rem}
.ts-score{font-size:3.2rem;font-weight:700;line-height:1}
.ts-score small{font-size:1.1rem;font-weight:400;opacity:.6}
</style>
"""
