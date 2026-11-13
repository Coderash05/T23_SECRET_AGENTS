"""Single-prompt LLM-only baseline for TermSheet AI evaluation.

Unlike the production pipeline, this baseline performs no retrieval, structured
fact extraction, source-span checking, or symbolic reasoning.  It exists only as
a controlled comparison point and should not be used for founder-facing advice.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from termsheet_ai.extract.llm_cache import make_cached_callable
from termsheet_ai.extract.llm_client import make_gemini_callable
from termsheet_ai.schemas.contracts import ClauseFamily

LLMCallable = Callable[[str], str]

RISK_FLAG_IDS = (
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
)

_PROMPT = """You are an LLM-only baseline for a venture-capital term-sheet evaluation.
Analyze the text directly. Do not use external tools, retrieval, rules, or hidden assumptions.
Return ONLY one JSON object using exactly this schema:
{
  "clause_family": one of {families},
  "risk_flags": [zero or more IDs from {flags}],
  "summary": "a concise plain-English explanation"
}

Use "other" if the text does not fit a listed family. A risk flag may be included only if
the text itself supports it. Text follows:
---
{text}
---"""


@dataclass(frozen=True)
class LLMOnlyPrediction:
    """Normalized output of one direct LLM call."""

    clause_family: ClauseFamily
    risk_flags: tuple[str, ...]
    summary: str
    raw_response: str

    def to_dict(self) -> dict[str, object]:
        return {
            "clause_family": self.clause_family.value,
            "risk_flags": list(self.risk_flags),
            "summary": self.summary,
            "raw_response": self.raw_response,
        }


def _parse_json_object(response: str) -> Optional[dict[str, object]]:
    """Parse a direct or fenced JSON object without trying to repair LLM output."""
    text = response.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class LLMOnlyBaseline:
    """Direct-prompt comparator, deliberately independent of production stages."""

    def __init__(self, llm_callable: LLMCallable):
        self.llm_callable = llm_callable

    @classmethod
    def with_gemini(
        cls, model: Optional[str] = None, api_key: Optional[str] = None, use_cache: bool = True
    ) -> "LLMOnlyBaseline":
        """Create the baseline with the project's existing Gemini client and prompt cache."""
        llm = make_gemini_callable(model=model, api_key=api_key, temperature=0.0)
        return cls(make_cached_callable(llm) if use_cache else llm)

    def predict(self, text: str) -> LLMOnlyPrediction:
        """Run one LLM call and normalize only the declared JSON fields.

        Invalid or empty model output is represented as an ``other`` prediction
        with no flags, making failures visible in evaluation rather than silently
        substituting any production-pipeline logic.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        prompt = _PROMPT.format(
            families=json.dumps([family.value for family in ClauseFamily]),
            flags=json.dumps(RISK_FLAG_IDS),
            text=text.strip(),
        )
        response = self.llm_callable(prompt)
        raw_response = response if isinstance(response, str) else ""
        payload = _parse_json_object(raw_response) or {}
        try:
            family = ClauseFamily(str(payload.get("clause_family", "other")).lower())
        except ValueError:
            family = ClauseFamily.OTHER
        supplied_flags = payload.get("risk_flags", [])
        if not isinstance(supplied_flags, list):
            supplied_flags = []
        flags = tuple(sorted({str(flag) for flag in supplied_flags if str(flag) in RISK_FLAG_IDS}))
        summary = payload.get("summary", "")
        return LLMOnlyPrediction(
            clause_family=family,
            risk_flags=flags,
            summary=summary.strip() if isinstance(summary, str) else "",
            raw_response=raw_response,
        )
