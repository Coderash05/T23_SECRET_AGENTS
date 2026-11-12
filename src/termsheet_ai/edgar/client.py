"""SEC EDGAR Form D investor-lookup client.

Searches SEC EDGAR for Form D filings by investor / fund name, returning a
structured result that the UI panel can surface alongside a term-sheet analysis.

Design constraints
~~~~~~~~~~~~~~~~~~
* **Rate limit**: EDGAR fair-access policy is 10 requests/second maximum.
  This client defaults to ~9 req/sec (0.11 s gap) with simple wall-clock
  rate-limiting.  All responses are cached for the lifetime of the client
  instance so repeated lookups for the same name are free.
* **No false positives**: A "not found" result is never reported as a risk flag.
  Many legitimate venture funds do not file Form D.  Only a confirmed "found"
  result surfaces a link.
* **Graceful degradation**: Any HTTP error, timeout, or JSON parse failure
  returns ``status="not_found"`` rather than raising.

Usage::

    from termsheet_ai.edgar.client import EdgarClient

    client = EdgarClient()
    result = client.lookup_investor("Sequoia Capital")
    print(result.status, result.filing_url)

    batch = client.lookup_batch(["Accel Partners", "Benchmark"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EDGAR API endpoints
# ---------------------------------------------------------------------------

_EDGAR_FULLTEXT_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%22{encoded_name}%22&forms=D&dateRange=custom&startdt=2000-01-01"
)
_EDGAR_COMPANY_SEARCH_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&company={encoded_name}&type=D&dateb=&owner=include&count=40"
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class EdgarResult:
    """Result of a single EDGAR Form D investor lookup.

    Attributes:
        query: The original investor name queried.
        status: One of ``"found"`` (single unambiguous match),
            ``"ambiguous"`` (multiple matches), or ``"not_found"`` (no match
            or lookup error).
        match_name: Entity name from the EDGAR filing (populated for
            ``"found"`` and optionally ``"ambiguous"``).
        filing_url: Link to the EDGAR company search page for this entity.
        cik: SEC file/CIK number if available.
        filings_count: Number of Form D filings found (0 for ``"not_found"``).
    """

    query: str
    status: str  # "found" | "not_found" | "ambiguous"
    match_name: Optional[str] = None
    filing_url: Optional[str] = None
    cik: Optional[str] = None
    filings_count: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class EdgarClient:
    """SEC EDGAR Form D investor-lookup client.

    Args:
        user_agent: HTTP ``User-Agent`` header sent with every request.
            EDGAR requires a descriptive agent string per their fair-access
            policy (https://www.sec.gov/developer).
        rate_limit_sec: Minimum seconds between consecutive HTTP requests.
            Defaults to 0.11 (≈ 9 req/sec), safely below the 10 req/sec cap.
    """

    def __init__(
        self,
        user_agent: str = "TermSheetAI/1.0 research@university.edu",
        rate_limit_sec: float = 0.11,
    ) -> None:
        self.user_agent = user_agent
        self.rate_limit_sec = rate_limit_sec
        self._last_call_time: float = 0.0
        self._cache: Dict[str, EdgarResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup_investor(self, name: str) -> EdgarResult:
        """Search EDGAR Form D filings for a single investor / fund name.

        Results are cached: a second call with the same name returns
        immediately without hitting the network.

        Args:
            name: Investor or fund name to search (e.g. "Sequoia Capital").

        Returns:
            An :class:`EdgarResult` with status ``"found"``, ``"ambiguous"``,
            or ``"not_found"``.
        """
        name = name.strip()
        if not name:
            raise ValueError("Investor name must not be empty.")
        if name in self._cache:
            logger.debug("EDGAR cache hit for '%s'", name)
            return self._cache[name]

        result = self._fetch(name)
        self._cache[name] = result
        return result

    def lookup_batch(self, names: List[str]) -> Dict[str, EdgarResult]:
        """Look up multiple investor names, respecting the rate limit.

        Args:
            names: List of investor / fund names.

        Returns:
            Dict mapping each name to its :class:`EdgarResult`.
        """
        return {name: self.lookup_investor(name) for name in names}

    def clear_cache(self) -> None:
        """Clear the in-memory response cache."""
        self._cache.clear()
        logger.debug("EDGAR cache cleared.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until the minimum inter-request gap has elapsed."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self.rate_limit_sec:
            time.sleep(self.rate_limit_sec - elapsed)

    def _fetch(self, name: str) -> EdgarResult:
        """Perform the actual HTTP request and parse the response."""
        self._rate_limit()

        encoded = quote(name)
        url = _EDGAR_FULLTEXT_URL.format(encoded_name=encoded)
        headers = {"User-Agent": self.user_agent}

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            self._last_call_time = time.monotonic()
            resp.raise_for_status()
            data = resp.json()
        except (RequestException, ConnectionError) as exc:
            logger.warning("EDGAR lookup failed for '%s': %s", name, exc)
            self._last_call_time = time.monotonic()
            return EdgarResult(query=name, status="not_found")
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("EDGAR response was invalid for '%s': %s", name, exc)
            self._last_call_time = time.monotonic()
            return EdgarResult(query=name, status="not_found")

        hits = data.get("hits", {}).get("hits", [])
        count = len(hits)

        filing_url = _EDGAR_COMPANY_SEARCH_URL.format(encoded_name=encoded)

        if count == 0:
            logger.debug("EDGAR: no Form D filings found for '%s'", name)
            return EdgarResult(query=name, status="not_found", filing_url=filing_url)

        if count == 1:
            src = hits[0].get("_source", {})
            match_name = src.get("entity_name") or src.get("period_of_report") or name
            cik = str(src.get("file_num", "") or src.get("cik", "") or "")
            logger.info("EDGAR: found 1 Form D match for '%s': %s", name, match_name)
            return EdgarResult(
                query=name,
                status="found",
                match_name=match_name,
                filing_url=filing_url,
                cik=cik or None,
                filings_count=1,
            )

        # Multiple hits — ambiguous
        top_name = hits[0].get("_source", {}).get("entity_name") or name
        logger.info(
            "EDGAR: %d Form D matches for '%s' (ambiguous); top: %s",
            count,
            name,
            top_name,
        )
        return EdgarResult(
            query=name,
            status="ambiguous",
            match_name=top_name,
            filing_url=filing_url,
            filings_count=count,
        )
