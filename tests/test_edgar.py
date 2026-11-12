"""Tests for the SEC EDGAR investor-lookup client.

All HTTP calls are mocked — no real network traffic is made.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from termsheet_ai.edgar.client import EdgarClient, EdgarResult

# ---------------------------------------------------------------------------
# Helpers — build realistic mock EDGAR responses
# ---------------------------------------------------------------------------


def _make_response(hits: list, status_code: int = 200) -> MagicMock:
    """Return a mock ``requests.Response`` with the given hits."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "hits": {
            "hits": hits,
            "total": {"value": len(hits)},
        }
    }
    return mock_resp


def _make_hit(entity_name: str, file_num: str = "000-12345") -> dict:
    return {"_source": {"entity_name": entity_name, "file_num": file_num}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEdgarClient:
    """Unit tests for EdgarClient — all HTTP is mocked."""

    # -- Lookup: single found result ----------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_lookup_found(self, mock_requests: MagicMock) -> None:
        """Single EDGAR hit → status='found' with match_name populated."""
        mock_requests.get.return_value = _make_response(
            [_make_hit("Sequoia Capital LLC", "000-56789")]
        )

        client = EdgarClient()
        result = client.lookup_investor("Sequoia Capital")

        assert isinstance(result, EdgarResult)
        assert result.status == "found"
        assert result.match_name == "Sequoia Capital LLC"
        assert result.cik == "000-56789"
        assert result.filings_count == 1
        assert result.filing_url is not None

    # -- Lookup: no hits → not_found -----------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_lookup_not_found(self, mock_requests: MagicMock) -> None:
        """Zero EDGAR hits → status='not_found'."""
        mock_requests.get.return_value = _make_response([])

        client = EdgarClient()
        result = client.lookup_investor("Nonexistent Fund XYZ")

        assert result.status == "not_found"
        assert result.match_name is None
        assert result.filings_count == 0

    @patch("termsheet_ai.edgar.client.requests")
    def test_lookup_rejects_empty_name(self, mock_requests: MagicMock) -> None:
        """An empty lookup must fail explicitly instead of querying EDGAR."""
        client = EdgarClient()

        with pytest.raises(ValueError, match="must not be empty"):
            client.lookup_investor("  ")

        mock_requests.get.assert_not_called()

    # -- Lookup: multiple hits → ambiguous ------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_lookup_ambiguous(self, mock_requests: MagicMock) -> None:
        """Multiple EDGAR hits → status='ambiguous'."""
        mock_requests.get.return_value = _make_response(
            [
                _make_hit("Accel Partners LLC"),
                _make_hit("Accel Partners Management LP"),
                _make_hit("Accel Partners IX LLC"),
            ]
        )

        client = EdgarClient()
        result = client.lookup_investor("Accel Partners")

        assert result.status == "ambiguous"
        assert result.filings_count == 3
        assert result.match_name == "Accel Partners LLC"  # top hit

    # -- HTTP error → graceful not_found --------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_lookup_http_error(self, mock_requests: MagicMock) -> None:
        """Network / HTTP failure → graceful not_found (no exception raised)."""
        mock_requests.get.side_effect = ConnectionError("Network unreachable")

        client = EdgarClient()
        result = client.lookup_investor("Benchmark Capital")

        assert result.status == "not_found"
        assert result.match_name is None

    # -- Caching ---------------------------------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_caching(self, mock_requests: MagicMock) -> None:
        """Second call with same name must NOT hit the network."""
        mock_requests.get.return_value = _make_response(
            [_make_hit("Greylock Partners LLC")]
        )

        client = EdgarClient()
        r1 = client.lookup_investor("Greylock Partners")
        r2 = client.lookup_investor("Greylock Partners")

        assert mock_requests.get.call_count == 1  # only 1 HTTP call
        assert r1 is r2  # same object from cache

    # -- Rate limiting ---------------------------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_rate_limiting(self, mock_requests: MagicMock) -> None:
        """Two rapid successive calls should take at least rate_limit_sec seconds."""
        mock_requests.get.return_value = _make_response([])

        rate = 0.1
        client = EdgarClient(rate_limit_sec=rate)

        start = time.monotonic()
        client.lookup_investor("Fund Alpha")
        client.lookup_investor("Fund Beta")
        elapsed = time.monotonic() - start

        # At least one inter-call gap must have occurred
        assert elapsed >= rate * 0.9  # allow 10% tolerance

    # -- Batch lookup ----------------------------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_batch_lookup(self, mock_requests: MagicMock) -> None:
        """lookup_batch returns a dict keyed by name for every queried name."""
        mock_requests.get.return_value = _make_response(
            [_make_hit("Khosla Ventures LLC")]
        )

        client = EdgarClient()
        names = ["Khosla Ventures", "Tiger Global"]
        results = client.lookup_batch(names)

        assert set(results.keys()) == set(names)
        assert all(isinstance(v, EdgarResult) for v in results.values())

    # -- Clear cache -----------------------------------------------------------

    @patch("termsheet_ai.edgar.client.requests")
    def test_clear_cache(self, mock_requests: MagicMock) -> None:
        """After clear_cache(), the next call fetches from the network again."""
        mock_requests.get.return_value = _make_response(
            [_make_hit("Index Ventures Management SA")]
        )

        client = EdgarClient()
        client.lookup_investor("Index Ventures")
        assert mock_requests.get.call_count == 1

        client.clear_cache()
        client.lookup_investor("Index Ventures")
        assert mock_requests.get.call_count == 2  # cache miss → new HTTP call
