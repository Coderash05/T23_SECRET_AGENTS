"""Gemini REST API client for TermSheet AI (Stage S3).

This module provides a factory function ``make_gemini_callable`` that returns a
callable matching the ``LLMCallable`` protocol expected by ``FactExtractor``.

The key is loaded from the ``GEMINI_API_KEY`` environment variable (set via
``.env`` in the project root). It is **never** logged, printed, or embedded in
source code.

Usage
-----
>>> from termsheet_ai.extract.llm_client import make_gemini_callable
>>> call = make_gemini_callable()          # uses gemini-3.5-flash by default
>>> response_text = call("Your prompt here")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions for clear, actionable failure reporting
# ---------------------------------------------------------------------------


class GeminiAPIError(RuntimeError):
    """Base exception for Gemini API errors."""


class GeminiQuotaError(GeminiAPIError):
    """Raised when Gemini API quota or rate limit is reached (HTTP 429)."""


class GeminiModelNotFoundError(GeminiAPIError):
    """Raised when the specified Gemini model is not found or unsupported (HTTP 404)."""


class GeminiAuthenticationError(GeminiAPIError):
    """Raised when API key is missing, invalid, or unauthorized (HTTP 401/403)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_DEFAULT_MODEL = "gemini-3.5-flash"
_TIMEOUT_SECONDS = 30.0  # reasonable timeout for free-tier flash models


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_gemini_callable(
    model: Optional[str] = None,
    temperature: float = 0.0,
    *,
    api_key: Optional[str] = None,
    timeout: float = _TIMEOUT_SECONDS,
) -> "LLMCallable":  # noqa: F821  (avoid circular import; type hint only)
    """Return a callable ``(prompt: str) -> str`` backed by the Gemini REST API.

    Parameters
    ----------
    model:
        Gemini model name. Defaults to ``"gemini-3.5-flash"`` (or the
        ``GEMINI_MODEL`` environment variable). Only free-tier models should be used.
    temperature:
        Sampling temperature. Defaults to ``0.0`` for deterministic,
        structured extraction output.
    api_key:
        Optional override for the API key. When *None* (the default) the key
        is read from the ``GEMINI_API_KEY`` environment variable after loading
        ``.env``.
    timeout:
        Request timeout in seconds. Defaults to 30.0.

    Returns
    -------
    callable
        A function ``call(prompt: str) -> str`` that posts the prompt to
        Gemini and returns the model's text reply.

    Raises
    ------
    EnvironmentError
        If no API key can be resolved at call time.
    GeminiQuotaError
        If the Gemini API returns HTTP 429 (quota or rate-limit exceeded).
    GeminiModelNotFoundError
        If the model is deprecated or not found (HTTP 404).
    GeminiAuthenticationError
        If the API key is unauthorized or forbidden (HTTP 401/403).
    """
    load_dotenv()

    resolved_model = model or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL

    # _use_forced: when an explicit api_key is supplied (even ""), use it
    # directly and skip os.getenv() so tests can safely override.
    _use_forced: bool = api_key is not None
    _forced_key: str = api_key if api_key is not None else ""

    def _call(prompt: str) -> str:
        """Send *prompt* to Gemini and return the response text."""
        if _use_forced:
            key: str = _forced_key
        else:
            key = os.getenv("GEMINI_API_KEY") or ""

        if not key:
            raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

        url = f"{_API_BASE}/{resolved_model}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }

        try:
            response = httpx.post(
                url,
                params={"key": key},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            err_text = exc.response.text

            if status == 429:
                logger.error(
                    "Gemini API quota or rate limit reached (HTTP 429) for model %s: %s",
                    resolved_model,
                    err_text[:200],
                )
                raise GeminiQuotaError(
                    f"Gemini API quota or rate limit exceeded (HTTP 429) for model '{resolved_model}'. "
                    f"Free tier has 15 RPM / 1500 RPD. Check Google AI Studio (https://aistudio.google.com/) "
                    f"to view your quota and reset times."
                ) from exc

            if status == 404:
                logger.error(
                    "Gemini model %s not found or unsupported (HTTP 404): %s",
                    resolved_model,
                    err_text[:200],
                )
                raise GeminiModelNotFoundError(
                    f"Gemini model '{resolved_model}' not found or unsupported (HTTP 404). "
                    f"Verify model name against available free models."
                ) from exc

            if status in (401, 403):
                logger.error(
                    "Gemini API authentication failed (HTTP %s): %s",
                    status,
                    err_text[:200],
                )
                raise GeminiAuthenticationError(
                    f"Gemini API authentication failed (HTTP {status}). "
                    f"Check that your API key is valid in Google AI Studio."
                ) from exc

            logger.warning(
                "Gemini API HTTP error %s for model %s: %s",
                status,
                resolved_model,
                err_text[:200],
            )
            return ""

        except httpx.RequestError as exc:
            logger.warning(
                "Gemini API connection/request error for model %s: %s",
                resolved_model,
                exc,
            )
            return ""

        try:
            data = response.json()
            candidates = data.get("candidates")
            if not candidates:
                feedback = data.get("promptFeedback", {})
                logger.warning(
                    "Gemini API returned no candidates for model %s. Prompt feedback: %s",
                    resolved_model,
                    feedback,
                )
                return ""

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return parts[0].get("text", "")
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning(
                "Gemini API unexpected response structure for model %s: %s",
                resolved_model,
                exc,
            )
            return ""

    return _call
