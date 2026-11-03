"""File-based LLM response cache for TermSheet AI.

The cache stores ``{sha256_hex: response_text}`` entries as JSON in a single
file (default: ``.llm_cache.json`` in the project root, git-ignored).

Motivation
----------
* Gemini free-tier allows 15 requests per minute. Caching prevents
  redundant network calls during development and repeated test runs.
* The cache is keyed on the **exact prompt text** (SHA-256), so even a single
  character change produces a different hash and triggers a fresh call.

Usage
-----
>>> from termsheet_ai.extract.llm_client import make_gemini_callable
>>> from termsheet_ai.extract.llm_cache import make_cached_callable
>>> raw = make_gemini_callable()
>>> call = make_cached_callable(raw)           # wraps with caching
>>> response = call("Some prompt …")           # cached after first hit
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias (mirrors LLMCallable to avoid circular import)
# ---------------------------------------------------------------------------
_LLMCallable = Callable[[str], str]

# ---------------------------------------------------------------------------
# Default cache path: <project_root>/.llm_cache.json
# ---------------------------------------------------------------------------
# parents[0] = extract/, parents[1] = termsheet_ai/, parents[2] = src/, parents[3] = repo root
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[3] / ".llm_cache.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_prompt(prompt: str) -> str:
    """Return the SHA-256 hex digest of *prompt* encoded as UTF-8."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, str]:
    """Load the cache dict from *cache_path*, returning ``{}`` on any error."""
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        logger.warning("Cache file %s has unexpected format; resetting.", cache_path)
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read cache %s: %s", cache_path, exc)
        return {}


def _save_cache(cache: dict[str, str], cache_path: Path) -> None:
    """Persist *cache* to *cache_path*, silently skipping on write errors."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.warning("Could not write cache %s: %s", cache_path, exc)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_cached_callable(
    inner: _LLMCallable,
    cache_path: Optional[Path] = None,
) -> _LLMCallable:
    """Wrap *inner* with a file-based prompt cache.

    Parameters
    ----------
    inner:
        The underlying LLM callable to wrap, e.g. the one returned by
        ``make_gemini_callable()``.
    cache_path:
        Path to the JSON cache file. Defaults to ``.llm_cache.json`` in the
        project root.

    Returns
    -------
    callable
        A function ``call(prompt: str) -> str`` that returns the cached
        response when available and otherwise delegates to *inner*, storing
        the new response before returning it.
    """
    resolved_path = cache_path if cache_path is not None else _DEFAULT_CACHE_PATH

    def _call(prompt: str) -> str:
        cache = _load_cache(resolved_path)
        key = _hash_prompt(prompt)

        if key in cache:
            logger.debug("Cache hit for prompt hash %s", key[:12])
            return cache[key]

        logger.debug("Cache miss for prompt hash %s — calling LLM", key[:12])
        response = inner(prompt)

        # Only cache non-empty responses so transient errors do not permanently
        # suppress future successful calls.
        if response:
            cache[key] = response
            _save_cache(cache, resolved_path)

        return response

    return _call
