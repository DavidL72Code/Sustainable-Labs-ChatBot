"""Input and output sanitizers shared across the pipeline.

Currently the source-URL allowlist, which the retrieval path, the dashboard
summariser and document ingestion all call. It lives here rather than in any
one of them because all three are peers — putting it in the dashboard would
make retrieval import the dashboard.
"""
from __future__ import annotations

from urllib.parse import urlsplit


_ALLOWED_SOURCE_URL_SCHEMES = {"http", "https"}


def _is_allowed_source_url(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    parsed = urlsplit(candidate)
    return parsed.scheme.lower() in _ALLOWED_SOURCE_URL_SCHEMES and bool(parsed.netloc)


def _safe_source_url(value: str) -> str:
    candidate = str(value or "").strip()
    return candidate if _is_allowed_source_url(candidate) else "URL not provided"
