"""Logging helpers that prevent credentials and personal data reaching logs."""
from __future__ import annotations

import logging
import re


_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_ENCODED_EMAIL_RE = re.compile(
    r"(?i)\b[A-Z0-9._+-]+%40[A-Z0-9.-]+(?:%2E|\.)[A-Z]{2,}\b"
)
_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:token|access_token|api_key|key|code|authuser|upload_id)=)[^&\s]+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_DOWNLOAD_URL_RE = re.compile(
    r"https://(?:lh3\.(?:googleusercontent|google)\.com|drum\.usercontent\.google\.com)/\S+",
    re.IGNORECASE,
)


def redact_log_text(value: object) -> str:
    """Return log text with emails, bearer tokens, and secret query values removed."""
    text = str(value)
    text = _DOWNLOAD_URL_RE.sub("[REDACTED_DOWNLOAD_URL]", text)
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _ENCODED_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _SECRET_QUERY_RE.sub(r"\1[REDACTED]", text)
    return _BEARER_RE.sub("Bearer [REDACTED]", text)


class RedactingFilter(logging.Filter):
    """Sanitize a LogRecord before any handler formats or persists it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_log_text(record.getMessage())
        record.args = ()
        return True


def install_redacting_filter(logger: logging.Logger | None = None) -> None:
    """Install the redaction filter on existing handlers."""
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(item, RedactingFilter) for item in handler.filters):
            handler.addFilter(RedactingFilter())
