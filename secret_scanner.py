"""
secret_scanner.py – Detect sensitive files and secret patterns before sending
content to any external API.
"""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Blocked filename patterns ─────────────────────────────────────────────────

_BLOCKED_FILENAME_PATTERNS: list[str] = [
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.pfx",
    "*.p12",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "id_ecdsa",
    "id_dsa",
    "credentials.json",
    "secrets.*",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    "*.keystore",
    "*.jks",
    "*.pkcs12",
    "service_account.json",
    "gcloud_*.json",
    "aws_credentials",
]

# ── Secret patterns ───────────────────────────────────────────────────────────

@dataclass
class SecretPattern:
    name: str
    pattern: re.Pattern[str]
    description: str


_SECRET_PATTERNS: list[SecretPattern] = [
    SecretPattern(
        name="openai_api_key",
        pattern=re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
        description="OpenAI / GapGPT API key",
    ),
    SecretPattern(
        name="generic_api_key",
        pattern=re.compile(
            r'(?i)(?:api[_\-]?key|apikey|api[_\-]?secret)\s*[=:]\s*["\']?([A-Za-z0-9\-_]{16,})["\']?'
        ),
        description="Generic API key assignment",
    ),
    SecretPattern(
        name="bearer_token",
        pattern=re.compile(r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*", re.IGNORECASE),
        description="Bearer token",
    ),
    SecretPattern(
        name="aws_access_key",
        pattern=re.compile(r"AKIA[0-9A-Z]{16}"),
        description="AWS Access Key ID",
    ),
    SecretPattern(
        name="aws_secret_key",
        pattern=re.compile(
            r'(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'
        ),
        description="AWS Secret Access Key",
    ),
    SecretPattern(
        name="github_token",
        pattern=re.compile(r"ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82}"),
        description="GitHub personal access token",
    ),
    SecretPattern(
        name="private_key_block",
        pattern=re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        description="PEM private key block",
    ),
    SecretPattern(
        name="password_assignment",
        pattern=re.compile(
            r'(?i)(?:password|passwd|pwd)\s*[=:]\s*["\']([^"\']{6,})["\']'
        ),
        description="Password in common assignment form",
    ),
    SecretPattern(
        name="connection_string",
        pattern=re.compile(
            r'(?i)(?:mongodb|postgresql|mysql|redis|amqp)://[^\s"\'<>]+'
        ),
        description="Database / service connection string with credentials",
    ),
    SecretPattern(
        name="google_api_key",
        pattern=re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
        description="Google API key",
    ),
    SecretPattern(
        name="slack_token",
        pattern=re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}"),
        description="Slack token",
    ),
    SecretPattern(
        name="stripe_key",
        pattern=re.compile(r"(?:sk|pk)_(?:live|test)_[0-9A-Za-z]{24,}"),
        description="Stripe API key",
    ),
]


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class SecretMatch:
    pattern_name: str
    description: str
    line_number: int
    excerpt: str          # short excerpt, secret value masked


@dataclass
class FileScanResult:
    path: Path
    is_blocked_by_name: bool
    blocked_pattern: Optional[str]        # which filename pattern matched
    secret_matches: list[SecretMatch] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.is_blocked_by_name and not self.secret_matches

    @property
    def has_secret_matches(self) -> bool:
        return bool(self.secret_matches)

    @property
    def risk_summary(self) -> str:
        if self.is_blocked_by_name:
            return f"Blocked: filename pattern '{self.blocked_pattern}'"
        if self.secret_matches:
            names = ", ".join(m.pattern_name for m in self.secret_matches[:3])
            return f"Suspicious: {names}"
        return "Safe"


# ── Public API ────────────────────────────────────────────────────────────────

def is_filename_blocked(filename: str, extra_patterns: Optional[list[str]] = None) -> tuple[bool, Optional[str]]:
    """Check if a filename matches any blocked pattern.

    Returns (is_blocked, matched_pattern).
    """
    name_lower = filename.lower()
    patterns = _BLOCKED_FILENAME_PATTERNS + (extra_patterns or [])
    for pat in patterns:
        if fnmatch.fnmatch(name_lower, pat.lower()):
            return True, pat
        # also match the stem-only part for patterns without extension
        if fnmatch.fnmatch(name_lower, pat.lower() + ".*"):
            return True, pat
    return False, None


def scan_text_for_secrets(text: str, max_lines: int = 500) -> list[SecretMatch]:
    """Scan up to max_lines of text for secret patterns.

    Returns list of SecretMatch (with masked excerpts).
    """
    matches: list[SecretMatch] = []
    lines = text.splitlines()[:max_lines]
    for lineno, line in enumerate(lines, 1):
        for sp in _SECRET_PATTERNS:
            m = sp.pattern.search(line)
            if m:
                # mask the matched value
                excerpt = line.strip()[:120]
                masked = sp.pattern.sub(lambda _: "***REDACTED***", excerpt)
                matches.append(SecretMatch(
                    pattern_name=sp.name,
                    description=sp.description,
                    line_number=lineno,
                    excerpt=masked,
                ))
                break  # one match per line is enough
    return matches


def scan_file(
    path: Path,
    scan_content: bool = True,
    max_content_bytes: int = 50_000,
    extra_blocked_patterns: Optional[list[str]] = None,
) -> FileScanResult:
    """Full scan: filename blocklist + content secret patterns.

    Content scan is limited to the first max_content_bytes bytes to avoid
    reading huge files.
    """
    blocked, matched_pat = is_filename_blocked(path.name, extra_blocked_patterns)
    result = FileScanResult(path=path, is_blocked_by_name=blocked, blocked_pattern=matched_pat)

    if blocked or not scan_content:
        return result

    try:
        raw = path.read_bytes()
        # Only scan text-like files
        if _looks_binary(raw[:1024]):
            return result
        text = raw[:max_content_bytes].decode("utf-8", errors="replace")
        result.secret_matches = scan_text_for_secrets(text)
    except (OSError, PermissionError):
        pass  # can't read → treat as no secret matches

    return result


def mask_api_key(key: str) -> str:
    """Return a masked version: sk-****Ab12 style."""
    if not key:
        return ""
    prefix = key[:3] if len(key) >= 3 else key
    suffix = key[-4:] if len(key) >= 8 else ""
    return f"{prefix}-****{suffix}"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _looks_binary(sample: bytes) -> bool:
    """Return True if the sample looks like binary data (not text)."""
    if not sample:
        return False
    # Heuristic: >30% non-printable bytes → binary
    non_printable = sum(1 for b in sample if b < 9 or (13 < b < 32) or b == 127)
    return non_printable / len(sample) > 0.30
