"""settings.py – Central configuration for NLM Audio Generator."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# ─── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
APP_VERSION = (
    (BASE_DIR / "VERSION").read_text(encoding="utf-8").strip()
    if (BASE_DIR / "VERSION").exists()
    else "0.0.0-dev"
)

RUNTIME_DIR   = DATA_DIR / "runtime"
ACCOUNTS_DIR  = DATA_DIR / "accounts"
PROJECTS_DIR  = DATA_DIR / "projects"
SHARED_GLOBAL_DIR   = DATA_DIR / "shared_sources" / "global"
SHARED_PROJECT_DIR  = DATA_DIR / "shared_sources" / "projects"
TOOLS_DIR     = DATA_DIR / "tools"
AUDIO_CONV_DIR      = TOOLS_DIR / "audio_conversion"
TRANSCRIPTIONS_DIR  = TOOLS_DIR / "transcriptions"

DB_PATH = DATA_DIR / "database.sqlite3"
BOOKLETS_DIR = DATA_DIR / "booklets"

RUNNER_PID_FILE = RUNTIME_DIR / "runner.pid"
RUNNER_LOG_FILE = RUNTIME_DIR / "runner.log"

# Create directories on import
for _d in (
    DATA_DIR, RUNTIME_DIR, ACCOUNTS_DIR, PROJECTS_DIR,
    SHARED_GLOBAL_DIR, SHARED_PROJECT_DIR,
    TOOLS_DIR, AUDIO_CONV_DIR, TRANSCRIPTIONS_DIR,
    BOOKLETS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ─── NotebookLM ────────────────────────────────────────────────────────────────
USE_FAKE_CLIENT: bool = False

# Browser for `notebooklm login` (chromium | msedge | chrome).
# On Windows, msedge avoids Playwright Chromium download (often blocked by region).
import sys as _sys
NOTEBOOKLM_LOGIN_BROWSER: str = "msedge" if _sys.platform == "win32" else "chromium"

# Default output language for audio generation (ISO 639-1 code)
# "fa" = Persian. Users can change this per-project or in Settings.
DEFAULT_AUDIO_LANGUAGE: str = "fa"

DEFAULT_AUDIO_PROMPT: str = (
    "لطفاً یک مرور صوتی جامع و کامل از این سند به زبان فارسی تهیه کن. "
    "مفاهیم کلیدی، نکات اصلی و جمع‌بندی مهم را به شکل واضح و جذاب بیان کن."
)
# Legacy alias
AUDIO_LANGUAGE: str = DEFAULT_AUDIO_LANGUAGE

SOURCE_WAIT_TIMEOUT: float = 300.0
AUDIO_WAIT_TIMEOUT: float  = 1200.0
# Extra buffer after every selected source is confirmed ready.  NotebookLM can
# briefly lag behind its source-ready response before audio generation sees all
# sources.
AUDIO_SOURCE_SETTLE_SECONDS: float = 60.0

# Daily notebook-creation limits on NotebookLM per account tier
ACCOUNT_DAILY_JOB_QUOTA: dict[str, int] = {
    "FREE":         3,
    "PAID":         20,
    "ORGANIZATION": 20,
    "UNKNOWN":      3,
}


def daily_job_quota_for(account_type) -> int:
    """Max notebooks/jobs per day for this NotebookLM account tier."""
    key = account_type.value if hasattr(account_type, "value") else str(account_type)
    return ACCOUNT_DAILY_JOB_QUOTA.get(key, 3)

# ─── Retry ─────────────────────────────────────────────────────────────────────
RETRY_DELAYS: list[int] = [10, 30, 60]

# ─── SQLite ────────────────────────────────────────────────────────────────────
DB_BUSY_TIMEOUT_MS: int = 30_000

# ─── FFmpeg ────────────────────────────────────────────────────────────────────
# Resolution order: FFMPEG_PATH env → PATH → common install locations.

def _find_winget_ffmpeg() -> str | None:
    """Locate ffmpeg.exe installed via winget (Gyan.FFmpeg)."""
    local_app = os.environ.get("LOCALAPPDATA", "")
    if not local_app:
        return None
    winget_root = Path(local_app) / "Microsoft" / "WinGet"
    for candidate in [
        winget_root / "Links" / "ffmpeg.exe",
        *winget_root.glob("Packages/Gyan.FFmpeg*/**/bin/ffmpeg.exe"),
    ]:
        if candidate.is_file():
            return str(candidate)
    return None


def get_ffmpeg_executable() -> str:
    """Return ffmpeg executable path. Raises FileNotFoundError if not found."""
    override = os.environ.get("FFMPEG_PATH", "").strip()
    if override and Path(override).is_file():
        return override
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Windows common locations (manual install, winget, scoop, chocolatey)
    for candidate in [
        _find_winget_ffmpeg(),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        str(Path.home() / "scoop" / "apps" / "ffmpeg" / "current" / "bin" / "ffmpeg.exe"),
        str(BASE_DIR / "ffmpeg" / "bin" / "ffmpeg.exe"),
    ]:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "FFmpeg not found. Install it and add to PATH, or set FFMPEG_PATH environment variable."
    )

DEFAULT_MP3_BITRATE: str = "128k"
KEEP_ORIGINAL_AUDIO: bool = True

# ─── Whisper ───────────────────────────────────────────────────────────────────
WHISPER_MODEL: str = "small"
WHISPER_LANGUAGE: str = ""        # empty = auto-detect
WHISPER_DEVICE: str = "auto"      # auto | cpu | cuda
WHISPER_COMPUTE_TYPE: str = "auto"  # auto | int8 | int8_float16 | float16 | float32
WHISPER_BEAM_SIZE: int = 5
WHISPER_VAD_FILTER: bool = True
MAX_CONCURRENT_TRANSCRIPTIONS: int = 1

# ─── UI ────────────────────────────────────────────────────────────────────────
UI_REFRESH_INTERVAL: int = 3

# ─── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"

# ─── AI Folder Processor ───────────────────────────────────────────────────────
# GapGPT API (OpenAI-compatible)
GAPGPT_BASE_URL: str = "https://api.gapgpt.app/v1"
GAPGPT_CDN_URL: str  = "https://api.gapapi.com/v1"
AI_DEFAULT_MODEL: str = "gpt-5.2"

# API key resolution order: env var → .env file → Streamlit session state
# Never hardcode API keys here.
GAPGPT_API_KEY_ENV_VAR: str = "GAPGPT_API_KEY"

# Concurrency
AI_MAX_CONCURRENCY: int = 3
AI_DEFAULT_TIMEOUT: int = 180   # seconds per request
AI_MAX_RETRIES: int = 3
AI_RETRY_DELAYS: list[int] = [10, 30, 60]  # exponential backoff steps

# Chunking defaults
AI_CHUNK_MAX_TOKENS: int = 6_000     # conservative default
AI_CHUNK_OVERLAP_TOKENS: int = 200
AI_CHARS_PER_TOKEN: float = 3.5      # conservative estimate for token counting

# XLSX extraction limits
AI_XLSX_MAX_ROWS: int = 5_000
AI_XLSX_MAX_COLS: int = 100

# ZIP extraction safety limits
AI_ZIP_MAX_FILES: int = 500
AI_ZIP_MAX_EXTRACT_MB: int = 200

# Intermediate storage (relative to output folder)
AI_INTERMEDIATE_DIR_NAME: str = ".intermediate"
AI_RESULTS_DIR_NAME: str = "_ai_results"

# File size limit per file (MB, 0 = no limit)
AI_DEFAULT_MAX_FILE_MB: int = 100

# Secret scanning
AI_SECRET_SCAN_ENABLED: bool = True

# ─── Word Booklet Maker (جزوه‌ساز Word) ───────────────────────────────────────
BOOKLET_DEFAULT_FONT_PERSIAN: str = "Vazirmatn"
BOOKLET_FALLBACK_FONT_PERSIAN: str = "B Nazanin"
BOOKLET_DEFAULT_FONT_ENGLISH: str = "Calibri"
BOOKLET_DEFAULT_FONT_HEADING: str = "Vazirmatn"
BOOKLET_DEFAULT_FONT_TABLE: str = "Calibri"
BOOKLET_DEFAULT_FONT_EMOJI: str = "Segoe UI Emoji"
BOOKLET_DEFAULT_FONT_CODE: str = "Courier New"

BOOKLET_BODY_FONT_SIZE_PT: int = 12
BOOKLET_H1_FONT_SIZE_PT: int = 20
BOOKLET_H2_FONT_SIZE_PT: int = 16
BOOKLET_H3_FONT_SIZE_PT: int = 14
BOOKLET_H4_FONT_SIZE_PT: int = 12
BOOKLET_LINE_SPACING: float = 1.15
BOOKLET_SPACE_AFTER_PT: int = 6

BOOKLET_MARGIN_TOP_CM: float = 2.0
BOOKLET_MARGIN_BOTTOM_CM: float = 2.0
BOOKLET_MARGIN_RIGHT_CM: float = 2.2
BOOKLET_MARGIN_LEFT_CM: float = 2.2

BOOKLET_HEADING_COLOR_HEX: str = "1F4E79"
BOOKLET_TABLE_HEADER_COLOR_HEX: str = "2E75B6"
BOOKLET_ACCENT_COLOR_HEX: str = "2E75B6"

BOOKLET_SCAN_EXTENSIONS: list[str] = [".txt", ".md", ".markdown"]
BOOKLET_IGNORED_PATTERNS: list[str] = [
    "*.metadata.json", "manifest.json", "*.log", "*.part", "*.tmp",
]
BOOKLET_IGNORED_DIRS: list[str] = [
    ".intermediate", ".cache", ".temp", "__pycache__",
]

# Blocked file patterns (default blocklist for secret files)
AI_BLOCKED_FILENAME_PATTERNS: list[str] = [
    ".env", ".env.*", "*.key", "*.pem", "*.pfx", "*.p12",
    "id_rsa", "id_ed25519", "credentials.json", "secrets.*",
    "*.sqlite", "*.sqlite3", "*.db",
]
