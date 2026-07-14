"""
ai_folder_service.py – Folder scanning and file discovery for the AI Folder Processor.

Responsibilities:
- Validate and resolve folder paths (Path Traversal guard)
- Scan recursively or flat with filters (extension, size, hidden, symlinks)
- Classify files into AIFileGroup
- Determine extraction method per file
- Return structured list of DiscoveredFile entries
"""
from __future__ import annotations

import fnmatch
import hashlib
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models import AIFileGroup
from secret_scanner import FileScanResult, is_filename_blocked, scan_file
from settings import (
    AI_BLOCKED_FILENAME_PATTERNS,
    AI_DEFAULT_MAX_FILE_MB,
    AI_SECRET_SCAN_ENABLED,
)

# ── File group / extension maps ───────────────────────────────────────────────

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".xml", ".yaml", ".yml",
    ".ini", ".cfg", ".toml", ".html", ".htm", ".sql",
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cs", ".cpp", ".c", ".h", ".hpp",
    ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".dart",
    ".sh", ".ps1", ".bat",
})

PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

OFFICE_EXTENSIONS: frozenset[str] = frozenset({
    ".docx", ".pptx", ".xlsx", ".xls", ".doc", ".ppt",
    ".odt", ".ods", ".odp",
})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif",
})

AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus",
})

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv",
})

ARCHIVE_EXTENSIONS: frozenset[str] = frozenset({".zip"})

# Map extension → (AIFileGroup, extraction_method)
_EXT_MAP: dict[str, tuple[AIFileGroup, str]] = {}
for _e in TEXT_EXTENSIONS:
    _EXT_MAP[_e] = (AIFileGroup.TEXT, "direct_text")
for _e in PDF_EXTENSIONS:
    _EXT_MAP[_e] = (AIFileGroup.PDF, "pymupdf")
for _e in OFFICE_EXTENSIONS:
    if _e in (".docx", ".doc"):
        _EXT_MAP[_e] = (AIFileGroup.OFFICE, "python_docx")
    elif _e in (".pptx", ".ppt"):
        _EXT_MAP[_e] = (AIFileGroup.OFFICE, "python_pptx")
    elif _e in (".xlsx", ".xls"):
        _EXT_MAP[_e] = (AIFileGroup.OFFICE, "openpyxl")
    else:
        _EXT_MAP[_e] = (AIFileGroup.OFFICE, "unsupported")
for _e in IMAGE_EXTENSIONS:
    _EXT_MAP[_e] = (AIFileGroup.IMAGE, "vision_or_skip")
for _e in AUDIO_EXTENSIONS:
    _EXT_MAP[_e] = (AIFileGroup.AUDIO, "whisper")
for _e in VIDEO_EXTENSIONS:
    _EXT_MAP[_e] = (AIFileGroup.VIDEO, "whisper_ffmpeg")
for _e in ARCHIVE_EXTENSIONS:
    _EXT_MAP[_e] = (AIFileGroup.ARCHIVE, "zip_extract")


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class DiscoveredFile:
    absolute_path: Path
    relative_path: Path           # relative to scan root
    filename: str
    extension: str
    mime_type: str
    file_size: int                # bytes
    file_group: AIFileGroup
    extraction_method: str
    enabled: bool = True          # user can toggle in UI
    skip_reason: Optional[str] = None
    scan_result: Optional[FileScanResult] = None

    @property
    def file_size_kb(self) -> float:
        return self.file_size / 1024

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    @property
    def is_supported(self) -> bool:
        return self.extraction_method not in ("unsupported", "unknown")

    @property
    def status_label(self) -> str:
        if self.skip_reason:
            return "Skipped"
        if self.scan_result and not self.scan_result.is_safe:
            return "Suspicious"
        return "Ready" if self.is_supported else "Unsupported"


@dataclass
class FolderScanConfig:
    root: Path
    recursive: bool = True
    include_hidden: bool = False
    follow_symlinks: bool = False
    allowed_extensions: Optional[list[str]] = None   # None = all supported
    blocked_extensions: Optional[list[str]] = None
    max_file_mb: float = float(AI_DEFAULT_MAX_FILE_MB)
    scan_secrets: bool = AI_SECRET_SCAN_ENABLED
    extra_text_extensions: Optional[list[str]] = None  # user-added extensions


@dataclass
class FolderScanResult:
    root: Path
    files: list[DiscoveredFile] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total(self) -> int:
        return len(self.files)

    @property
    def supported(self) -> list[DiscoveredFile]:
        return [f for f in self.files if f.is_supported and not f.skip_reason]

    @property
    def skipped(self) -> list[DiscoveredFile]:
        return [f for f in self.files if f.skip_reason]

    @property
    def suspicious(self) -> list[DiscoveredFile]:
        return [
            f for f in self.files
            if f.scan_result and not f.scan_result.is_safe and not f.skip_reason
        ]

    @property
    def needs_whisper(self) -> int:
        return sum(
            1 for f in self.supported
            if f.file_group in (AIFileGroup.AUDIO, AIFileGroup.VIDEO)
        )

    @property
    def needs_vision(self) -> int:
        return sum(1 for f in self.supported if f.file_group == AIFileGroup.IMAGE)

    @property
    def total_size_mb(self) -> float:
        return sum(f.file_size for f in self.files) / (1024 * 1024)

    @property
    def supported_size_mb(self) -> float:
        return sum(f.file_size for f in self.supported) / (1024 * 1024)


# ── Validation ────────────────────────────────────────────────────────────────

class FolderValidationError(Exception):
    """Raised when folder path is invalid or unsafe."""
    def __init__(self, message: str, error_code: str = "INVALID_FOLDER"):
        super().__init__(message)
        self.error_code = error_code


def validate_folder(path_str: str) -> Path:
    """Resolve and validate folder path. Raises FolderValidationError on failure."""
    if not path_str or not path_str.strip():
        raise FolderValidationError("Folder path is empty.", "INVALID_FOLDER")

    try:
        p = Path(path_str.strip()).resolve()
    except Exception as exc:
        raise FolderValidationError(f"Invalid path: {exc}", "INVALID_FOLDER") from exc

    if not p.exists():
        raise FolderValidationError(f"Folder not found: {p}", "FOLDER_NOT_FOUND")

    if not p.is_dir():
        raise FolderValidationError(f"Path is not a folder: {p}", "INVALID_FOLDER")

    if not os.access(p, os.R_OK):
        raise FolderValidationError(f"Read access denied: {p}", "PERMISSION_DENIED")

    return p


def _is_path_traversal(root: Path, candidate: Path) -> bool:
    """Return True if candidate escapes root (path traversal)."""
    try:
        candidate.resolve().relative_to(root.resolve())
        return False
    except ValueError:
        return True


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_folder(config: FolderScanConfig) -> FolderScanResult:
    """Scan root folder and return discovered files according to config."""
    result = FolderScanResult(root=config.root)

    try:
        root_resolved = config.root.resolve()
    except Exception as exc:
        result.error = str(exc)
        return result

    extra_text = frozenset(
        (e if e.startswith(".") else f".{e}").lower()
        for e in (config.extra_text_extensions or [])
    )
    allowed = (
        frozenset(
            (e if e.startswith(".") else f".{e}").lower()
            for e in config.allowed_extensions
        )
        if config.allowed_extensions is not None
        else None
    )
    blocked = frozenset(
        (e if e.startswith(".") else f".{e}").lower()
        for e in (config.blocked_extensions or [])
    )

    iterator = root_resolved.rglob("*") if config.recursive else root_resolved.iterdir()

    for entry in iterator:
        # Skip symlinks first (before is_file so we never follow them)
        if entry.is_symlink():
            continue

        if not entry.is_file():
            continue

        # Path traversal guard
        if _is_path_traversal(root_resolved, entry):
            continue

        # Hidden files
        if not config.include_hidden and _is_hidden(entry):
            continue

        relative = entry.relative_to(root_resolved)
        filename = entry.name
        ext = entry.suffix.lower()
        mime_type = mimetypes.guess_type(str(entry))[0] or ""
        file_size = 0
        skip_reason: Optional[str] = None

        try:
            file_size = entry.stat().st_size
        except OSError:
            skip_reason = "Failed to read file metadata"

        # Extension filters
        if ext in blocked:
            skip_reason = f"Blocked extension: {ext}"
        elif allowed is not None and ext not in allowed and ext not in extra_text:
            skip_reason = f"Extension not allowed: {ext}"

        # Size filter
        if not skip_reason and config.max_file_mb > 0:
            size_mb = file_size / (1024 * 1024)
            if size_mb > config.max_file_mb:
                skip_reason = f"File too large: {size_mb:.1f} MB"

        # Determine group & method
        effective_ext = ext
        if ext in extra_text:
            group, method = AIFileGroup.TEXT, "direct_text"
        elif effective_ext in _EXT_MAP:
            group, method = _EXT_MAP[effective_ext]
        else:
            group, method = AIFileGroup.UNKNOWN, "unsupported"

        # Secret scan (filename first)
        scan_result: Optional[FileScanResult] = None
        if config.scan_secrets:
            is_blocked_name, matched_pat = is_filename_blocked(filename, AI_BLOCKED_FILENAME_PATTERNS)
            if is_blocked_name:
                scan_result = FileScanResult(
                    path=entry,
                    is_blocked_by_name=True,
                    blocked_pattern=matched_pat,
                )
                if not skip_reason:
                    skip_reason = f"Sensitive file: {matched_pat}"
            elif group == AIFileGroup.TEXT and not skip_reason:
                scan_result = scan_file(entry)

        df = DiscoveredFile(
            absolute_path=entry,
            relative_path=relative,
            filename=filename,
            extension=ext,
            mime_type=mime_type,
            file_size=file_size,
            file_group=group,
            extraction_method=method if not skip_reason else "skipped",
            enabled=not bool(skip_reason) and method != "unsupported",
            skip_reason=skip_reason,
            scan_result=scan_result,
        )
        result.files.append(df)

    # Sort: supported first, then by relative path
    result.files.sort(key=lambda f: (bool(f.skip_reason), str(f.relative_path)))
    return result


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hex digest of a file. Returns '' on error."""
    h = hashlib.new(algorithm)
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_hidden(path: Path) -> bool:
    """Return True if any component of the path is hidden (starts with '.')."""
    return any(part.startswith(".") for part in path.parts if part not in (".", ".."))
