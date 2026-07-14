"""
booklet_sort_service.py – File scanning, metadata extraction and sorting
for the Word Booklet Maker (جزوه‌ساز Word).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from settings import BOOKLET_IGNORED_DIRS, BOOKLET_IGNORED_PATTERNS, BOOKLET_SCAN_EXTENSIONS


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileEntry:
    source_path: Path
    relative_path: str
    filename: str
    extension: str
    file_size: int
    created_time: Optional[datetime]
    modified_time: datetime
    created_time_source: str          # "birthtime" | "ctime" | "mtime_fallback"
    extracted_number: Optional[int]
    detected_title: str
    heading_count: int
    table_count: int
    word_count: int
    encoding: str
    file_hash: str
    readable: bool
    error_message: str
    # mutable fields set by UI
    sort_order: int = 0
    enabled: bool = True
    custom_title: str = ""

    @property
    def effective_title(self) -> str:
        return self.custom_title.strip() or self.detected_title or self.filename


# ══════════════════════════════════════════════════════════════════════════════
# Sort modes
# ══════════════════════════════════════════════════════════════════════════════

SORT_MODE_LABELS = {
    "MANUAL":          "Manual order",
    "NAME_ASC":        "Filename (A–Z)",
    "NAME_DESC":       "Filename (Z–A)",
    "NATURAL":         "Natural filename sort",
    "CREATED_ASC":     "Created date (oldest first)",
    "CREATED_DESC":    "Created date (newest first)",
    "MODIFIED_ASC":    "Modified date (oldest first)",
    "MODIFIED_DESC":   "Modified date (newest first)",
    "PATH_AND_NAME":   "Folder path, then filename",
    "EXTRACTED_NUM":   "Number extracted from filename",
}


# ══════════════════════════════════════════════════════════════════════════════
# Natural sort key
# ══════════════════════════════════════════════════════════════════════════════

def _natural_key(s: str) -> list:
    parts = re.split(r"(\d+)", s.lower())
    result = []
    for p in parts:
        if p.isdigit():
            result.append(int(p))
        else:
            result.append(p)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Creation time helper (cross-platform)
# ══════════════════════════════════════════════════════════════════════════════

def _get_times(path: Path) -> Tuple[Optional[datetime], datetime, str]:
    """Return (created_time, modified_time, created_time_source).

    created_time_source is one of:
      "birthtime"      – real birth time (macOS / some Windows NTFS)
      "ctime"          – st_ctime (Windows: usually creation; Linux: inode change)
      "mtime_fallback" – no reliable creation time; using mtime
    """
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime)

    if sys.platform == "win32":
        # On Windows st_ctime is creation time for NTFS
        ctime_ts = stat.st_ctime
        if ctime_ts > 0:
            return datetime.fromtimestamp(ctime_ts), mtime, "ctime"
        return None, mtime, "mtime_fallback"

    if sys.platform == "darwin":
        birthtime_ts = getattr(stat, "st_birthtime", None)
        if birthtime_ts:
            return datetime.fromtimestamp(birthtime_ts), mtime, "birthtime"

    # Linux: st_ctime is inode-change time, not birth time
    return None, mtime, "mtime_fallback"


# ══════════════════════════════════════════════════════════════════════════════
# File hash
# ══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# Text analysis helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_encoding(path: Path) -> Tuple[str, str]:
    """Return (encoding, content).  Falls back to latin-1 on failure."""
    for enc in ("utf-8-sig", "utf-8", "windows-1256", "iso-8859-1", "latin-1"):
        try:
            content = path.read_text(encoding=enc)
            return enc, content
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1", path.read_text(encoding="latin-1", errors="replace")


_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_JSON_TABLE_RE = re.compile(r'"table"\s*:', re.IGNORECASE)


def _analyse_content(content: str) -> Tuple[str, int, int, int]:
    """Return (detected_title, heading_count, table_count, word_count)."""
    m = _H1_RE.search(content)
    detected_title = ""
    if m:
        raw = m.group(1).strip()
        # Strip leading emoji/markdown bold markers
        raw = re.sub(r"^[^\w\u0600-\u06FF]+", "", raw, flags=re.UNICODE)
        detected_title = raw.strip()[:200]

    heading_count = len(_HEADING_RE.findall(content))

    # Count JSON table candidates (quick heuristic; full parse happens in parser)
    table_count = len(_JSON_TABLE_RE.findall(content))

    word_count = len(content.split())
    return detected_title, heading_count, table_count, word_count


def _extract_number(filename: str, pattern: str = r"(\d+)") -> Optional[int]:
    try:
        m = re.search(pattern, filename)
        if m:
            return int(m.group(1))
    except (re.error, ValueError):
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Ignore helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_ignored_file(filename: str, patterns: list[str] = BOOKLET_IGNORED_PATTERNS) -> bool:
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def _is_ignored_dir(dirname: str, dirs: list[str] = BOOKLET_IGNORED_DIRS) -> bool:
    return dirname in dirs


# ══════════════════════════════════════════════════════════════════════════════
# Core scan
# ══════════════════════════════════════════════════════════════════════════════

def scan_folder(
    folder: Path,
    extensions: list[str] | None = None,
    recursive: bool = True,
    ignored_patterns: list[str] = BOOKLET_IGNORED_PATTERNS,
    ignored_dirs: list[str] = BOOKLET_IGNORED_DIRS,
    number_regex: str = r"(\d+)",
) -> list[FileEntry]:
    """Scan *folder* and return a list of FileEntry objects (unsorted)."""
    if extensions is None:
        extensions = list(BOOKLET_SCAN_EXTENSIONS)

    ext_set = {e.lower() for e in extensions}
    entries: list[FileEntry] = []

    if not folder.is_dir():
        return entries

    if recursive:
        all_files: list[Path] = []
        for root, dirs, files in os.walk(folder):
            # Filter ignored dirs in-place so os.walk skips them
            dirs[:] = [d for d in dirs if not _is_ignored_dir(d, ignored_dirs)]
            for fname in files:
                all_files.append(Path(root) / fname)
    else:
        all_files = list(folder.iterdir())

    for fpath in all_files:
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in ext_set:
            continue
        if _is_ignored_file(fpath.name, ignored_patterns):
            continue

        created_time, modified_time, created_time_source = _get_times(fpath)
        file_size = fpath.stat().st_size
        file_hash = _sha256(fpath)

        try:
            relative = str(fpath.relative_to(folder))
        except ValueError:
            relative = fpath.name

        readable = True
        error_message = ""
        detected_title = ""
        heading_count = 0
        table_count = 0
        word_count = 0
        encoding = "unknown"

        if file_size == 0:
            readable = False
            error_message = "File is empty"
        else:
            try:
                encoding, content = _detect_encoding(fpath)
                detected_title, heading_count, table_count, word_count = _analyse_content(content)
            except Exception as exc:
                readable = False
                error_message = str(exc)

        extracted_number = _extract_number(fpath.stem, number_regex)

        entries.append(FileEntry(
            source_path=fpath,
            relative_path=relative,
            filename=fpath.name,
            extension=fpath.suffix.lower(),
            file_size=file_size,
            created_time=created_time,
            modified_time=modified_time,
            created_time_source=created_time_source,
            extracted_number=extracted_number,
            detected_title=detected_title,
            heading_count=heading_count,
            table_count=table_count,
            word_count=word_count,
            encoding=encoding,
            file_hash=file_hash,
            readable=readable,
            error_message=error_message,
        ))

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Sorting
# ══════════════════════════════════════════════════════════════════════════════

def sort_entries(
    entries: list[FileEntry],
    mode: str = "NATURAL",
) -> list[FileEntry]:
    """Sort entries in-place according to *mode* and return them."""

    if mode == "MANUAL":
        entries.sort(key=lambda e: (e.sort_order, e.filename))

    elif mode == "NAME_ASC":
        entries.sort(key=lambda e: e.filename.lower())

    elif mode == "NAME_DESC":
        entries.sort(key=lambda e: e.filename.lower(), reverse=True)

    elif mode == "NATURAL":
        entries.sort(key=lambda e: _natural_key(e.filename))

    elif mode == "CREATED_ASC":
        entries.sort(key=lambda e: (e.created_time or e.modified_time))

    elif mode == "CREATED_DESC":
        entries.sort(key=lambda e: (e.created_time or e.modified_time), reverse=True)

    elif mode == "MODIFIED_ASC":
        entries.sort(key=lambda e: e.modified_time)

    elif mode == "MODIFIED_DESC":
        entries.sort(key=lambda e: e.modified_time, reverse=True)

    elif mode == "PATH_AND_NAME":
        entries.sort(key=lambda e: (_natural_key(e.relative_path), _natural_key(e.filename)))

    elif mode == "EXTRACTED_NUM":
        entries.sort(key=lambda e: (
            0 if e.extracted_number is not None else 1,
            e.extracted_number if e.extracted_number is not None else 0,
            _natural_key(e.filename),
        ))

    # Re-assign sort_order to reflect final positions
    for i, entry in enumerate(entries):
        entry.sort_order = i + 1

    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Manual reorder helpers
# ══════════════════════════════════════════════════════════════════════════════

def move_entry_up(entries: list[FileEntry], index: int) -> list[FileEntry]:
    if index <= 0 or index >= len(entries):
        return entries
    entries[index - 1], entries[index] = entries[index], entries[index - 1]
    _reassign_orders(entries)
    return entries


def move_entry_down(entries: list[FileEntry], index: int) -> list[FileEntry]:
    if index < 0 or index >= len(entries) - 1:
        return entries
    entries[index], entries[index + 1] = entries[index + 1], entries[index]
    _reassign_orders(entries)
    return entries


def move_entry_to_top(entries: list[FileEntry], index: int) -> list[FileEntry]:
    if index <= 0 or index >= len(entries):
        return entries
    entries.insert(0, entries.pop(index))
    _reassign_orders(entries)
    return entries


def move_entry_to_bottom(entries: list[FileEntry], index: int) -> list[FileEntry]:
    if index < 0 or index >= len(entries) - 1:
        return entries
    entries.append(entries.pop(index))
    _reassign_orders(entries)
    return entries


def apply_manual_orders(entries: list[FileEntry], order_map: dict[str, int]) -> list[FileEntry]:
    """Apply a {filename: new_order} map and re-sort by the assigned orders."""
    for entry in entries:
        if entry.filename in order_map:
            entry.sort_order = order_map[entry.filename]
    entries.sort(key=lambda e: (e.sort_order, e.filename))
    _reassign_orders(entries)
    return entries


def _reassign_orders(entries: list[FileEntry]) -> None:
    for i, e in enumerate(entries):
        e.sort_order = i + 1


# ══════════════════════════════════════════════════════════════════════════════
# Duplicate detection
# ══════════════════════════════════════════════════════════════════════════════

def find_duplicate_hashes(entries: list[FileEntry]) -> dict[str, list[FileEntry]]:
    """Return {hash: [entries]} for hashes that appear more than once."""
    from collections import defaultdict
    buckets: dict[str, list[FileEntry]] = defaultdict(list)
    for e in entries:
        if e.file_hash:
            buckets[e.file_hash].append(e)
    return {h: lst for h, lst in buckets.items() if len(lst) > 1}


def find_duplicate_titles(entries: list[FileEntry]) -> dict[str, list[FileEntry]]:
    """Return {title: [entries]} for effective titles that appear more than once."""
    from collections import defaultdict
    buckets: dict[str, list[FileEntry]] = defaultdict(list)
    for e in entries:
        t = e.effective_title.strip().lower()
        if t:
            buckets[t].append(e)
    return {t: lst for t, lst in buckets.items() if len(lst) > 1}
