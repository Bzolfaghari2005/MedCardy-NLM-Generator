"""
booklet_manifest_service.py – Manifest generation, SHA-256 snapshot hashing,
and merged_content.md preview for the Word Booklet Maker (جزوه‌ساز Word).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from booklet_parser import ParsedChapter
from booklet_sort_service import FileEntry


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot hash
# ══════════════════════════════════════════════════════════════════════════════

def compute_snapshot_hash(
    entries: list[FileEntry],
    template_settings: dict,
) -> str:
    """Return SHA-256 of (file order + file hashes + chapter titles + template).

    Used to detect whether a rebuild is truly necessary.
    """
    h = hashlib.sha256()
    for entry in entries:
        if not entry.enabled:
            continue
        h.update(entry.file_hash.encode())
        h.update(entry.effective_title.encode())
        h.update(str(entry.sort_order).encode())
    h.update(json.dumps(template_settings, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Manifest
# ══════════════════════════════════════════════════════════════════════════════

def build_manifest(
    title: str,
    source_folder: str,
    sort_mode: str,
    entries: list[FileEntry],
    chapters: list[ParsedChapter],
    build_stats: dict,
    template_settings: dict,
    include_full_paths: bool = False,
) -> dict:
    """Return a manifest dict (JSON-serialisable)."""
    files_list = []
    for i, entry in enumerate(e for e in entries if e.enabled):
        chapter = chapters[i] if i < len(chapters) else None
        warnings = []
        if chapter:
            warnings = chapter.warnings

        file_record: dict = {
            "order": entry.sort_order,
            "filename": entry.filename,
            "relative_path": entry.relative_path,
            "chapter_title": entry.effective_title,
            "file_hash": entry.file_hash,
            "file_size": entry.file_size,
            "encoding": entry.encoding,
            "word_count": entry.word_count,
            "heading_count": entry.heading_count,
            "table_count": chapter.table_count if chapter else 0,
            "invalid_table_count": chapter.invalid_table_count if chapter else 0,
            "parse_warnings": warnings,
        }
        if include_full_paths:
            file_record["full_path"] = str(entry.source_path)

        files_list.append(file_record)

    return {
        "title": title,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_folder": source_folder if include_full_paths else "",
        "sort_mode": sort_mode,
        "build_stats": build_stats,
        "template": {k: v for k, v in template_settings.items() if k != "logo_path"},
        "files": files_list,
    }


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Merged preview Markdown
# ══════════════════════════════════════════════════════════════════════════════

def build_merged_markdown(
    chapters: list[ParsedChapter],
    entries: list[FileEntry],
    chapter_numbering_mode: str = "NONE",
) -> str:
    """Return a merged Markdown string for debug / preview purposes."""
    parts: list[str] = []
    fa_digits = "۰۱۲۳۴۵۶۷۸۹"

    for i, chapter in enumerate(chapters):
        n = i + 1
        fa_n = "".join(fa_digits[int(d)] for d in str(n))

        prefix = ""
        if chapter_numbering_mode == "CHAPTER_FA":
            prefix = f"فصل {fa_n}: "
        elif chapter_numbering_mode == "CHAPTER_EN":
            prefix = f"Chapter {n}: "

        parts.append(f"# {prefix}{chapter.title}\n")

        source_name = entries[i].filename if i < len(entries) else str(chapter.source_path.name)
        parts.append(f"*منبع: {source_name}*\n\n")
        parts.append("---\n\n")

        # Re-read raw content for preview (simpler than reconstructing from blocks)
        try:
            raw = chapter.source_path.read_text(encoding="utf-8-sig")
        except Exception:
            try:
                raw = chapter.source_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                raw = f"[خطا در خواندن فایل: {chapter.source_path.name}]"

        parts.append(raw)
        parts.append("\n\n")

    return "".join(parts)


def save_merged_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
