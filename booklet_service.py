"""
booklet_service.py – Orchestrator for the Word Booklet Maker (جزوه‌ساز Word).

Coordinates scanning, sorting, parsing, rendering, manifest generation,
DB persistence, and rebuild-cache logic.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

from database import get_db
from settings import BOOKLETS_DIR, DB_PATH

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Error codes
# ══════════════════════════════════════════════════════════════════════════════

class BookletError(str, Enum):
    SOURCE_FOLDER_NOT_FOUND  = "SOURCE_FOLDER_NOT_FOUND"
    NO_TEXT_FILES_FOUND      = "NO_TEXT_FILES_FOUND"
    FILE_READ_FAILED         = "FILE_READ_FAILED"
    INVALID_ENCODING         = "INVALID_ENCODING"
    EMPTY_FILE               = "EMPTY_FILE"
    MARKDOWN_PARSE_FAILED    = "MARKDOWN_PARSE_FAILED"
    INVALID_TABLE_JSON       = "INVALID_TABLE_JSON"
    DOCX_RENDER_FAILED       = "DOCX_RENDER_FAILED"
    OUTPUT_PERMISSION_DENIED = "OUTPUT_PERMISSION_DENIED"
    FONT_NOT_AVAILABLE       = "FONT_NOT_AVAILABLE"
    IMAGE_LOAD_FAILED        = "IMAGE_LOAD_FAILED"
    MANIFEST_SAVE_FAILED     = "MANIFEST_SAVE_FAILED"
    BUILD_CANCELLED          = "BUILD_CANCELLED"
    UNKNOWN_ERROR            = "UNKNOWN_ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# Filename sanitisation
# ══════════════════════════════════════════════════════════════════════════════

_WINDOWS_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_PATH_TRAVERSAL = re.compile(r"\.\./|\.\.\\")


def sanitize_filename(name: str) -> str:
    """Return a safe Windows filename (no path traversal, no invalid chars)."""
    name = _PATH_TRAVERSAL.sub("", name)
    name = _WINDOWS_INVALID_CHARS.sub("_", name)
    name = name.strip(". ")
    if not name:
        name = "booklet"
    return name[:200]


def unique_output_path(output_dir: Path, filename: str) -> Path:
    """Return a non-colliding path by appending _v2, _v3 … if needed."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".docx"
    candidate = output_dir / filename
    if not candidate.exists():
        return candidate
    v = 2
    while True:
        candidate = output_dir / f"{stem}_v{v}{suffix}"
        if not candidate.exists():
            return candidate
        v += 1


def make_slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^\w\u0600-\u06FF]+", "_", s)
    s = s.strip("_")
    return s[:60] or "booklet"


# ══════════════════════════════════════════════════════════════════════════════
# DB helpers
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.utcnow().isoformat()


def _create_or_get_project(name: str, slug: str, source_folder: str) -> int:
    with get_db(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM booklet_projects WHERE slug=?", (slug,)
        ).fetchone()
        if row:
            return row["id"]
        now = _now()
        cur = conn.execute(
            """INSERT INTO booklet_projects
               (name, slug, source_folder, output_folder, created_at, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (name, slug, source_folder,
             str(BOOKLETS_DIR / slug / "output"), now, now),
        )
        return cur.lastrowid


def _upsert_booklet_items(project_id: int, entries) -> None:
    now = _now()
    with get_db(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM booklet_items WHERE booklet_project_id=?", (project_id,)
        )
        for entry in entries:
            conn.execute(
                """INSERT INTO booklet_items
                   (booklet_project_id, source_file_path, relative_path, filename,
                    file_hash, file_size, created_time, modified_time,
                    detected_title, custom_title, sort_order, enabled,
                    word_count, heading_count, table_count, invalid_table_count,
                    parse_warnings_json, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    project_id,
                    str(entry.source_path),
                    entry.relative_path,
                    entry.filename,
                    entry.file_hash,
                    entry.file_size,
                    entry.created_time.isoformat() if entry.created_time else None,
                    entry.modified_time.isoformat(),
                    entry.detected_title,
                    entry.custom_title,
                    entry.sort_order,
                    1 if entry.enabled else 0,
                    entry.word_count,
                    entry.heading_count,
                    entry.table_count,
                    0,
                    json.dumps([], ensure_ascii=False),
                    now, now,
                ),
            )


def _create_build_record(project_id: int, snapshot_hash: str) -> int:
    now = _now()
    with get_db(DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO booklet_builds
               (booklet_project_id, source_snapshot_hash, status, started_at, created_at)
               VALUES (?,?,'RUNNING',?,?)""",
            (project_id, snapshot_hash, now, now),
        )
        return cur.lastrowid


def _finish_build_record(
    build_id: int,
    status: str,
    docx_path: str,
    manifest_path: str,
    merged_md_path: str,
    stats: dict,
    error_message: str = "",
) -> None:
    now = _now()
    with get_db(DB_PATH) as conn:
        conn.execute(
            """UPDATE booklet_builds SET
               status=?, output_docx_path=?, manifest_path=?, merged_markdown_path=?,
               chapter_count=?, word_count=?, table_count=?, invalid_table_count=?,
               error_message=?, completed_at=?
               WHERE id=?""",
            (
                status, docx_path, manifest_path, merged_md_path,
                stats.get("chapter_count", 0),
                stats.get("word_count", 0),
                stats.get("table_count", 0),
                stats.get("invalid_table_count", 0),
                error_message, now, build_id,
            ),
        )


def get_last_build(project_id: int) -> Optional[dict]:
    with get_db(DB_PATH) as conn:
        row = conn.execute(
            """SELECT * FROM booklet_builds
               WHERE booklet_project_id=? AND status='COMPLETED'
               ORDER BY id DESC LIMIT 1""",
            (project_id,),
        ).fetchone()
    if row:
        return dict(row)
    return None


def list_booklet_projects() -> list[dict]:
    with get_db(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT * FROM booklet_projects ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_booklet_project(project_id: int) -> None:
    with get_db(DB_PATH) as conn:
        conn.execute("DELETE FROM booklet_projects WHERE id=?", (project_id,))


# ══════════════════════════════════════════════════════════════════════════════
# Preset helpers
# ══════════════════════════════════════════════════════════════════════════════

def list_presets() -> list[dict]:
    with get_db(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT * FROM booklet_template_presets ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def save_preset(name: str, description: str, settings: dict) -> int:
    now = _now()
    with get_db(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM booklet_template_presets WHERE name=?", (name,)
        ).fetchone()
        if row:
            conn.execute(
                """UPDATE booklet_template_presets SET
                   description=?, settings_json=?, updated_at=? WHERE id=?""",
                (description, json.dumps(settings, ensure_ascii=False), now, row["id"]),
            )
            return row["id"]
        cur = conn.execute(
            """INSERT INTO booklet_template_presets
               (name, description, settings_json, is_default, created_at, updated_at)
               VALUES (?,?,?,0,?,?)""",
            (name, description, json.dumps(settings, ensure_ascii=False), now, now),
        )
        return cur.lastrowid


def delete_preset(preset_id: int) -> None:
    with get_db(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM booklet_template_presets WHERE id=?", (preset_id,)
        )


def get_preset(preset_id: int) -> Optional[dict]:
    with get_db(DB_PATH) as conn:
        row = conn.execute(
            "SELECT * FROM booklet_template_presets WHERE id=?", (preset_id,)
        ).fetchone()
    if row:
        d = dict(row)
        try:
            d["settings"] = json.loads(d.get("settings_json", "{}"))
        except Exception:
            d["settings"] = {}
        return d
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main build pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_build(
    entries,                          # list[FileEntry] – already sorted & enabled-filtered
    chapters,                         # list[ParsedChapter] – already parsed
    settings: dict,
    output_filename: str,
    project_name: str = "booklet",
    source_folder: str = "",
    sort_mode: str = "NATURAL",
    save_merged_preview: bool = True,
    include_full_paths_in_manifest: bool = False,
    force_rebuild: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Orchestrate the full booklet build.

    Returns a result dict with keys:
      success, docx_path, manifest_path, merged_md_path,
      stats, warnings, error_code, error_message, skipped (bool)
    """
    from booklet_manifest_service import (
        build_manifest, build_merged_markdown,
        compute_snapshot_hash, save_manifest, save_merged_markdown,
    )
    from docx_renderer import build_docx

    result: dict = {
        "success": False,
        "docx_path": None,
        "manifest_path": None,
        "merged_md_path": None,
        "stats": {},
        "warnings": [],
        "error_code": None,
        "error_message": "",
        "skipped": False,
    }

    # ── Project record ───────────────────────────────────────────────────────
    slug = make_slug(project_name)
    try:
        project_id = _create_or_get_project(project_name, slug, source_folder)
    except Exception as exc:
        logger.warning("DB project record failed: %s", exc)
        project_id = None

    # ── Output paths ─────────────────────────────────────────────────────────
    booklet_dir = BOOKLETS_DIR / slug
    output_dir = booklet_dir / "output"
    manifest_dir = booklet_dir / "manifests"
    preview_dir = booklet_dir / "previews"
    log_dir = booklet_dir / "logs"
    for d in (output_dir, manifest_dir, preview_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(output_filename)
    if not safe_name.endswith(".docx"):
        safe_name += ".docx"

    overwrite_mode = settings.get("overwrite_mode", "NEW_VERSION")
    if overwrite_mode == "OVERWRITE":
        docx_path = output_dir / safe_name
    else:
        docx_path = unique_output_path(output_dir, safe_name)

    manifest_path = manifest_dir / "booklet_manifest.json"
    merged_md_path = preview_dir / "merged_content.md"

    # ── Snapshot hash & rebuild cache ────────────────────────────────────────
    snapshot_hash = compute_snapshot_hash(entries, settings)
    if not force_rebuild and project_id is not None:
        last = get_last_build(project_id)
        if last and last.get("source_snapshot_hash") == snapshot_hash:
            result["skipped"] = True
            result["success"] = True
            result["docx_path"] = last.get("output_docx_path")
            result["manifest_path"] = last.get("manifest_path")
            result["merged_md_path"] = last.get("merged_markdown_path")
            result["stats"] = {}
            result["warnings"] = ["Previous output is still valid. No changes detected."]
            return result

    # ── DB build record ──────────────────────────────────────────────────────
    build_id = None
    if project_id is not None:
        try:
            build_id = _create_build_record(project_id, snapshot_hash)
        except Exception as exc:
            logger.warning("DB build record failed: %s", exc)

    # ── Save items to DB ─────────────────────────────────────────────────────
    if project_id is not None:
        try:
            _upsert_booklet_items(project_id, entries)
        except Exception as exc:
            logger.warning("DB item save failed: %s", exc)

    # ── Build DOCX ───────────────────────────────────────────────────────────
    try:
        stats = build_docx(chapters, settings, docx_path, progress_callback)
        result["stats"] = stats
        result["warnings"].extend(stats.get("font_warnings", []))
    except PermissionError as exc:
        result["error_code"] = BookletError.OUTPUT_PERMISSION_DENIED
        result["error_message"] = str(exc)
        if build_id:
            _finish_build_record(build_id, "FAILED", "", "", "", {},
                                  result["error_message"])
        return result
    except Exception as exc:
        result["error_code"] = BookletError.DOCX_RENDER_FAILED
        result["error_message"] = str(exc)
        logger.exception("DOCX render failed")
        if build_id:
            _finish_build_record(build_id, "FAILED", "", "", "", {},
                                  result["error_message"])
        return result

    # ── Manifest ─────────────────────────────────────────────────────────────
    try:
        manifest = build_manifest(
            title=settings.get("title", project_name),
            source_folder=source_folder,
            sort_mode=sort_mode,
            entries=entries,
            chapters=chapters,
            build_stats=stats,
            template_settings=settings,
            include_full_paths=include_full_paths_in_manifest,
        )
        save_manifest(manifest, manifest_path)
    except Exception as exc:
        result["warnings"].append(f"Manifest save failed: {exc}")
        result["error_code"] = BookletError.MANIFEST_SAVE_FAILED
        manifest_path = None

    # ── Merged preview ────────────────────────────────────────────────────────
    merged_md_path_actual = None
    if save_merged_preview:
        try:
            merged_md = build_merged_markdown(
                chapters, entries,
                chapter_numbering_mode=settings.get("chapter_numbering_mode", "NONE"),
            )
            save_merged_markdown(merged_md, merged_md_path)
            merged_md_path_actual = merged_md_path
        except Exception as exc:
            result["warnings"].append(f"Merged preview save failed: {exc}")

    # ── Finish DB record ─────────────────────────────────────────────────────
    if build_id:
        try:
            _finish_build_record(
                build_id, "COMPLETED",
                str(docx_path),
                str(manifest_path) if manifest_path else "",
                str(merged_md_path_actual) if merged_md_path_actual else "",
                stats,
            )
        except Exception as exc:
            logger.warning("DB finish build failed: %s", exc)

    result["success"] = True
    result["docx_path"] = str(docx_path)
    result["manifest_path"] = str(manifest_path) if manifest_path else None
    result["merged_md_path"] = str(merged_md_path_actual) if merged_md_path_actual else None
    return result
