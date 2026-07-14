"""
file_service.py – File management, ZIP exports, disk usage reporting.
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
import time
import zipfile
from pathlib import Path
from typing import Optional

import database as db
from settings import DB_PATH

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Disk usage
# ══════════════════════════════════════════════════════════════════════════════

def dir_size(path: Path) -> int:
    """Return total size in bytes of all files under path."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fmt_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024**2:.1f} MB"
    return f"{size_bytes / 1024**3:.2f} GB"


def project_storage_report(project_id: int, path: Path = DB_PATH) -> dict:
    """Return a breakdown of storage used by a project."""
    proj = db.get_project(project_id, path)
    if not proj or not proj.output_dir:
        return {}

    project_dir = Path(proj.output_dir)
    dirs = {
        "original": project_dir / "original",
        "chunks": project_dir / "chunks",
        "audio_original": project_dir / "audio_original",
        "audio_mp3": project_dir / "audio_mp3",
        "transcripts": project_dir / "transcripts",
        "exports": project_dir / "exports",
    }

    report = {}
    total = 0
    for label, d in dirs.items():
        size = dir_size(d)
        total += size
        report[label] = {"size": size, "size_fmt": fmt_size(size)}

    report["total"] = {"size": total, "size_fmt": fmt_size(total)}
    return report


def global_storage_report() -> dict:
    """Return storage breakdown across all data directories."""
    from settings import (
        DATA_DIR, PROJECTS_DIR, SHARED_GLOBAL_DIR,
        SHARED_PROJECT_DIR, TOOLS_DIR,
    )
    dirs = {
        "Projects": PROJECTS_DIR,
        "Global Sources": SHARED_GLOBAL_DIR,
        "Project Sources": SHARED_PROJECT_DIR,
        "Tools": TOOLS_DIR,
    }
    report = {}
    total = 0
    for label, d in dirs.items():
        size = dir_size(d)
        total += size
        report[label] = {"size": size, "fmt": fmt_size(size)}
    report["Total"] = {"size": total, "fmt": fmt_size(total)}
    return report


# ══════════════════════════════════════════════════════════════════════════════
# File listing
# ══════════════════════════════════════════════════════════════════════════════

def list_project_files(project_id: int, path: Path = DB_PATH) -> dict[str, list[dict]]:
    """Return categorized file list for a project."""
    proj = db.get_project(project_id, path)
    if not proj or not proj.output_dir:
        return {}

    project_dir = Path(proj.output_dir)
    categories = {
        "Original PDF":     (project_dir / "original",       [".pdf"]),
        "PDF Chunks":       (project_dir / "chunks",          [".pdf"]),
        "Audio M4A":        (project_dir / "audio_original",  [".m4a"]),
        "MP3":              (project_dir / "audio_mp3",       [".mp3"]),
        "Transcripts":      (project_dir / "transcripts",     [".txt", ".srt", ".vtt", ".json"]),
        "ZIP Exports":      (project_dir / "exports",         [".zip"]),
    }

    result: dict[str, list[dict]] = {}
    for label, (d, exts) in categories.items():
        files = []
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size": fmt_size(f.stat().st_size),
                        "size_bytes": f.stat().st_size,
                        "exists": True,
                    })
        result[label] = files

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ZIP export
# ══════════════════════════════════════════════════════════════════════════════

def create_project_zip(
    project_id: int,
    include_original_pdf: bool = True,
    include_chunks: bool = False,
    include_audio_m4a: bool = True,
    include_audio_mp3: bool = True,
    include_transcripts: bool = True,
    path: Path = DB_PATH,
) -> Optional[Path]:
    """Create a ZIP archive for a project. Returns path to ZIP or None on failure."""
    proj = db.get_project(project_id, path)
    if not proj or not proj.output_dir:
        return None

    project_dir = Path(proj.output_dir)
    exports_dir = project_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    safe_name = proj.slug or f"project_{project_id}"
    zip_path = exports_dir / f"{safe_name}_complete.zip"

    include_map = {
        "original": include_original_pdf,
        "chunks":   include_chunks,
        "audio_original": include_audio_m4a,
        "audio_mp3": include_audio_mp3,
        "transcripts": include_transcripts,
    }

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder, include in include_map.items():
            if not include:
                continue
            folder_path = project_dir / folder
            if not folder_path.exists():
                continue
            for f in sorted(folder_path.rglob("*")):
                if f.is_file():
                    arcname = f.relative_to(project_dir)
                    zf.write(f, arcname)

    logger.info("Created ZIP: %s (%.1f MB)", zip_path.name, zip_path.stat().st_size / 1024**2)
    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════════

def delete_temp_files(project_id: int, path: Path = DB_PATH) -> int:
    """Remove incomplete temporary files. Returns count deleted."""
    proj = db.get_project(project_id, path)
    if not proj or not proj.output_dir:
        return 0
    count = 0
    candidates = {
        *Path(proj.output_dir).rglob("*.part"),
        *Path(proj.output_dir).rglob("*.part.*"),
        *Path(proj.output_dir).rglob("*.tmp"),
    }
    for f in candidates:
        try:
            f.unlink(missing_ok=True)
            count += 1
        except PermissionError:
            logger.warning("Temporary file is currently locked: %s", f)
    return count


def delete_project_files(project_id: int, path: Path = DB_PATH) -> None:
    """Delete all local files for a project directory, retrying Windows locks."""
    proj = db.get_project(project_id, path)
    if not proj or not proj.output_dir:
        return
    d = Path(proj.output_dir)
    if d.exists():
        last_error: OSError | None = None
        for attempt in range(5):
            try:
                shutil.rmtree(d, onerror=_make_writable_and_retry)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.4 * (attempt + 1))
        if last_error is not None:
            raise RuntimeError(
                "Project files are in use. Pause the project and stop any "
                "conversion/download before deleting it."
            ) from last_error
        logger.info("Deleted project directory: %s", d)


def _make_writable_and_retry(function, file_path: str, exc_info) -> None:
    """Handle read-only files encountered by shutil.rmtree."""
    if not isinstance(exc_info[1], PermissionError):
        raise exc_info[1]
    os.chmod(file_path, stat.S_IWRITE)
    function(file_path)


def safe_delete_file(file_path: Path) -> tuple[bool, str]:
    """Delete a single file safely. Returns (success, message)."""
    try:
        if not file_path.exists():
            return False, "File not found."
        file_path.unlink()
        return True, "File deleted."
    except Exception as exc:
        return False, str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# Slug helper
# ══════════════════════════════════════════════════════════════════════════════

def slugify(text: str) -> str:
    """Create a filesystem-safe slug from project name."""
    import re
    from datetime import datetime
    safe = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe.strip())
    safe = re.sub(r"[-_]+", "_", safe)
    if not safe:
        safe = "project"
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{safe[:40]}_{ts}"
