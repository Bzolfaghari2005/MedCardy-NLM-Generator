"""
shared_source_service.py – Shared source library management.

Handles file storage, SHA-256 deduplication, and upload tracking.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
from pathlib import Path
from typing import Optional

import database as db
from models import AttachMode, SharedSource, SourceScope, UploadStatus
from settings import DB_PATH, SHARED_GLOBAL_DIR, SHARED_PROJECT_DIR

logger = logging.getLogger(__name__)

# Supported file extensions (based on notebooklm-py capabilities)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ══════════════════════════════════════════════════════════════════════════════
# File hashing
# ══════════════════════════════════════════════════════════════════════════════

def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# MIME detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    ext = Path(filename).suffix.lower()
    return {
        ".pdf":  "application/pdf",
        ".txt":  "text/plain",
        ".md":   "text/markdown",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(ext, "application/octet-stream")


def is_supported_format(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


# ══════════════════════════════════════════════════════════════════════════════
# Add source from uploaded bytes
# ══════════════════════════════════════════════════════════════════════════════

def add_global_source(
    file_bytes: bytes,
    original_filename: str,
    display_name: str = "",
    description: str = "",
    path: Path = DB_PATH,
) -> SharedSource:
    """Store a global shared source and register it in DB."""
    return _add_source(
        file_bytes=file_bytes,
        original_filename=original_filename,
        scope=SourceScope.GLOBAL,
        display_name=display_name,
        description=description,
        storage_dir=SHARED_GLOBAL_DIR,
        project_id=None,
        path=path,
    )


def add_project_source(
    file_bytes: bytes,
    original_filename: str,
    project_id: int,
    display_name: str = "",
    description: str = "",
    path: Path = DB_PATH,
) -> SharedSource:
    """
    Store a project-scoped shared source and auto-link it to the project.

    Project sources are ONLY uploaded to notebooks belonging to this project.
    They are never shared with other projects.
    """
    project_dir = SHARED_PROJECT_DIR / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    source = _add_source(
        file_bytes=file_bytes,
        original_filename=original_filename,
        scope=SourceScope.PROJECT,
        display_name=display_name,
        description=description,
        storage_dir=project_dir,
        project_id=project_id,
        path=path,
    )
    # Auto-attach: project sources are always active for their project
    db.attach_shared_source_to_project(
        project_id, source.id, AttachMode.ALL_NOTEBOOKS, path=path
    )
    return source


def _add_source(
    file_bytes: bytes,
    original_filename: str,
    scope: SourceScope,
    display_name: str,
    description: str,
    storage_dir: Path,
    project_id: Optional[int],
    path: Path,
) -> SharedSource:
    file_hash = compute_bytes_hash(file_bytes)
    mime_type = detect_mime_type(original_filename)
    suffix = Path(original_filename).suffix
    stored_filename = f"{file_hash[:16]}{suffix}"
    stored_path = storage_dir / stored_filename

    if not stored_path.exists():
        stored_path.write_bytes(file_bytes)

    source_id = db.create_shared_source(
        scope=scope,
        display_name=display_name or original_filename,
        file_path=str(stored_path),
        original_filename=original_filename,
        file_hash=file_hash,
        mime_type=mime_type,
        file_size=len(file_bytes),
        description=description,
        project_id=project_id,
        path=path,
    )
    source = db.get_shared_source(source_id, path)
    assert source is not None
    return source


def replace_source_file(
    source_id: int,
    new_file_bytes: bytes,
    new_filename: str,
    path: Path = DB_PATH,
) -> SharedSource:
    """Replace a source's file with new content (re-hashes and re-stores)."""
    source = db.get_shared_source(source_id, path)
    if not source:
        raise ValueError(f"Shared source {source_id} not found.")

    new_hash = compute_bytes_hash(new_file_bytes)
    old_path = Path(source.file_path)
    storage_dir = old_path.parent
    suffix = Path(new_filename).suffix
    new_stored_filename = f"{new_hash[:16]}{suffix}"
    new_stored_path = storage_dir / new_stored_filename

    if not new_stored_path.exists():
        new_stored_path.write_bytes(new_file_bytes)

    db.update_shared_source(source_id, {
        "file_path": str(new_stored_path),
        "original_filename": new_filename,
        "file_hash": new_hash,
        "file_size": len(new_file_bytes),
        "mime_type": detect_mime_type(new_filename),
    }, path)

    updated = db.get_shared_source(source_id, path)
    assert updated is not None
    return updated


# ══════════════════════════════════════════════════════════════════════════════
# Project attachment management
# ══════════════════════════════════════════════════════════════════════════════

def attach_global_sources_to_project(
    project_id: int,
    source_ids: Optional[list[int]] = None,
    attach_mode: AttachMode = AttachMode.ALL_NOTEBOOKS,
    path: Path = DB_PATH,
) -> None:
    """Attach global sources (all or specified) to a project."""
    if source_ids is None:
        sources = db.list_global_shared_sources(path)
        source_ids = [s.id for s in sources if s.enabled]

    for sid in source_ids:
        db.attach_shared_source_to_project(project_id, sid, attach_mode, path=path)


def get_sources_for_notebook(
    project_id: int,
    chunk_id: int,
    account_id: int,
    path: Path = DB_PATH,
) -> list[SharedSource]:
    """
    Return all SharedSources that should be uploaded to a specific notebook.
    Respects attach_mode filtering.
    """
    import json
    links = db.get_project_source_links(project_id, path)
    result: list[SharedSource] = []

    for link in links:
        if not link.enabled:
            continue
        source = db.get_shared_source(link.shared_source_id, path)
        if not source or not source.enabled:
            continue

        mode = link.attach_mode
        if mode == AttachMode.DISABLED:
            continue
        elif mode == AttachMode.ALL_NOTEBOOKS:
            result.append(source)
        elif mode == AttachMode.SELECTED_CHUNKS:
            try:
                selected = json.loads(link.selected_chunk_ids_json or "[]")
                if chunk_id in selected:
                    result.append(source)
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.error(
                    "Invalid selected chunk list for source link %d; source skipped.",
                    link.id,
                )
        elif mode == AttachMode.SELECTED_ACCOUNTS:
            try:
                selected = json.loads(link.selected_account_ids_json or "[]")
                if account_id in selected:
                    result.append(source)
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.error(
                    "Invalid selected account list for source link %d; source skipped.",
                    link.id,
                )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Deduplication check
# ══════════════════════════════════════════════════════════════════════════════

def is_source_already_uploaded(
    job_id: int,
    shared_source_id: int,
    current_hash: str,
    path: Path = DB_PATH,
) -> Optional[str]:
    """
    Check if this shared source was already uploaded for this job with same hash.
    Returns source_id string if already uploaded, None otherwise.
    """
    existing = db.get_source_upload(job_id, shared_source_id, path)
    if existing and existing.status == UploadStatus.UPLOADED:
        if existing.file_hash == current_hash and existing.source_id:
            return existing.source_id
    return None


def record_source_upload(
    job_id: int,
    shared_source_id: int,
    file_hash: str,
    source_id: str,
    path: Path = DB_PATH,
) -> None:
    existing = db.get_source_upload(job_id, shared_source_id, path)
    if existing:
        db.update_source_upload(existing.id, {
            "file_hash": file_hash,
            "source_id": source_id,
            "status": UploadStatus.UPLOADED.value,
            "error_message": None,
        }, path)
    else:
        upload_id = db.create_source_upload(job_id, shared_source_id, file_hash, path)
        db.update_source_upload(upload_id, {
            "source_id": source_id,
            "status": UploadStatus.UPLOADED.value,
        }, path)


def record_source_upload_failed(
    job_id: int,
    shared_source_id: int,
    file_hash: str,
    error: str,
    path: Path = DB_PATH,
) -> None:
    existing = db.get_source_upload(job_id, shared_source_id, path)
    if existing:
        db.update_source_upload(existing.id, {
            "file_hash": file_hash,
            "status": UploadStatus.FAILED.value,
            "error_message": error,
        }, path)
    else:
        upload_id = db.create_source_upload(job_id, shared_source_id, file_hash, path)
        db.update_source_upload(upload_id, {
            "status": UploadStatus.FAILED.value,
            "error_message": error,
        }, path)


# ══════════════════════════════════════════════════════════════════════════════
# Source count validation
# ══════════════════════════════════════════════════════════════════════════════

NOTEBOOKLM_MAX_SOURCES = 50  # provider limit


def estimate_sources_per_notebook(project_id: int, path: Path = DB_PATH) -> dict:
    """Estimate how many sources each notebook will have."""
    active = db.get_active_sources_for_project(project_id, path)
    total = 1 + len(active)  # 1 PDF chunk + shared sources
    warning = total >= NOTEBOOKLM_MAX_SOURCES * 0.8
    error = total > NOTEBOOKLM_MAX_SOURCES
    return {
        "main_pdf": 1,
        "shared_sources": len(active),
        "total": total,
        "limit": NOTEBOOKLM_MAX_SOURCES,
        "warning": warning,
        "error": error,
        "message": (
            f"Each notebook will include {total} source(s) "
            f"(maximum allowed: {NOTEBOOKLM_MAX_SOURCES})"
        ),
    }
