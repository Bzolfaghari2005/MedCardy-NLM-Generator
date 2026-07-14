"""
database.py – SQLite WAL access layer.

Single-writer design: only runner.py (main process / event consumer) writes.
Streamlit and worker processes read only (except for specific init operations).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from models import (
    Account, AccountStatus, AccountType,
    AllocationMode, AttachMode,
    AudioConversion, ConversionStatus,
    Chunk, Job, JobStatus,
    NotebookSourceUpload,
    Project, ProjectAccountAllocation, ProjectSharedSource, ProjectStatus,
    SharedSource, SourceScope,
    Transcription, TranscriptionStatus, UploadStatus,
    AIPromptProfile, AIBatchRun, AIFileJob, AIFileChunk,
    AIJobStatus, AIChunkMode, AIRunStatus,
)
from settings import DB_BUSY_TIMEOUT_MS, DB_PATH, daily_job_quota_for


# ══════════════════════════════════════════════════════════════════════════════
# Connection
# ══════════════════════════════════════════════════════════════════════════════

def _get_connection(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db(path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    conn = _get_connection(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# Schema DDL
# ══════════════════════════════════════════════════════════════════════════════

_DDL = """
CREATE TABLE IF NOT EXISTS accounts (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name              TEXT    NOT NULL UNIQUE,
    display_name              TEXT    NOT NULL DEFAULT '',
    description               TEXT    NOT NULL DEFAULT '',
    account_type              TEXT    NOT NULL DEFAULT 'UNKNOWN',
    enabled                   INTEGER NOT NULL DEFAULT 1,
    sort_order                INTEGER NOT NULL DEFAULT 0,
    default_project_job_limit INTEGER NOT NULL DEFAULT 3,
    default_concurrency       INTEGER NOT NULL DEFAULT 3,
    auth_status               TEXT    NOT NULL DEFAULT 'LOGIN_REQUIRED',
    last_auth_check_at        TEXT,
    created_at                TEXT    NOT NULL,
    updated_at                TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL,
    slug                 TEXT    NOT NULL UNIQUE,
    original_filename    TEXT    NOT NULL DEFAULT '',
    original_pdf_path    TEXT    NOT NULL DEFAULT '',
    original_pdf_hash    TEXT    NOT NULL DEFAULT '',
    total_pages          INTEGER NOT NULL DEFAULT 0,
    prompt_template      TEXT    NOT NULL DEFAULT '',
    language             TEXT    NOT NULL DEFAULT 'fa',
    audio_format         TEXT    NOT NULL DEFAULT 'audio',
    auto_convert_to_mp3  INTEGER NOT NULL DEFAULT 0,
    mp3_bitrate          TEXT    NOT NULL DEFAULT '128k',
    keep_original_audio  INTEGER NOT NULL DEFAULT 1,
    auto_transcribe      INTEGER NOT NULL DEFAULT 0,
    whisper_model        TEXT    NOT NULL DEFAULT 'small',
    whisper_language     TEXT    NOT NULL DEFAULT '',
    allocation_mode      TEXT    NOT NULL DEFAULT 'EXACT',
    status               TEXT    NOT NULL DEFAULT 'PENDING',
    output_dir           TEXT    NOT NULL DEFAULT '',
    created_at           TEXT    NOT NULL,
    updated_at           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS project_account_allocations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id            INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    account_id            INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    enabled               INTEGER NOT NULL DEFAULT 1,
    max_jobs_for_project  INTEGER NOT NULL DEFAULT 3,
    max_concurrent_jobs   INTEGER NOT NULL DEFAULT 3,
    priority              INTEGER NOT NULL DEFAULT 0,
    allow_overflow        INTEGER NOT NULL DEFAULT 0,
    assigned_jobs_count   INTEGER NOT NULL DEFAULT 0,
    completed_jobs_count  INTEGER NOT NULL DEFAULT 0,
    failed_jobs_count     INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL,
    UNIQUE(project_id, account_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    start_page          INTEGER NOT NULL,
    end_page            INTEGER NOT NULL,
    pdf_path            TEXT    NOT NULL DEFAULT '',
    pdf_hash            TEXT    NOT NULL DEFAULT '',
    assigned_account_id INTEGER REFERENCES accounts(id),
    status              TEXT    NOT NULL DEFAULT 'PENDING',
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id             INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    chunk_id               INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    account_id             INTEGER REFERENCES accounts(id),
    notebook_id            TEXT,
    main_source_id         TEXT,
    artifact_id            TEXT,
    prompt_rendered        TEXT,
    downloaded_audio_path  TEXT,
    converted_mp3_path     TEXT,
    transcript_txt_path    TEXT,
    status                 TEXT    NOT NULL DEFAULT 'PENDING',
    current_step           TEXT,
    attempt_count          INTEGER NOT NULL DEFAULT 0,
    error_code             TEXT,
    error_message          TEXT,
    started_at             TEXT,
    completed_at           TEXT,
    created_at             TEXT    NOT NULL,
    updated_at             TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS shared_sources (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    scope             TEXT    NOT NULL DEFAULT 'GLOBAL',
    project_id        INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    display_name      TEXT    NOT NULL DEFAULT '',
    description       TEXT    NOT NULL DEFAULT '',
    file_path         TEXT    NOT NULL DEFAULT '',
    original_filename TEXT    NOT NULL DEFAULT '',
    file_hash         TEXT    NOT NULL DEFAULT '',
    mime_type         TEXT    NOT NULL DEFAULT '',
    file_size         INTEGER NOT NULL DEFAULT 0,
    enabled           INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS project_shared_sources (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id                INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    shared_source_id          INTEGER NOT NULL REFERENCES shared_sources(id) ON DELETE CASCADE,
    enabled                   INTEGER NOT NULL DEFAULT 1,
    attach_mode               TEXT    NOT NULL DEFAULT 'ALL_NOTEBOOKS',
    selected_chunk_ids_json   TEXT    NOT NULL DEFAULT '[]',
    selected_account_ids_json TEXT    NOT NULL DEFAULT '[]',
    created_at                TEXT    NOT NULL,
    updated_at                TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS notebook_source_uploads (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    shared_source_id INTEGER NOT NULL REFERENCES shared_sources(id) ON DELETE CASCADE,
    file_hash        TEXT    NOT NULL DEFAULT '',
    source_id        TEXT,
    status           TEXT    NOT NULL DEFAULT 'PENDING',
    error_message    TEXT,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_conversions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    job_id        INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    input_path    TEXT    NOT NULL,
    output_path   TEXT,
    target_format TEXT    NOT NULL DEFAULT 'mp3',
    bitrate       TEXT    NOT NULL DEFAULT '128k',
    status        TEXT    NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    started_at    TEXT,
    completed_at  TEXT,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id       INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    job_id           INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    source_type      TEXT    NOT NULL DEFAULT 'file',
    input_path       TEXT    NOT NULL,
    input_hash       TEXT    NOT NULL DEFAULT '',
    model_name       TEXT    NOT NULL DEFAULT 'small',
    language         TEXT    NOT NULL DEFAULT '',
    device           TEXT    NOT NULL DEFAULT 'auto',
    compute_type     TEXT    NOT NULL DEFAULT 'auto',
    output_txt_path  TEXT,
    output_srt_path  TEXT,
    output_vtt_path  TEXT,
    output_json_path TEXT,
    status           TEXT    NOT NULL DEFAULT 'PENDING',
    progress         REAL    NOT NULL DEFAULT 0.0,
    error_message    TEXT,
    started_at       TEXT,
    completed_at     TEXT,
    created_at       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

-- ── AI Folder Processor tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ai_prompt_profiles (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    name                  TEXT    NOT NULL UNIQUE,
    description           TEXT    NOT NULL DEFAULT '',
    system_prompt         TEXT    NOT NULL DEFAULT '',
    user_prompt_template  TEXT    NOT NULL DEFAULT '',
    file_group            TEXT    NOT NULL DEFAULT 'ALL',
    is_default            INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_batch_runs (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    name                        TEXT    NOT NULL DEFAULT '',
    input_folder                TEXT    NOT NULL,
    output_folder               TEXT    NOT NULL,
    recursive                   INTEGER NOT NULL DEFAULT 1,
    model                       TEXT    NOT NULL DEFAULT 'gpt-5.2',
    base_url                    TEXT    NOT NULL DEFAULT 'https://api.gapgpt.app/v1',
    prompt_profile_id           INTEGER,
    max_concurrency             INTEGER NOT NULL DEFAULT 3,
    timeout_seconds             INTEGER NOT NULL DEFAULT 180,
    max_retries                 INTEGER NOT NULL DEFAULT 3,
    chunk_max_tokens            INTEGER NOT NULL DEFAULT 6000,
    chunk_overlap_tokens        INTEGER NOT NULL DEFAULT 200,
    chunk_mode                  TEXT    NOT NULL DEFAULT 'CHUNKED_MERGE',
    include_hidden_files        INTEGER NOT NULL DEFAULT 0,
    preserve_directory_structure INTEGER NOT NULL DEFAULT 1,
    status                      TEXT    NOT NULL DEFAULT 'PENDING',
    total_files                 INTEGER NOT NULL DEFAULT 0,
    completed_files             INTEGER NOT NULL DEFAULT 0,
    failed_files                INTEGER NOT NULL DEFAULT 0,
    skipped_files               INTEGER NOT NULL DEFAULT 0,
    estimated_input_tokens      INTEGER,
    actual_input_tokens         INTEGER NOT NULL DEFAULT 0,
    actual_output_tokens        INTEGER NOT NULL DEFAULT 0,
    started_at                  TEXT,
    completed_at                TEXT,
    created_at                  TEXT    NOT NULL,
    updated_at                  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_file_jobs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES ai_batch_runs(id) ON DELETE CASCADE,
    relative_path        TEXT    NOT NULL,
    absolute_input_path  TEXT    NOT NULL,
    input_filename       TEXT    NOT NULL,
    extension            TEXT    NOT NULL DEFAULT '',
    mime_type            TEXT    NOT NULL DEFAULT '',
    file_size            INTEGER NOT NULL DEFAULT 0,
    file_hash            TEXT    NOT NULL DEFAULT '',
    file_group           TEXT    NOT NULL DEFAULT '',
    extraction_method    TEXT    NOT NULL DEFAULT '',
    extracted_text_path  TEXT,
    prompt_profile_id    INTEGER,
    rendered_prompt_hash TEXT,
    model                TEXT    NOT NULL DEFAULT '',
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    completed_chunk_count INTEGER NOT NULL DEFAULT 0,
    output_txt_path      TEXT,
    output_json_path     TEXT,
    status               TEXT    NOT NULL DEFAULT 'DISCOVERED',
    attempt_count        INTEGER NOT NULL DEFAULT 0,
    input_tokens         INTEGER,
    output_tokens        INTEGER,
    error_code           TEXT,
    error_message        TEXT,
    started_at           TEXT,
    completed_at         TEXT,
    created_at           TEXT    NOT NULL,
    updated_at           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_file_chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_job_id  INTEGER NOT NULL REFERENCES ai_file_jobs(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    chunk_hash   TEXT    NOT NULL DEFAULT '',
    input_text_path  TEXT,
    output_text_path TEXT,
    status       TEXT    NOT NULL DEFAULT 'PENDING',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    error_code    TEXT,
    error_message TEXT,
    started_at    TEXT,
    completed_at  TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

-- ═══════════════════════════════════════════════════════
-- Word Booklet Maker (جزوه‌ساز Word)
-- ═══════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS booklet_template_presets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    description   TEXT    NOT NULL DEFAULT '',
    settings_json TEXT    NOT NULL DEFAULT '{}',
    is_default    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS booklet_projects (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    name                     TEXT    NOT NULL,
    slug                     TEXT    NOT NULL UNIQUE,
    source_type              TEXT    NOT NULL DEFAULT 'FOLDER',
    source_batch_run_id      INTEGER REFERENCES ai_batch_runs(id) ON DELETE SET NULL,
    source_folder            TEXT    NOT NULL DEFAULT '',
    output_folder            TEXT    NOT NULL DEFAULT '',
    title                    TEXT    NOT NULL DEFAULT '',
    subtitle                 TEXT    NOT NULL DEFAULT '',
    course_name              TEXT    NOT NULL DEFAULT '',
    university_name          TEXT    NOT NULL DEFAULT '',
    author_name              TEXT    NOT NULL DEFAULT '',
    sort_mode                TEXT    NOT NULL DEFAULT 'NATURAL_FILENAME',
    sort_direction           TEXT    NOT NULL DEFAULT 'ASC',
    recursive                INTEGER NOT NULL DEFAULT 1,
    include_extensions       TEXT    NOT NULL DEFAULT '[".txt",".md",".markdown"]',
    include_toc              INTEGER NOT NULL DEFAULT 1,
    include_cover            INTEGER NOT NULL DEFAULT 1,
    include_header           INTEGER NOT NULL DEFAULT 1,
    include_footer           INTEGER NOT NULL DEFAULT 1,
    include_page_numbers     INTEGER NOT NULL DEFAULT 1,
    chapter_numbering_mode   TEXT    NOT NULL DEFAULT 'NONE',
    first_h1_behavior        TEXT    NOT NULL DEFAULT 'USE_AS_TITLE',
    on_file_error            TEXT    NOT NULL DEFAULT 'SKIP',
    show_source_filename     TEXT    NOT NULL DEFAULT 'NONE',
    template_preset_id       INTEGER REFERENCES booklet_template_presets(id) ON DELETE SET NULL,
    template_settings_json   TEXT    NOT NULL DEFAULT '{}',
    status                   TEXT    NOT NULL DEFAULT 'DRAFT',
    created_at               TEXT    NOT NULL,
    updated_at               TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS booklet_items (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    booklet_project_id    INTEGER NOT NULL REFERENCES booklet_projects(id) ON DELETE CASCADE,
    source_file_path      TEXT    NOT NULL DEFAULT '',
    relative_path         TEXT    NOT NULL DEFAULT '',
    filename              TEXT    NOT NULL DEFAULT '',
    file_hash             TEXT    NOT NULL DEFAULT '',
    file_size             INTEGER NOT NULL DEFAULT 0,
    created_time          TEXT,
    modified_time         TEXT,
    detected_title        TEXT    NOT NULL DEFAULT '',
    custom_title          TEXT    NOT NULL DEFAULT '',
    sort_order            INTEGER NOT NULL DEFAULT 0,
    enabled               INTEGER NOT NULL DEFAULT 1,
    word_count            INTEGER NOT NULL DEFAULT 0,
    heading_count         INTEGER NOT NULL DEFAULT 0,
    table_count           INTEGER NOT NULL DEFAULT 0,
    invalid_table_count   INTEGER NOT NULL DEFAULT 0,
    parse_warnings_json   TEXT    NOT NULL DEFAULT '[]',
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS booklet_builds (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    booklet_project_id    INTEGER NOT NULL REFERENCES booklet_projects(id) ON DELETE CASCADE,
    output_docx_path      TEXT,
    manifest_path         TEXT,
    merged_markdown_path  TEXT,
    source_snapshot_hash  TEXT    NOT NULL DEFAULT '',
    status                TEXT    NOT NULL DEFAULT 'PENDING',
    chapter_count         INTEGER NOT NULL DEFAULT 0,
    word_count            INTEGER NOT NULL DEFAULT 0,
    table_count           INTEGER NOT NULL DEFAULT 0,
    invalid_table_count   INTEGER NOT NULL DEFAULT 0,
    error_message         TEXT,
    started_at            TEXT,
    completed_at          TEXT,
    created_at            TEXT    NOT NULL
);
"""


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_db(path) as conn:
        conn.executescript(_DDL)
        # Older databases could contain duplicate source links because the
        # original ON CONFLICT clause had no matching unique constraint.
        conn.execute(
            """DELETE FROM project_shared_sources
               WHERE id NOT IN (
                   SELECT MIN(id)
                   FROM project_shared_sources
                   GROUP BY project_id, shared_source_id
               )"""
        )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS
               ux_project_shared_source
               ON project_shared_sources(project_id, shared_source_id)"""
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.utcnow().isoformat()


def _dt(val: Optional[str]) -> Optional[datetime]:
    if not val:
        return None
    return datetime.fromisoformat(val)


# ══════════════════════════════════════════════════════════════════════════════
# accounts
# ══════════════════════════════════════════════════════════════════════════════

def create_account(
    profile_name: str,
    display_name: str = "",
    description: str = "",
    account_type: AccountType = AccountType.UNKNOWN,
    default_job_limit: Optional[int] = None,
    default_concurrency: Optional[int] = None,
    path: Path = DB_PATH,
) -> int:
    if default_job_limit is None:
        default_job_limit = daily_job_quota_for(account_type)
    if default_concurrency is None:
        default_concurrency = default_job_limit

    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO accounts
               (profile_name, display_name, description, account_type,
                default_project_job_limit, default_concurrency,
                auth_status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,'LOGIN_REQUIRED',?,?)
               ON CONFLICT(profile_name) DO UPDATE SET
                 display_name=excluded.display_name,
                 updated_at=excluded.updated_at""",
            (profile_name, display_name or profile_name, description,
             account_type.value, default_job_limit, default_concurrency, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_account(account_id: int, path: Path = DB_PATH) -> Optional[Account]:
    with get_db(path) as conn:
        row = conn.execute("SELECT * FROM accounts WHERE id=?", (account_id,)).fetchone()
    return _row_to_account(row) if row else None


def get_account_by_profile(profile_name: str, path: Path = DB_PATH) -> Optional[Account]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM accounts WHERE profile_name=?", (profile_name,)
        ).fetchone()
    return _row_to_account(row) if row else None


def list_accounts(path: Path = DB_PATH) -> list[Account]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM accounts ORDER BY sort_order, id"
        ).fetchall()
    return [_row_to_account(r) for r in rows]


def get_active_accounts(path: Path = DB_PATH) -> list[Account]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM accounts WHERE enabled=1 AND auth_status='ACTIVE' ORDER BY sort_order, id"
        ).fetchall()
    return [_row_to_account(r) for r in rows]


def update_account(account_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [account_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE accounts SET {sets} WHERE id=?", vals)


def update_account_auth_status(
    account_id: int, status: AccountStatus, path: Path = DB_PATH
) -> None:
    now = _now()
    with get_db(path) as conn:
        conn.execute(
            "UPDATE accounts SET auth_status=?, last_auth_check_at=?, updated_at=? WHERE id=?",
            (status.value, now, now, account_id),
        )


def delete_account(account_id: int, path: Path = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.execute("DELETE FROM accounts WHERE id=?", (account_id,))


def _row_to_account(row: sqlite3.Row) -> Account:
    return Account(
        id=row["id"],
        profile_name=row["profile_name"],
        display_name=row["display_name"],
        description=row["description"],
        account_type=AccountType(row["account_type"]),
        enabled=bool(row["enabled"]),
        sort_order=row["sort_order"],
        default_project_job_limit=row["default_project_job_limit"],
        default_concurrency=row["default_concurrency"],
        auth_status=AccountStatus(row["auth_status"]),
        last_auth_check_at=_dt(row["last_auth_check_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# projects
# ══════════════════════════════════════════════════════════════════════════════

def create_project(
    name: str,
    slug: str,
    original_filename: str = "",
    original_pdf_path: str = "",
    total_pages: int = 0,
    prompt_template: str = "",
    language: str = "fa",
    allocation_mode: AllocationMode = AllocationMode.EXACT,
    output_dir: str = "",
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO projects
               (name, slug, original_filename, original_pdf_path, total_pages,
                prompt_template, language, allocation_mode, output_dir,
                status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,'PENDING',?,?)""",
            (name, slug, original_filename, original_pdf_path, total_pages,
             prompt_template, language, allocation_mode.value, output_dir, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_project(project_id: int, path: Path = DB_PATH) -> Optional[Project]:
    with get_db(path) as conn:
        row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    return _row_to_project(row) if row else None


def list_projects(path: Path = DB_PATH) -> list[Project]:
    with get_db(path) as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
    return [_row_to_project(r) for r in rows]


def update_project(project_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [project_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE projects SET {sets} WHERE id=?", vals)


def update_project_status(
    project_id: int, status: ProjectStatus, path: Path = DB_PATH
) -> None:
    with get_db(path) as conn:
        conn.execute(
            "UPDATE projects SET status=?, updated_at=? WHERE id=?",
            (status.value, _now(), project_id),
        )


def delete_project(project_id: int, path: Path = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        slug=row["slug"],
        original_filename=row["original_filename"],
        original_pdf_path=row["original_pdf_path"],
        original_pdf_hash=row["original_pdf_hash"],
        total_pages=row["total_pages"],
        prompt_template=row["prompt_template"],
        language=row["language"],
        audio_format=row["audio_format"],
        auto_convert_to_mp3=bool(row["auto_convert_to_mp3"]),
        mp3_bitrate=row["mp3_bitrate"],
        keep_original_audio=bool(row["keep_original_audio"]),
        auto_transcribe=bool(row["auto_transcribe"]),
        whisper_model=row["whisper_model"],
        whisper_language=row["whisper_language"],
        allocation_mode=AllocationMode(row["allocation_mode"]),
        status=ProjectStatus(row["status"]),
        output_dir=row["output_dir"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# project_account_allocations
# ══════════════════════════════════════════════════════════════════════════════

def upsert_allocation(
    project_id: int,
    account_id: int,
    max_jobs_for_project: int,
    max_concurrent_jobs: int,
    enabled: bool = True,
    priority: int = 0,
    allow_overflow: bool = False,
    path: Path = DB_PATH,
) -> None:
    now = _now()
    with get_db(path) as conn:
        conn.execute(
            """INSERT INTO project_account_allocations
               (project_id, account_id, enabled, max_jobs_for_project,
                max_concurrent_jobs, priority, allow_overflow,
                assigned_jobs_count, completed_jobs_count, failed_jobs_count,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,0,0,0,?,?)
               ON CONFLICT(project_id, account_id) DO UPDATE SET
                 enabled=excluded.enabled,
                 max_jobs_for_project=excluded.max_jobs_for_project,
                 max_concurrent_jobs=excluded.max_concurrent_jobs,
                 priority=excluded.priority,
                 allow_overflow=excluded.allow_overflow,
                 updated_at=excluded.updated_at""",
            (project_id, account_id, int(enabled), max_jobs_for_project,
             max_concurrent_jobs, priority, int(allow_overflow), now, now),
        )


def get_allocations_for_project(
    project_id: int, path: Path = DB_PATH
) -> list[ProjectAccountAllocation]:
    with get_db(path) as conn:
        rows = conn.execute(
            """SELECT * FROM project_account_allocations
               WHERE project_id=? ORDER BY priority DESC, id""",
            (project_id,),
        ).fetchall()
    return [_row_to_allocation(r) for r in rows]


def get_enabled_allocations(
    project_id: int, path: Path = DB_PATH
) -> list[ProjectAccountAllocation]:
    with get_db(path) as conn:
        rows = conn.execute(
            """SELECT paa.* FROM project_account_allocations paa
               JOIN accounts a ON a.id = paa.account_id
               WHERE paa.project_id=? AND paa.enabled=1
                 AND a.enabled=1 AND a.auth_status='ACTIVE'
               ORDER BY paa.priority DESC, paa.id""",
            (project_id,),
        ).fetchall()
    return [_row_to_allocation(r) for r in rows]


def increment_allocation_counter(
    project_id: int, account_id: int, field: str, path: Path = DB_PATH
) -> None:
    allowed = {"assigned_jobs_count", "completed_jobs_count", "failed_jobs_count"}
    if field not in allowed:
        raise ValueError(f"Invalid counter field: {field}")
    with get_db(path) as conn:
        conn.execute(
            f"UPDATE project_account_allocations SET {field}={field}+1, updated_at=? "
            "WHERE project_id=? AND account_id=?",
            (_now(), project_id, account_id),
        )


def _row_to_allocation(row: sqlite3.Row) -> ProjectAccountAllocation:
    return ProjectAccountAllocation(
        id=row["id"],
        project_id=row["project_id"],
        account_id=row["account_id"],
        enabled=bool(row["enabled"]),
        max_jobs_for_project=row["max_jobs_for_project"],
        max_concurrent_jobs=row["max_concurrent_jobs"],
        priority=row["priority"],
        allow_overflow=bool(row["allow_overflow"]),
        assigned_jobs_count=row["assigned_jobs_count"],
        completed_jobs_count=row["completed_jobs_count"],
        failed_jobs_count=row["failed_jobs_count"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# chunks
# ══════════════════════════════════════════════════════════════════════════════

def create_chunk(
    project_id: int,
    chunk_index: int,
    start_page: int,
    end_page: int,
    pdf_path: str = "",
    pdf_hash: str = "",
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO chunks
               (project_id, chunk_index, start_page, end_page, pdf_path, pdf_hash,
                status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,'PENDING',?,?)""",
            (project_id, chunk_index, start_page, end_page, pdf_path, pdf_hash, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_chunks_for_project(project_id: int, path: Path = DB_PATH) -> list[Chunk]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE project_id=? ORDER BY chunk_index",
            (project_id,),
        ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def update_chunk(chunk_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [chunk_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE chunks SET {sets} WHERE id=?", vals)


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        project_id=row["project_id"],
        chunk_index=row["chunk_index"],
        start_page=row["start_page"],
        end_page=row["end_page"],
        pdf_path=row["pdf_path"],
        pdf_hash=row["pdf_hash"],
        assigned_account_id=row["assigned_account_id"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# jobs
# ══════════════════════════════════════════════════════════════════════════════

def create_job(
    project_id: int,
    chunk_id: int,
    account_id: Optional[int] = None,
    path: Path = DB_PATH,
) -> int:
    job_id, _ = create_job_if_absent(project_id, chunk_id, account_id, path)
    return job_id


def create_job_if_absent(
    project_id: int,
    chunk_id: int,
    account_id: Optional[int] = None,
    path: Path = DB_PATH,
) -> tuple[int, bool]:
    """Return ``(job_id, created)`` for a project's chunk.

    Keeping the created flag in the database layer lets callers update counters
    only when the insert actually happened.
    """
    now = _now()
    with get_db(path) as conn:
        existing = conn.execute(
            "SELECT id FROM jobs WHERE project_id=? AND chunk_id=? ORDER BY id LIMIT 1",
            (project_id, chunk_id),
        ).fetchone()
        if existing:
            return existing["id"], False
        cur = conn.execute(
            """INSERT INTO jobs
               (project_id, chunk_id, account_id, status, created_at, updated_at)
               VALUES (?,?,?,'PENDING',?,?)""",
            (project_id, chunk_id, account_id, now, now),
        )
        return cur.lastrowid, True  # type: ignore[return-value]


def get_job(job_id: int, path: Path = DB_PATH) -> Optional[Job]:
    with get_db(path) as conn:
        row = conn.execute(_JOB_SELECT + " WHERE j.id=?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def get_jobs_for_project(project_id: int, path: Path = DB_PATH) -> list[Job]:
    with get_db(path) as conn:
        rows = conn.execute(
            _JOB_SELECT + " WHERE j.project_id=? ORDER BY c.chunk_index",
            (project_id,),
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def get_pending_jobs(project_id: int, path: Path = DB_PATH) -> list[Job]:
    with get_db(path) as conn:
        rows = conn.execute(
            _JOB_SELECT + " WHERE j.project_id=? AND j.status NOT IN "
            "('COMPLETED','FAILED','CANCELLED') ORDER BY c.chunk_index",
            (project_id,),
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def get_failed_jobs(project_id: int, path: Path = DB_PATH) -> list[Job]:
    with get_db(path) as conn:
        rows = conn.execute(
            _JOB_SELECT + " WHERE j.project_id=? AND j.status='FAILED' ORDER BY c.chunk_index",
            (project_id,),
        ).fetchall()
    return [_row_to_job(r) for r in rows]


def update_job(job_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [job_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE jobs SET {sets} WHERE id=?", vals)


def update_job_status(
    job_id: int,
    status: JobStatus,
    *,
    current_step: Optional[str] = None,
    notebook_id: Optional[str] = None,
    main_source_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    downloaded_audio_path: Optional[str] = None,
    converted_mp3_path: Optional[str] = None,
    transcript_txt_path: Optional[str] = None,
    error_message: Optional[str] = None,
    error_code: Optional[str] = None,
    increment_attempt: bool = False,
    path: Path = DB_PATH,
) -> None:
    now = _now()
    sets: list[str] = ["status=?", "updated_at=?"]
    params: list[Any] = [status.value, now]

    if current_step is not None:
        sets.append("current_step=?"); params.append(current_step)
    if notebook_id is not None:
        sets.append("notebook_id=?"); params.append(notebook_id)
    if main_source_id is not None:
        sets.append("main_source_id=?"); params.append(main_source_id)
    if artifact_id is not None:
        sets.append("artifact_id=?"); params.append(artifact_id)
    if downloaded_audio_path is not None:
        sets.append("downloaded_audio_path=?"); params.append(downloaded_audio_path)
    if converted_mp3_path is not None:
        sets.append("converted_mp3_path=?"); params.append(converted_mp3_path)
    if transcript_txt_path is not None:
        sets.append("transcript_txt_path=?"); params.append(transcript_txt_path)
    if error_message is not None:
        sets.append("error_message=?"); params.append(error_message)
    if error_code is not None:
        sets.append("error_code=?"); params.append(error_code)
    if increment_attempt:
        sets.append("attempt_count=attempt_count+1")
    if status == JobStatus.CREATING_NOTEBOOK:
        sets.append("started_at=?"); params.append(now)
    if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        sets.append("completed_at=?"); params.append(now)

    params.append(job_id)
    with get_db(path) as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id=?", params)


def reset_job_for_retry(
    job_id: int,
    account_id: Optional[int] = None,
    path: Path = DB_PATH,
) -> None:
    updates: dict[str, Any] = {
        "status": "PENDING",
        "artifact_id": None,
        "error_message": None,
        "error_code": None,
        "current_step": None,
        "completed_at": None,
    }
    if account_id is not None:
        updates["account_id"] = account_id
    update_job(job_id, updates, path)


_JOB_SELECT = """
SELECT j.*,
       c.chunk_index, c.start_page, c.end_page,
       a.profile_name as account_profile
FROM jobs j
JOIN chunks c ON c.id = j.chunk_id
LEFT JOIN accounts a ON a.id = j.account_id
"""


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        project_id=row["project_id"],
        chunk_id=row["chunk_id"],
        account_id=row["account_id"],
        notebook_id=row["notebook_id"],
        main_source_id=row["main_source_id"],
        artifact_id=row["artifact_id"],
        prompt_rendered=row["prompt_rendered"],
        downloaded_audio_path=row["downloaded_audio_path"],
        converted_mp3_path=row["converted_mp3_path"],
        transcript_txt_path=row["transcript_txt_path"],
        status=JobStatus(row["status"]),
        current_step=row["current_step"],
        attempt_count=row["attempt_count"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        chunk_index=row["chunk_index"],
        start_page=row["start_page"],
        end_page=row["end_page"],
        account_profile=row["account_profile"] or "",
    )


# ══════════════════════════════════════════════════════════════════════════════
# shared_sources
# ══════════════════════════════════════════════════════════════════════════════

def create_shared_source(
    scope: SourceScope,
    display_name: str,
    file_path: str,
    original_filename: str,
    file_hash: str,
    mime_type: str,
    file_size: int,
    description: str = "",
    project_id: Optional[int] = None,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO shared_sources
               (scope, project_id, display_name, description, file_path,
                original_filename, file_hash, mime_type, file_size, enabled,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,1,?,?)""",
            (scope.value, project_id, display_name, description, file_path,
             original_filename, file_hash, mime_type, file_size, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def list_global_shared_sources(path: Path = DB_PATH) -> list[SharedSource]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM shared_sources WHERE scope='GLOBAL' ORDER BY id"
        ).fetchall()
    return [_row_to_shared_source(r) for r in rows]


def list_project_shared_sources(
    project_id: int, path: Path = DB_PATH
) -> list[SharedSource]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM shared_sources WHERE scope='PROJECT' AND project_id=? ORDER BY id",
            (project_id,),
        ).fetchall()
    return [_row_to_shared_source(r) for r in rows]


def get_shared_source(source_id: int, path: Path = DB_PATH) -> Optional[SharedSource]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM shared_sources WHERE id=?", (source_id,)
        ).fetchone()
    return _row_to_shared_source(row) if row else None


def update_shared_source(
    source_id: int, updates: dict[str, Any], path: Path = DB_PATH
) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [source_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE shared_sources SET {sets} WHERE id=?", vals)


def delete_shared_source(source_id: int, path: Path = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.execute("DELETE FROM shared_sources WHERE id=?", (source_id,))


def _row_to_shared_source(row: sqlite3.Row) -> SharedSource:
    return SharedSource(
        id=row["id"],
        scope=SourceScope(row["scope"]),
        project_id=row["project_id"],
        display_name=row["display_name"],
        description=row["description"],
        file_path=row["file_path"],
        original_filename=row["original_filename"],
        file_hash=row["file_hash"],
        mime_type=row["mime_type"],
        file_size=row["file_size"],
        enabled=bool(row["enabled"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# project_shared_sources  (linking table)
# ══════════════════════════════════════════════════════════════════════════════

def attach_shared_source_to_project(
    project_id: int,
    shared_source_id: int,
    attach_mode: AttachMode = AttachMode.ALL_NOTEBOOKS,
    enabled: bool = True,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        row = conn.execute(
            """INSERT INTO project_shared_sources
               (project_id, shared_source_id, enabled, attach_mode,
                selected_chunk_ids_json, selected_account_ids_json,
                created_at, updated_at)
               VALUES (?,?,?,?,'[]','[]',?,?)
               ON CONFLICT(project_id, shared_source_id) DO UPDATE SET
                   enabled=excluded.enabled,
                   attach_mode=excluded.attach_mode,
                   updated_at=excluded.updated_at
               RETURNING id""",
            (project_id, shared_source_id, int(enabled), attach_mode.value, now, now),
        ).fetchone()
        return row["id"]


def get_project_source_links(
    project_id: int, path: Path = DB_PATH
) -> list[ProjectSharedSource]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM project_shared_sources WHERE project_id=? ORDER BY id",
            (project_id,),
        ).fetchall()
    return [_row_to_pss(r) for r in rows]


def get_active_sources_for_project(
    project_id: int, path: Path = DB_PATH
) -> list[SharedSource]:
    """Return all active SharedSources attached to a project (global + project-scoped)."""
    with get_db(path) as conn:
        rows = conn.execute(
            """SELECT ss.* FROM shared_sources ss
               JOIN project_shared_sources pss ON pss.shared_source_id = ss.id
               WHERE pss.project_id=? AND pss.enabled=1 AND ss.enabled=1
               ORDER BY ss.id""",
            (project_id,),
        ).fetchall()
    return [_row_to_shared_source(r) for r in rows]


def _row_to_pss(row: sqlite3.Row) -> ProjectSharedSource:
    return ProjectSharedSource(
        id=row["id"],
        project_id=row["project_id"],
        shared_source_id=row["shared_source_id"],
        enabled=bool(row["enabled"]),
        attach_mode=AttachMode(row["attach_mode"]),
        selected_chunk_ids_json=row["selected_chunk_ids_json"],
        selected_account_ids_json=row["selected_account_ids_json"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# notebook_source_uploads
# ══════════════════════════════════════════════════════════════════════════════

def create_source_upload(
    job_id: int,
    shared_source_id: int,
    file_hash: str,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO notebook_source_uploads
               (job_id, shared_source_id, file_hash, status, created_at, updated_at)
               VALUES (?,?,?,'PENDING',?,?)""",
            (job_id, shared_source_id, file_hash, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_source_upload(
    job_id: int, shared_source_id: int, path: Path = DB_PATH
) -> Optional[NotebookSourceUpload]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM notebook_source_uploads WHERE job_id=? AND shared_source_id=?",
            (job_id, shared_source_id),
        ).fetchone()
    return _row_to_nsu(row) if row else None


def update_source_upload(upload_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [upload_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE notebook_source_uploads SET {sets} WHERE id=?", vals)


def _row_to_nsu(row: sqlite3.Row) -> NotebookSourceUpload:
    return NotebookSourceUpload(
        id=row["id"],
        job_id=row["job_id"],
        shared_source_id=row["shared_source_id"],
        file_hash=row["file_hash"],
        source_id=row["source_id"],
        status=UploadStatus(row["status"]),
        error_message=row["error_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# audio_conversions
# ══════════════════════════════════════════════════════════════════════════════

def create_audio_conversion(
    input_path: str,
    bitrate: str = "128k",
    project_id: Optional[int] = None,
    job_id: Optional[int] = None,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO audio_conversions
               (project_id, job_id, input_path, target_format, bitrate, status, created_at)
               VALUES (?,?,?,'mp3',?,'PENDING',?)""",
            (project_id, job_id, input_path, bitrate, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def list_audio_conversions(
    project_id: Optional[int] = None, path: Path = DB_PATH
) -> list[AudioConversion]:
    with get_db(path) as conn:
        if project_id is not None:
            rows = conn.execute(
                "SELECT * FROM audio_conversions WHERE project_id=? ORDER BY id DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audio_conversions ORDER BY id DESC"
            ).fetchall()
    return [_row_to_ac(r) for r in rows]


def update_audio_conversion(conv_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [conv_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE audio_conversions SET {sets} WHERE id=?", vals)


def _row_to_ac(row: sqlite3.Row) -> AudioConversion:
    return AudioConversion(
        id=row["id"],
        project_id=row["project_id"],
        job_id=row["job_id"],
        input_path=row["input_path"],
        output_path=row["output_path"],
        target_format=row["target_format"],
        bitrate=row["bitrate"],
        status=ConversionStatus(row["status"]),
        error_message=row["error_message"],
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# transcriptions
# ══════════════════════════════════════════════════════════════════════════════

def create_transcription(
    input_path: str,
    input_hash: str,
    model_name: str = "small",
    language: str = "",
    device: str = "auto",
    compute_type: str = "auto",
    project_id: Optional[int] = None,
    job_id: Optional[int] = None,
    source_type: str = "file",
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO transcriptions
               (project_id, job_id, source_type, input_path, input_hash,
                model_name, language, device, compute_type, status, progress, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,'PENDING',0.0,?)""",
            (project_id, job_id, source_type, input_path, input_hash,
             model_name, language, device, compute_type, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def list_transcriptions(
    project_id: Optional[int] = None, path: Path = DB_PATH
) -> list[Transcription]:
    with get_db(path) as conn:
        if project_id is not None:
            rows = conn.execute(
                "SELECT * FROM transcriptions WHERE project_id=? ORDER BY id DESC",
                (project_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transcriptions ORDER BY id DESC"
            ).fetchall()
    return [_row_to_tr(r) for r in rows]


def update_transcription(tr_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [tr_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE transcriptions SET {sets} WHERE id=?", vals)


def delete_transcription(tr_id: int, path: Path = DB_PATH) -> None:
    """Permanently remove a transcription record from the DB."""
    with get_db(path) as conn:
        conn.execute("DELETE FROM transcriptions WHERE id=?", (tr_id,))


def _row_to_tr(row: sqlite3.Row) -> Transcription:
    return Transcription(
        id=row["id"],
        project_id=row["project_id"],
        job_id=row["job_id"],
        source_type=row["source_type"],
        input_path=row["input_path"],
        input_hash=row["input_hash"],
        model_name=row["model_name"],
        language=row["language"],
        device=row["device"],
        compute_type=row["compute_type"],
        output_txt_path=row["output_txt_path"],
        output_srt_path=row["output_srt_path"],
        output_vtt_path=row["output_vtt_path"],
        output_json_path=row["output_json_path"],
        status=TranscriptionStatus(row["status"]),
        progress=row["progress"],
        error_message=row["error_message"],
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# app_settings
# ══════════════════════════════════════════════════════════════════════════════

def get_setting(key: str, default: str = "", path: Path = DB_PATH) -> str:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key=?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str, path: Path = DB_PATH) -> None:
    now = _now()
    with get_db(path) as conn:
        conn.execute(
            """INSERT INTO app_settings (key, value, updated_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, now),
        )


def get_all_settings(path: Path = DB_PATH) -> dict[str, str]:
    with get_db(path) as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


# ══════════════════════════════════════════════════════════════════════════════
# ai_prompt_profiles
# ══════════════════════════════════════════════════════════════════════════════

def create_ai_prompt_profile(
    name: str,
    system_prompt: str = "",
    user_prompt_template: str = "",
    description: str = "",
    file_group: str = "ALL",
    is_default: bool = False,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO ai_prompt_profiles
               (name, description, system_prompt, user_prompt_template,
                file_group, is_default, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (name, description, system_prompt, user_prompt_template,
             file_group, int(is_default), now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_ai_prompt_profile(profile_id: int, path: Path = DB_PATH) -> Optional[AIPromptProfile]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM ai_prompt_profiles WHERE id=?", (profile_id,)
        ).fetchone()
    return _row_to_ai_prompt_profile(row) if row else None


def list_ai_prompt_profiles(path: Path = DB_PATH) -> list[AIPromptProfile]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM ai_prompt_profiles ORDER BY is_default DESC, name"
        ).fetchall()
    return [_row_to_ai_prompt_profile(r) for r in rows]


def get_default_ai_prompt_profile(path: Path = DB_PATH) -> Optional[AIPromptProfile]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM ai_prompt_profiles WHERE is_default=1 LIMIT 1"
        ).fetchone()
    return _row_to_ai_prompt_profile(row) if row else None


def update_ai_prompt_profile(
    profile_id: int, updates: dict[str, Any], path: Path = DB_PATH
) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [profile_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE ai_prompt_profiles SET {sets} WHERE id=?", vals)


def set_default_ai_prompt_profile(profile_id: int, path: Path = DB_PATH) -> None:
    now = _now()
    with get_db(path) as conn:
        conn.execute("UPDATE ai_prompt_profiles SET is_default=0, updated_at=?", (now,))
        conn.execute(
            "UPDATE ai_prompt_profiles SET is_default=1, updated_at=? WHERE id=?",
            (now, profile_id),
        )


def delete_ai_prompt_profile(profile_id: int, path: Path = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.execute("DELETE FROM ai_prompt_profiles WHERE id=?", (profile_id,))


def _row_to_ai_prompt_profile(row: sqlite3.Row) -> AIPromptProfile:
    return AIPromptProfile(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        system_prompt=row["system_prompt"],
        user_prompt_template=row["user_prompt_template"],
        file_group=row["file_group"],
        is_default=bool(row["is_default"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_batch_runs
# ══════════════════════════════════════════════════════════════════════════════

def create_ai_batch_run(
    input_folder: str,
    output_folder: str,
    model: str = "gpt-5.2",
    base_url: str = "https://api.gapgpt.app/v1",
    name: str = "",
    recursive: bool = True,
    prompt_profile_id: Optional[int] = None,
    max_concurrency: int = 3,
    timeout_seconds: int = 180,
    max_retries: int = 3,
    chunk_max_tokens: int = 6000,
    chunk_overlap_tokens: int = 200,
    chunk_mode: str = "CHUNKED_MERGE",
    include_hidden_files: bool = False,
    preserve_directory_structure: bool = True,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO ai_batch_runs
               (name, input_folder, output_folder, recursive, model, base_url,
                prompt_profile_id, max_concurrency, timeout_seconds, max_retries,
                chunk_max_tokens, chunk_overlap_tokens, chunk_mode,
                include_hidden_files, preserve_directory_structure,
                status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'PENDING',?,?)""",
            (name, input_folder, output_folder, int(recursive), model, base_url,
             prompt_profile_id, max_concurrency, timeout_seconds, max_retries,
             chunk_max_tokens, chunk_overlap_tokens, chunk_mode,
             int(include_hidden_files), int(preserve_directory_structure), now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_ai_batch_run(run_id: int, path: Path = DB_PATH) -> Optional[AIBatchRun]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM ai_batch_runs WHERE id=?", (run_id,)
        ).fetchone()
    return _row_to_ai_batch_run(row) if row else None


def list_ai_batch_runs(path: Path = DB_PATH) -> list[AIBatchRun]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM ai_batch_runs ORDER BY id DESC"
        ).fetchall()
    return [_row_to_ai_batch_run(r) for r in rows]


def update_ai_batch_run(run_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [run_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE ai_batch_runs SET {sets} WHERE id=?", vals)


def delete_ai_batch_run(run_id: int, path: Path = DB_PATH) -> None:
    with get_db(path) as conn:
        conn.execute("DELETE FROM ai_batch_runs WHERE id=?", (run_id,))


def _row_to_ai_batch_run(row: sqlite3.Row) -> AIBatchRun:
    return AIBatchRun(
        id=row["id"],
        name=row["name"],
        input_folder=row["input_folder"],
        output_folder=row["output_folder"],
        recursive=bool(row["recursive"]),
        model=row["model"],
        base_url=row["base_url"],
        prompt_profile_id=row["prompt_profile_id"],
        max_concurrency=row["max_concurrency"],
        timeout_seconds=row["timeout_seconds"],
        max_retries=row["max_retries"],
        chunk_max_tokens=row["chunk_max_tokens"],
        chunk_overlap_tokens=row["chunk_overlap_tokens"],
        chunk_mode=AIChunkMode(row["chunk_mode"]),
        include_hidden_files=bool(row["include_hidden_files"]),
        preserve_directory_structure=bool(row["preserve_directory_structure"]),
        status=AIRunStatus(row["status"]),
        total_files=row["total_files"],
        completed_files=row["completed_files"],
        failed_files=row["failed_files"],
        skipped_files=row["skipped_files"],
        estimated_input_tokens=row["estimated_input_tokens"],
        actual_input_tokens=row["actual_input_tokens"],
        actual_output_tokens=row["actual_output_tokens"],
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_file_jobs
# ══════════════════════════════════════════════════════════════════════════════

def create_ai_file_job(
    run_id: int,
    relative_path: str,
    absolute_input_path: str,
    input_filename: str,
    extension: str = "",
    mime_type: str = "",
    file_size: int = 0,
    file_hash: str = "",
    file_group: str = "",
    extraction_method: str = "",
    model: str = "",
    prompt_profile_id: Optional[int] = None,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO ai_file_jobs
               (run_id, relative_path, absolute_input_path, input_filename,
                extension, mime_type, file_size, file_hash, file_group,
                extraction_method, model, prompt_profile_id,
                status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'DISCOVERED',?,?)""",
            (run_id, relative_path, absolute_input_path, input_filename,
             extension, mime_type, file_size, file_hash, file_group,
             extraction_method, model, prompt_profile_id, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def get_ai_file_job(job_id: int, path: Path = DB_PATH) -> Optional[AIFileJob]:
    with get_db(path) as conn:
        row = conn.execute(
            "SELECT * FROM ai_file_jobs WHERE id=?", (job_id,)
        ).fetchone()
    return _row_to_ai_file_job(row) if row else None


def list_ai_file_jobs(run_id: int, path: Path = DB_PATH) -> list[AIFileJob]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM ai_file_jobs WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
    return [_row_to_ai_file_job(r) for r in rows]


def list_ai_file_jobs_by_status(
    run_id: int, status: AIJobStatus, path: Path = DB_PATH
) -> list[AIFileJob]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM ai_file_jobs WHERE run_id=? AND status=? ORDER BY id",
            (run_id, status.value),
        ).fetchall()
    return [_row_to_ai_file_job(r) for r in rows]


def update_ai_file_job(job_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [job_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE ai_file_jobs SET {sets} WHERE id=?", vals)


def find_completed_ai_file_job(
    run_id: int,
    rendered_prompt_hash: str,
    path: Path = DB_PATH,
) -> Optional[AIFileJob]:
    """Find a COMPLETED job with matching dedup key."""
    with get_db(path) as conn:
        row = conn.execute(
            """SELECT * FROM ai_file_jobs
               WHERE run_id=? AND rendered_prompt_hash=? AND status='COMPLETED'
               LIMIT 1""",
            (run_id, rendered_prompt_hash),
        ).fetchone()
    return _row_to_ai_file_job(row) if row else None


def _row_to_ai_file_job(row: sqlite3.Row) -> AIFileJob:
    return AIFileJob(
        id=row["id"],
        run_id=row["run_id"],
        relative_path=row["relative_path"],
        absolute_input_path=row["absolute_input_path"],
        input_filename=row["input_filename"],
        extension=row["extension"],
        mime_type=row["mime_type"],
        file_size=row["file_size"],
        file_hash=row["file_hash"],
        file_group=row["file_group"],
        extraction_method=row["extraction_method"],
        extracted_text_path=row["extracted_text_path"],
        prompt_profile_id=row["prompt_profile_id"],
        rendered_prompt_hash=row["rendered_prompt_hash"],
        model=row["model"],
        chunk_count=row["chunk_count"],
        completed_chunk_count=row["completed_chunk_count"],
        output_txt_path=row["output_txt_path"],
        output_json_path=row["output_json_path"],
        status=AIJobStatus(row["status"]),
        attempt_count=row["attempt_count"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_file_chunks
# ══════════════════════════════════════════════════════════════════════════════

def create_ai_file_chunk(
    file_job_id: int,
    chunk_index: int,
    chunk_hash: str = "",
    input_text_path: Optional[str] = None,
    path: Path = DB_PATH,
) -> int:
    now = _now()
    with get_db(path) as conn:
        cur = conn.execute(
            """INSERT INTO ai_file_chunks
               (file_job_id, chunk_index, chunk_hash, input_text_path,
                status, created_at, updated_at)
               VALUES (?,?,?,?,'PENDING',?,?)""",
            (file_job_id, chunk_index, chunk_hash, input_text_path, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]


def list_ai_file_chunks(file_job_id: int, path: Path = DB_PATH) -> list[AIFileChunk]:
    with get_db(path) as conn:
        rows = conn.execute(
            "SELECT * FROM ai_file_chunks WHERE file_job_id=? ORDER BY chunk_index",
            (file_job_id,),
        ).fetchall()
    return [_row_to_ai_file_chunk(r) for r in rows]


def update_ai_file_chunk(chunk_id: int, updates: dict[str, Any], path: Path = DB_PATH) -> None:
    updates["updated_at"] = _now()
    sets = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [chunk_id]
    with get_db(path) as conn:
        conn.execute(f"UPDATE ai_file_chunks SET {sets} WHERE id=?", vals)


def _row_to_ai_file_chunk(row: sqlite3.Row) -> AIFileChunk:
    return AIFileChunk(
        id=row["id"],
        file_job_id=row["file_job_id"],
        chunk_index=row["chunk_index"],
        chunk_hash=row["chunk_hash"],
        input_text_path=row["input_text_path"],
        output_text_path=row["output_text_path"],
        status=row["status"],
        attempt_count=row["attempt_count"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        started_at=_dt(row["started_at"]),
        completed_at=_dt(row["completed_at"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
