"""models.py – Enums and dataclasses for NLM Audio Generator."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class AccountStatus(str, Enum):
    ACTIVE       = "ACTIVE"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    RATE_LIMITED = "RATE_LIMITED"
    DISABLED     = "DISABLED"
    CHECKING     = "CHECKING"
    ERROR        = "ERROR"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"


class AccountType(str, Enum):
    FREE         = "FREE"
    PAID         = "PAID"
    ORGANIZATION = "ORGANIZATION"
    UNKNOWN      = "UNKNOWN"


class ProjectStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    PAUSED    = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    STOPPED   = "STOPPED"


class AllocationMode(str, Enum):
    EXACT    = "EXACT"    # each account gets exactly its quota
    FLEXIBLE = "FLEXIBLE" # overflow allowed if another account is available


class JobStatus(str, Enum):
    PENDING                = "PENDING"
    ASSIGNED               = "ASSIGNED"
    CREATING_NOTEBOOK      = "CREATING_NOTEBOOK"
    UPLOADING_MAIN_SOURCE  = "UPLOADING_MAIN_SOURCE"
    UPLOADING_SHARED_SOURCES = "UPLOADING_SHARED_SOURCES"
    WAITING_FOR_SOURCES    = "WAITING_FOR_SOURCES"
    GENERATING_AUDIO       = "GENERATING_AUDIO"
    WAITING_FOR_AUDIO      = "WAITING_FOR_AUDIO"
    DOWNLOADING_AUDIO      = "DOWNLOADING_AUDIO"
    CONVERTING_AUDIO       = "CONVERTING_AUDIO"
    TRANSCRIBING_AUDIO     = "TRANSCRIBING_AUDIO"
    COMPLETED              = "COMPLETED"
    FAILED                 = "FAILED"
    CANCELLED              = "CANCELLED"


class SourceScope(str, Enum):
    GLOBAL  = "GLOBAL"
    PROJECT = "PROJECT"


class AttachMode(str, Enum):
    ALL_NOTEBOOKS      = "ALL_NOTEBOOKS"
    SELECTED_CHUNKS    = "SELECTED_CHUNKS"
    SELECTED_ACCOUNTS  = "SELECTED_ACCOUNTS"
    DISABLED           = "DISABLED"


class UploadStatus(str, Enum):
    PENDING   = "PENDING"
    UPLOADED  = "UPLOADED"
    FAILED    = "FAILED"
    SKIPPED   = "SKIPPED"


class ConversionStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


class TranscriptionStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


# ── AI Folder Processor enums ──────────────────────────────────────────────────

class AIJobStatus(str, Enum):
    DISCOVERED      = "DISCOVERED"
    SELECTED        = "SELECTED"
    EXTRACTING      = "EXTRACTING"
    EXTRACTED       = "EXTRACTED"
    WAITING         = "WAITING"
    CHUNKING        = "CHUNKING"
    SENDING         = "SENDING"
    WAITING_FOR_API = "WAITING_FOR_API"
    MERGING         = "MERGING"
    SAVING_RESULT   = "SAVING_RESULT"
    COMPLETED       = "COMPLETED"
    SKIPPED         = "SKIPPED"
    FAILED          = "FAILED"
    CANCELLED       = "CANCELLED"


class AIChunkMode(str, Enum):
    DIRECT        = "DIRECT"
    CHUNKED       = "CHUNKED"
    CHUNKED_MERGE = "CHUNKED_MERGE"


class AIRunStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    PAUSED    = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    STOPPED   = "STOPPED"


class AIFileGroup(str, Enum):
    TEXT    = "TEXT"
    PDF     = "PDF"
    OFFICE  = "OFFICE"
    CODE    = "CODE"
    IMAGE   = "IMAGE"
    AUDIO   = "AUDIO"
    VIDEO   = "VIDEO"
    ARCHIVE = "ARCHIVE"
    UNKNOWN = "UNKNOWN"
    ALL     = "ALL"


class AIConnectionStatus(str, Enum):
    CONNECTED           = "CONNECTED"
    INVALID_API_KEY     = "INVALID_API_KEY"
    INSUFFICIENT_CREDIT = "INSUFFICIENT_CREDIT"
    RATE_LIMITED        = "RATE_LIMITED"
    NETWORK_ERROR       = "NETWORK_ERROR"
    API_ERROR           = "API_ERROR"


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Account:
    id: int
    profile_name: str
    display_name: str
    description: str
    account_type: AccountType
    enabled: bool
    sort_order: int
    default_project_job_limit: int
    default_concurrency: int
    auth_status: AccountStatus
    last_auth_check_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class Project:
    id: int
    name: str
    slug: str
    original_filename: str
    original_pdf_path: str
    original_pdf_hash: str
    total_pages: int
    prompt_template: str
    language: str
    audio_format: str
    auto_convert_to_mp3: bool
    mp3_bitrate: str
    keep_original_audio: bool
    auto_transcribe: bool
    whisper_model: str
    whisper_language: str
    allocation_mode: AllocationMode
    status: ProjectStatus
    output_dir: str
    created_at: datetime
    updated_at: datetime


@dataclass
class ProjectAccountAllocation:
    id: int
    project_id: int
    account_id: int
    enabled: bool
    max_jobs_for_project: int
    max_concurrent_jobs: int
    priority: int
    allow_overflow: bool
    assigned_jobs_count: int
    completed_jobs_count: int
    failed_jobs_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class Chunk:
    id: int
    project_id: int
    chunk_index: int
    start_page: int
    end_page: int
    pdf_path: str
    pdf_hash: str
    assigned_account_id: Optional[int]
    status: str
    created_at: datetime
    updated_at: datetime

    @property
    def page_label(self) -> str:
        return f"{self.start_page}-{self.end_page}"

    @property
    def filename_stem(self) -> str:
        return f"{self.chunk_index:03d}_pages_{self.start_page:03d}_{self.end_page:03d}"


@dataclass
class Job:
    id: int
    project_id: int
    chunk_id: int
    account_id: Optional[int]
    notebook_id: Optional[str]
    main_source_id: Optional[str]
    artifact_id: Optional[str]
    prompt_rendered: Optional[str]
    downloaded_audio_path: Optional[str]
    converted_mp3_path: Optional[str]
    transcript_txt_path: Optional[str]
    status: JobStatus
    current_step: Optional[str]
    attempt_count: int
    error_code: Optional[str]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    # populated by joins
    chunk_index: int = 0
    start_page: int = 0
    end_page: int = 0
    account_profile: str = ""

    @property
    def page_label(self) -> str:
        return f"{self.start_page}-{self.end_page}"

    @property
    def chunk_filename_stem(self) -> str:
        return f"{self.chunk_index:03d}_pages_{self.start_page:03d}_{self.end_page:03d}"


@dataclass
class SharedSource:
    id: int
    scope: SourceScope
    project_id: Optional[int]
    display_name: str
    description: str
    file_path: str
    original_filename: str
    file_hash: str
    mime_type: str
    file_size: int
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class ProjectSharedSource:
    id: int
    project_id: int
    shared_source_id: int
    enabled: bool
    attach_mode: AttachMode
    selected_chunk_ids_json: str
    selected_account_ids_json: str
    created_at: datetime
    updated_at: datetime


@dataclass
class NotebookSourceUpload:
    id: int
    job_id: int
    shared_source_id: int
    file_hash: str
    source_id: Optional[str]
    status: UploadStatus
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class AudioConversion:
    id: int
    project_id: Optional[int]
    job_id: Optional[int]
    input_path: str
    output_path: Optional[str]
    target_format: str
    bitrate: str
    status: ConversionStatus
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


@dataclass
class Transcription:
    id: int
    project_id: Optional[int]
    job_id: Optional[int]
    source_type: str
    input_path: str
    input_hash: str
    model_name: str
    language: str
    device: str
    compute_type: str
    output_txt_path: Optional[str]
    output_srt_path: Optional[str]
    output_vtt_path: Optional[str]
    output_json_path: Optional[str]
    status: TranscriptionStatus
    progress: float
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime


# ── AI Folder Processor dataclasses ──────────────────────────────────────────

@dataclass
class AIPromptProfile:
    id: int
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    file_group: str
    is_default: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class AIBatchRun:
    id: int
    name: str
    input_folder: str
    output_folder: str
    recursive: bool
    model: str
    base_url: str
    prompt_profile_id: Optional[int]
    max_concurrency: int
    timeout_seconds: int
    max_retries: int
    chunk_max_tokens: int
    chunk_overlap_tokens: int
    chunk_mode: AIChunkMode
    include_hidden_files: bool
    preserve_directory_structure: bool
    status: AIRunStatus
    total_files: int
    completed_files: int
    failed_files: int
    skipped_files: int
    estimated_input_tokens: Optional[int]
    actual_input_tokens: int
    actual_output_tokens: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class AIFileJob:
    id: int
    run_id: int
    relative_path: str
    absolute_input_path: str
    input_filename: str
    extension: str
    mime_type: str
    file_size: int
    file_hash: str
    file_group: str
    extraction_method: str
    extracted_text_path: Optional[str]
    prompt_profile_id: Optional[int]
    rendered_prompt_hash: Optional[str]
    model: str
    chunk_count: int
    completed_chunk_count: int
    output_txt_path: Optional[str]
    output_json_path: Optional[str]
    status: AIJobStatus
    attempt_count: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    error_code: Optional[str]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class AIFileChunk:
    id: int
    file_job_id: int
    chunk_index: int
    chunk_hash: str
    input_text_path: Optional[str]
    output_text_path: Optional[str]
    status: str
    attempt_count: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    error_code: Optional[str]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


# ─── Worker event (sent through multiprocessing.Queue) ────────────────────────

@dataclass
class WorkerEvent:
    event: str   # "job_status" | "account_status" | "heartbeat"
    account_id: int
    job_id: Optional[int] = None
    status: Optional[str] = None
    message: Optional[str] = None
    notebook_id: Optional[str] = None
    main_source_id: Optional[str] = None
    artifact_id: Optional[str] = None
    downloaded_audio_path: Optional[str] = None
    error_message: Optional[str] = None
    account_status: Optional[str] = None
    current_step: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
