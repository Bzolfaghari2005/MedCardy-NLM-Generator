"""
ai_batch_runner.py – Resumable parallel batch processor for AI folder processing.

Architecture:
  - AiBatchRunner runs in a daemon thread started from Streamlit
  - ThreadPoolExecutor handles concurrent API calls (I/O-bound)
  - State is persisted in SQLite after every transition
  - Rate-limit: Exponential Backoff with Retry-After header support
  - Deduplication: completed jobs with matching hash are reused
  - Cancellation: cooperative via a threading.Event
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import database as db
from ai_api_service import AITextProvider, AITextResult, FakeAIProvider, build_provider
from ai_folder_service import DiscoveredFile, compute_file_hash
from chunking_service import (
    build_merge_prompt,
    chunk_input_path,
    chunk_output_path,
    hierarchical_merge,
    load_chunk_output,
    merge_output_path,
    needs_chunking,
    save_chunk_input,
    save_chunk_output,
    split_text,
)
from file_extractor_service import ExtractionResult, extract_file
from models import AIChunkMode, AIFileJob, AIJobStatus, AIRunStatus
from prompt_service import (
    compute_dedup_key,
    compute_extraction_settings_hash,
    compute_prompt_hash,
    get_profile_for_file,
    render_prompt,
)
from settings import (
    AI_INTERMEDIATE_DIR_NAME,
    AI_RESULTS_DIR_NAME,
    AI_RETRY_DELAYS,
    DB_PATH,
)

log = logging.getLogger(__name__)


# ── Run configuration ─────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    run_id: int
    input_folder: Path
    output_folder: Path
    model: str
    base_url: str
    provider: AITextProvider
    max_concurrency: int = 3
    timeout_seconds: int = 180
    max_retries: int = 3
    chunk_max_tokens: int = 6000
    chunk_overlap_tokens: int = 200
    chunk_mode: AIChunkMode = AIChunkMode.CHUNKED_MERGE
    preserve_structure: bool = True
    vision_enabled: bool = False
    audio_mode: str = "transcribe_and_send"
    zip_enabled: bool = False
    force_reprocess: bool = False
    include_absolute_path: bool = False
    output_format: str = "txt"            # txt | txt_json | txt_markdown
    output_header: bool = True
    db_path: Path = DB_PATH

    @property
    def intermediate_dir(self) -> Path:
        return self.output_folder / AI_INTERMEDIATE_DIR_NAME


# ── Runner state ──────────────────────────────────────────────────────────────

@dataclass
class RunnerStats:
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    running: int = 0
    waiting: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: Optional[datetime] = None

    @property
    def elapsed_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        return (datetime.utcnow() - self.started_at).total_seconds()


class AiBatchRunner:
    """Resumable parallel batch runner.

    Usage:
        runner = AiBatchRunner(config)
        thread = threading.Thread(target=runner.run, daemon=True)
        thread.start()

        # To stop gracefully:
        runner.stop()

        # To cancel a specific job:
        runner.cancel_job(job_id)
    """

    def __init__(self, config: RunConfig, on_progress: Optional[Callable] = None):
        self._cfg = config
        self._on_progress = on_progress  # optional callback(stats) called after each job
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._cancelled_jobs: set[int] = set()
        self._lock = threading.Lock()
        self.stats = RunnerStats()
        self._rate_limit_backoff: float = 0.0   # seconds to wait before next request

    def stop(self) -> None:
        """Signal the runner to stop accepting new jobs (current jobs finish)."""
        self._stop_event.set()

    def cancel_job(self, job_id: int) -> None:
        with self._lock:
            self._cancelled_jobs.add(job_id)

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self._cfg
        log.info("AiBatchRunner starting run_id=%s", cfg.run_id)
        self.stats.started_at = datetime.utcnow()

        db.update_ai_batch_run(cfg.run_id, {
            "status": AIRunStatus.RUNNING.value,
            "started_at": self.stats.started_at.isoformat(),
        }, cfg.db_path)

        try:
            jobs = db.list_ai_file_jobs(cfg.run_id, cfg.db_path)
            self.stats.total = len(jobs)

            # Filter: skip already completed (unless force_reprocess)
            pending_jobs = []
            for job in jobs:
                if job.status == AIJobStatus.COMPLETED and not cfg.force_reprocess:
                    self.stats.completed += 1
                    self.stats.skipped += 1
                elif job.status == AIJobStatus.SKIPPED:
                    self.stats.skipped += 1
                elif job.status == AIJobStatus.CANCELLED:
                    pass
                else:
                    pending_jobs.append(job)

            with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
                future_map: dict[Future, AIFileJob] = {}

                for job in pending_jobs:
                    if self._stop_event.is_set():
                        break
                    with self._lock:
                        if job.id in self._cancelled_jobs:
                            _update_job_status(job.id, AIJobStatus.CANCELLED, cfg.db_path)
                            continue

                    # Rate limit gate
                    if self._rate_limit_backoff > 0:
                        time.sleep(self._rate_limit_backoff)
                        self._rate_limit_backoff = 0.0

                    fut = pool.submit(self._process_job, job)
                    future_map[fut] = job

                for fut in as_completed(future_map):
                    job = future_map[fut]
                    try:
                        result_status = fut.result()
                    except Exception as exc:
                        log.error("Unhandled exception in job %s: %s", job.id, exc)
                        result_status = AIJobStatus.FAILED

                    with self._lock:
                        if result_status == AIJobStatus.COMPLETED:
                            self.stats.completed += 1
                        elif result_status == AIJobStatus.FAILED:
                            self.stats.failed += 1
                        elif result_status == AIJobStatus.SKIPPED:
                            self.stats.skipped += 1

                    self._update_run_counters()
                    if self._on_progress:
                        try:
                            self._on_progress(self.stats)
                        except Exception:
                            pass

            final_status = (
                AIRunStatus.STOPPED if self._stop_event.is_set() else AIRunStatus.COMPLETED
            )
            db.update_ai_batch_run(cfg.run_id, {
                "status": final_status.value,
                "completed_at": datetime.utcnow().isoformat(),
                "completed_files": self.stats.completed,
                "failed_files": self.stats.failed,
                "skipped_files": self.stats.skipped,
                "actual_input_tokens": self.stats.input_tokens,
                "actual_output_tokens": self.stats.output_tokens,
            }, cfg.db_path)

        except Exception as exc:
            log.error("AiBatchRunner fatal error: %s", exc, exc_info=True)
            db.update_ai_batch_run(cfg.run_id, {
                "status": AIRunStatus.FAILED.value,
                "completed_at": datetime.utcnow().isoformat(),
            }, cfg.db_path)

    # ── Per-file processing ───────────────────────────────────────────────────

    def _process_job(self, job: AIFileJob) -> AIJobStatus:
        cfg = self._cfg
        job_id = job.id

        # Cancellation check
        with self._lock:
            if job_id in self._cancelled_jobs:
                _update_job_status(job_id, AIJobStatus.CANCELLED, cfg.db_path)
                return AIJobStatus.CANCELLED

        _update_job_status(job_id, AIJobStatus.EXTRACTING, cfg.db_path,
                           started_at=datetime.utcnow().isoformat())

        # ── Deduplication check ──
        if not cfg.force_reprocess and job.rendered_prompt_hash:
            existing = db.find_completed_ai_file_job(cfg.run_id, job.rendered_prompt_hash, cfg.db_path)
            if existing and existing.id != job_id and existing.output_txt_path:
                output_path = Path(existing.output_txt_path)
                if output_path.exists():
                    _update_job_status(job_id, AIJobStatus.COMPLETED, cfg.db_path,
                                       completed_at=datetime.utcnow().isoformat(),
                                       output_txt_path=str(output_path))
                    return AIJobStatus.COMPLETED

        # ── Extract content ──
        input_path = Path(job.absolute_input_path)
        if not input_path.exists():
            return self._fail_job(job_id, "FILE_NOT_FOUND", "فایل ورودی یافت نشد.")

        intermediate = cfg.intermediate_dir / Path(job.relative_path).stem
        extraction = extract_file(
            input_path,
            vision_enabled=cfg.vision_enabled,
            audio_mode=cfg.audio_mode,
            zip_enabled=cfg.zip_enabled,
            intermediate_dir=intermediate,
        )

        if not extraction.success:
            if extraction.error_code == "SKIPPED":
                _update_job_status(job_id, AIJobStatus.SKIPPED, cfg.db_path,
                                   error_code=extraction.error_code,
                                   error_message=extraction.error_message)
                return AIJobStatus.SKIPPED
            if extraction.error_code == "VISION_NOT_SUPPORTED":
                _update_job_status(job_id, AIJobStatus.SKIPPED, cfg.db_path,
                                   error_code=extraction.error_code,
                                   error_message=extraction.error_message)
                return AIJobStatus.SKIPPED
            return self._fail_job(job_id, extraction.error_code or "TEXT_EXTRACTION_FAILED",
                                   extraction.error_message or "خطا در استخراج محتوا")

        if not extraction.has_content:
            return self._fail_job(job_id, "TEXT_EXTRACTION_FAILED", "محتوای قابل استخراج یافت نشد.")

        _update_job_status(job_id, AIJobStatus.EXTRACTED, cfg.db_path)

        # ── Get prompt profile ──
        profiles = db.list_ai_prompt_profiles(cfg.db_path)
        from models import AIFileGroup
        try:
            file_group = AIFileGroup(job.file_group) if job.file_group else AIFileGroup.UNKNOWN
        except ValueError:
            file_group = AIFileGroup.UNKNOWN

        profile = get_profile_for_file(
            extension=job.extension,
            file_group=file_group,
            profiles=profiles,
            run_profile_id=job.prompt_profile_id,
        )

        if not profile:
            return self._fail_job(job_id, "API_ERROR", "هیچ Prompt Profile ای یافت نشد.")

        # ── Image (Vision) path ──
        if extraction.image_data_url and not extraction.text:
            return self._process_vision(job, extraction, profile)

        text = extraction.text or ""

        # ── Render prompt for dedup check ──
        rendered_user = render_prompt(
            profile.user_prompt_template,
            filename=job.input_filename,
            relative_path=job.relative_path,
            absolute_path=job.absolute_input_path,
            extension=job.extension,
            mime_type=job.mime_type,
            file_size=job.file_size,
            file_hash=job.file_hash,
            extraction_method=job.extraction_method,
            page_count=extraction.page_count,
            sheet_count=extraction.sheet_count,
            slide_count=extraction.slide_count,
            file_content="{{PLACEHOLDER}}",   # placeholder before content
            include_absolute_path=cfg.include_absolute_path,
        )
        ph = compute_prompt_hash(profile.system_prompt, rendered_user)
        ext_hash = compute_extraction_settings_hash(
            job.extraction_method, cfg.chunk_max_tokens, cfg.chunk_overlap_tokens,
            cfg.chunk_mode.value, cfg.vision_enabled, cfg.audio_mode,
        )
        dedup_key = compute_dedup_key(job.file_hash, ph, job.model, cfg.base_url, ext_hash)

        db.update_ai_file_job(job_id, {"rendered_prompt_hash": dedup_key}, cfg.db_path)

        # ── Deduplication check with computed key ──
        if not cfg.force_reprocess:
            existing = db.find_completed_ai_file_job(cfg.run_id, dedup_key, cfg.db_path)
            if existing and existing.id != job_id and existing.output_txt_path:
                op = Path(existing.output_txt_path)
                if op.exists():
                    _update_job_status(job_id, AIJobStatus.COMPLETED, cfg.db_path,
                                       completed_at=datetime.utcnow().isoformat(),
                                       output_txt_path=str(op))
                    return AIJobStatus.COMPLETED

        # ── Chunking decision ──
        if needs_chunking(text, cfg.chunk_max_tokens):
            return self._process_chunked(job, text, profile, dedup_key, extraction)
        else:
            return self._process_direct(job, text, profile, dedup_key, extraction)

    def _process_direct(
        self,
        job: AIFileJob,
        text: str,
        profile: Any,
        dedup_key: str,
        extraction: ExtractionResult,
    ) -> AIJobStatus:
        cfg = self._cfg
        _update_job_status(job.id, AIJobStatus.SENDING, cfg.db_path,
                           chunk_count=1)

        user_prompt = render_prompt(
            profile.user_prompt_template,
            filename=job.input_filename,
            relative_path=job.relative_path,
            absolute_path=job.absolute_input_path,
            extension=job.extension,
            mime_type=job.mime_type,
            file_size=job.file_size,
            file_hash=job.file_hash,
            extraction_method=job.extraction_method,
            page_count=extraction.page_count,
            sheet_count=extraction.sheet_count,
            slide_count=extraction.slide_count,
            file_content=text,
            include_absolute_path=cfg.include_absolute_path,
        )

        result = self._call_with_retry(
            job.id, job.model, profile.system_prompt, user_prompt
        )
        if not result.success:
            return self._fail_job(job.id, result.error_code, result.error_message)

        return self._save_result(job, result.text or "", result, profile, dedup_key, chunk_count=1)

    def _process_chunked(
        self,
        job: AIFileJob,
        text: str,
        profile: Any,
        dedup_key: str,
        extraction: ExtractionResult,
    ) -> AIJobStatus:
        cfg = self._cfg
        _update_job_status(job.id, AIJobStatus.CHUNKING, cfg.db_path)

        chunks = split_text(text, cfg.chunk_max_tokens, cfg.chunk_overlap_tokens)
        chunk_count = len(chunks)
        intermediate = cfg.intermediate_dir / Path(job.relative_path).stem
        intermediate.mkdir(parents=True, exist_ok=True)

        db.update_ai_file_job(job.id, {"chunk_count": chunk_count}, cfg.db_path)

        chunk_results: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for chunk in chunks:
            in_path = chunk_input_path(intermediate, Path(job.relative_path).stem, chunk.index)
            out_path = chunk_output_path(intermediate, Path(job.relative_path).stem, chunk.index)

            # Resume: skip already completed chunks
            cached = load_chunk_output(out_path)
            if cached is not None:
                chunk_results.append(cached)
                db.update_ai_file_job(job.id, {
                    "completed_chunk_count": len(chunk_results)
                }, cfg.db_path)
                continue

            save_chunk_input(in_path, chunk.text)

            user_prompt = render_prompt(
                profile.user_prompt_template,
                filename=job.input_filename,
                relative_path=job.relative_path,
                absolute_path=job.absolute_input_path,
                extension=job.extension,
                mime_type=job.mime_type,
                file_size=job.file_size,
                file_hash=job.file_hash,
                extraction_method=job.extraction_method,
                page_count=extraction.page_count,
                sheet_count=extraction.sheet_count,
                slide_count=extraction.slide_count,
                file_content=chunk.text,
                file_index=chunk.index + 1,
                total_files=chunk_count,
                include_absolute_path=cfg.include_absolute_path,
            )

            _update_job_status(job.id, AIJobStatus.SENDING, cfg.db_path)
            result = self._call_with_retry(
                job.id, job.model, profile.system_prompt, user_prompt
            )

            if not result.success:
                return self._fail_job(job.id, result.error_code, result.error_message)

            chunk_text = result.text or ""
            save_chunk_output(out_path, chunk_text)
            chunk_results.append(chunk_text)
            total_input_tokens += result.input_tokens or 0
            total_output_tokens += result.output_tokens or 0

            db.update_ai_file_job(job.id, {
                "completed_chunk_count": len(chunk_results)
            }, cfg.db_path)

        # ── Merge ──
        if cfg.chunk_mode == AIChunkMode.CHUNKED_MERGE and chunk_count > 1:
            _update_job_status(job.id, AIJobStatus.MERGING, cfg.db_path)

            merge_batches = hierarchical_merge(chunk_results, cfg.chunk_max_tokens)
            current_results = chunk_results
            level = 1

            while len(current_results) > 1:
                merged: list[str] = []
                for batch in hierarchical_merge(current_results, cfg.chunk_max_tokens):
                    if len(batch) == 1:
                        merged.append(batch[0])
                        continue
                    merge_prompt = build_merge_prompt(batch, profile.system_prompt,
                                                       profile.user_prompt_template,
                                                       job.input_filename)
                    result = self._call_with_retry(
                        job.id, job.model, profile.system_prompt, merge_prompt
                    )
                    if not result.success:
                        return self._fail_job(job.id, result.error_code, result.error_message)

                    merge_text = result.text or ""
                    mp = merge_output_path(intermediate, Path(job.relative_path).stem, level)
                    save_chunk_output(mp, merge_text)
                    merged.append(merge_text)
                    total_input_tokens += result.input_tokens or 0
                    total_output_tokens += result.output_tokens or 0
                    level += 1

                if len(merged) == len(current_results):
                    break
                current_results = merged

            final_text = current_results[0] if current_results else ""
        else:
            final_text = "\n\n---\n\n".join(chunk_results)

        mock_result = AITextResult(
            text=final_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )
        return self._save_result(job, final_text, mock_result, profile, dedup_key, chunk_count=chunk_count)

    def _process_vision(
        self,
        job: AIFileJob,
        extraction: ExtractionResult,
        profile: Any,
    ) -> AIJobStatus:
        cfg = self._cfg
        _update_job_status(job.id, AIJobStatus.SENDING, cfg.db_path)

        user_text = render_prompt(
            profile.user_prompt_template,
            filename=job.input_filename,
            relative_path=job.relative_path,
            extension=job.extension,
            mime_type=job.mime_type,
            file_size=job.file_size,
            file_hash=job.file_hash,
            extraction_method="vision",
            file_content="[تصویر پیوست شده]",
            include_absolute_path=cfg.include_absolute_path,
        )

        _update_job_status(job.id, AIJobStatus.WAITING_FOR_API, cfg.db_path)
        result = cfg.provider.generate_vision(
            model=job.model,
            system_prompt=profile.system_prompt,
            user_text=user_text,
            image_data_url=extraction.image_data_url or "",
            timeout=cfg.timeout_seconds,
        )

        if not result.success:
            return self._fail_job(job.id, result.error_code, result.error_message)

        dedup_key = compute_dedup_key(
            job.file_hash,
            compute_prompt_hash(profile.system_prompt, user_text),
            job.model, cfg.base_url,
            "vision",
        )
        return self._save_result(job, result.text or "", result, profile, dedup_key, chunk_count=1)

    # ── API call with retry ───────────────────────────────────────────────────

    def _call_with_retry(
        self,
        job_id: int,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> AITextResult:
        cfg = self._cfg
        delays = AI_RETRY_DELAYS

        for attempt in range(cfg.max_retries + 1):
            if self._stop_event.is_set():
                return AITextResult(error_code="CANCELLED_BY_USER", error_message="متوقف شد.")

            _update_job_status(job_id, AIJobStatus.WAITING_FOR_API, cfg.db_path,
                               attempt_count=attempt + 1)

            result = cfg.provider.generate_text(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout=cfg.timeout_seconds,
            )

            if result.success:
                with self._lock:
                    self.stats.input_tokens += result.input_tokens or 0
                    self.stats.output_tokens += result.output_tokens or 0
                return result

            # Context too long → reduce and surface immediately (caller handles)
            if result.error_code == "CONTEXT_LENGTH_EXCEEDED":
                return result

            # Rate limit → back off
            if result.error_code == "RATE_LIMITED":
                if attempt < len(delays):
                    wait = delays[attempt]
                else:
                    wait = delays[-1] * (2 ** (attempt - len(delays) + 1))
                log.warning("Rate limited on job %s, waiting %ss", job_id, wait)
                with self._lock:
                    self._rate_limit_backoff = wait
                time.sleep(wait)
                continue

            # Permanent errors → don't retry
            if result.error_code in ("INVALID_API_KEY", "INSUFFICIENT_CREDIT", "CANCELLED_BY_USER"):
                return result

            # Transient errors → retry
            if attempt < cfg.max_retries:
                wait = delays[attempt] if attempt < len(delays) else delays[-1]
                log.warning("Job %s attempt %s failed (%s), retrying in %ss",
                            job_id, attempt + 1, result.error_code, wait)
                time.sleep(wait)
            else:
                return result

        return AITextResult(error_code="API_ERROR", error_message="همه تلاش‌های مجدد شکست خوردند.")

    # ── Result saving ─────────────────────────────────────────────────────────

    def _save_result(
        self,
        job: AIFileJob,
        text: str,
        api_result: AITextResult,
        profile: Any,
        dedup_key: str,
        chunk_count: int = 1,
    ) -> AIJobStatus:
        cfg = self._cfg
        _update_job_status(job.id, AIJobStatus.SAVING_RESULT, cfg.db_path)

        rel = Path(job.relative_path)
        if cfg.preserve_structure and rel.parent != Path("."):
            out_dir = cfg.output_folder / rel.parent
        else:
            out_dir = cfg.output_folder

        out_dir.mkdir(parents=True, exist_ok=True)
        stem = rel.stem
        txt_path = out_dir / f"{stem}__ai_result.txt"

        # Build output content
        if cfg.output_header:
            header_lines = [
                f"نام فایل: {job.input_filename}",
                f"مسیر نسبی: {job.relative_path}",
                f"مدل: {job.model}",
                f"پرامپت: {profile.name}",
                f"زمان پردازش: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
                f"تعداد Chunk: {chunk_count}",
                "",
                "=" * 50,
                "نتیجه",
                "=" * 50,
                "",
            ]
            content = "\n".join(header_lines) + text
        else:
            content = text

        try:
            txt_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return self._fail_job(job.id, "OUTPUT_SAVE_FAILED", str(exc))

        # Optional JSON metadata
        json_path_str: Optional[str] = None
        if cfg.output_format in ("txt_json", "txt_markdown"):
            meta = {
                "input_relative_path": job.relative_path,
                "input_hash": job.file_hash,
                "output_path": str(txt_path),
                "model": job.model,
                "base_url": cfg.base_url,
                "prompt_hash": dedup_key,
                "extraction_method": job.extraction_method,
                "chunk_count": chunk_count,
                "status": "COMPLETED",
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": datetime.utcnow().isoformat(),
                "usage": {
                    "input_tokens": api_result.input_tokens,
                    "output_tokens": api_result.output_tokens,
                    "total_tokens": api_result.total_tokens,
                },
                "error": None,
            }
            json_path = txt_path.with_suffix(".json")
            try:
                json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                json_path_str = str(json_path)
            except OSError:
                pass

        db.update_ai_file_job(job.id, {
            "status": AIJobStatus.COMPLETED.value,
            "output_txt_path": str(txt_path),
            "output_json_path": json_path_str,
            "rendered_prompt_hash": dedup_key,
            "chunk_count": chunk_count,
            "completed_chunk_count": chunk_count,
            "input_tokens": api_result.input_tokens,
            "output_tokens": api_result.output_tokens,
            "completed_at": datetime.utcnow().isoformat(),
            "error_code": None,
            "error_message": None,
        }, cfg.db_path)

        return AIJobStatus.COMPLETED

    def _fail_job(
        self,
        job_id: int,
        error_code: Optional[str],
        error_message: Optional[str],
    ) -> AIJobStatus:
        db.update_ai_file_job(job_id, {
            "status": AIJobStatus.FAILED.value,
            "error_code": error_code or "UNKNOWN_ERROR",
            "error_message": (error_message or "")[:1000],
            "completed_at": datetime.utcnow().isoformat(),
        }, self._cfg.db_path)
        return AIJobStatus.FAILED

    def _update_run_counters(self) -> None:
        db.update_ai_batch_run(self._cfg.run_id, {
            "completed_files": self.stats.completed,
            "failed_files": self.stats.failed,
            "skipped_files": self.stats.skipped,
            "actual_input_tokens": self.stats.input_tokens,
            "actual_output_tokens": self.stats.output_tokens,
        }, self._cfg.db_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _update_job_status(
    job_id: int,
    status: AIJobStatus,
    db_path: Path,
    **kwargs: Any,
) -> None:
    updates: dict[str, Any] = {"status": status.value}
    updates.update(kwargs)
    db.update_ai_file_job(job_id, updates, db_path)


# ── Factory helper ────────────────────────────────────────────────────────────

def create_run_and_jobs(
    input_folder: Path,
    output_folder: Path,
    discovered_files: list[DiscoveredFile],
    model: str,
    base_url: str,
    prompt_profile_id: Optional[int],
    config_kwargs: dict,
    db_path: Path = DB_PATH,
) -> int:
    """Create ai_batch_run + ai_file_jobs records and return run_id."""
    run_id = db.create_ai_batch_run(
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        model=model,
        base_url=base_url,
        prompt_profile_id=prompt_profile_id,
        path=db_path,
        **config_kwargs,
    )

    enabled_files = [f for f in discovered_files if f.enabled and not f.skip_reason]
    skipped_files = [f for f in discovered_files if not f.enabled or f.skip_reason]

    for df in enabled_files:
        file_hash = compute_file_hash(df.absolute_path)
        db.create_ai_file_job(
            run_id=run_id,
            relative_path=str(df.relative_path),
            absolute_input_path=str(df.absolute_path),
            input_filename=df.filename,
            extension=df.extension,
            mime_type=df.mime_type,
            file_size=df.file_size,
            file_hash=file_hash,
            file_group=df.file_group.value,
            extraction_method=df.extraction_method,
            model=model,
            prompt_profile_id=prompt_profile_id,
            path=db_path,
        )

    for df in skipped_files:
        db.create_ai_file_job(
            run_id=run_id,
            relative_path=str(df.relative_path),
            absolute_input_path=str(df.absolute_path),
            input_filename=df.filename,
            extension=df.extension,
            mime_type=df.mime_type,
            file_size=df.file_size,
            file_hash="",
            file_group=df.file_group.value,
            extraction_method="skipped",
            model=model,
            prompt_profile_id=prompt_profile_id,
            path=db_path,
        )
        db.update_ai_file_job(
            db.list_ai_file_jobs(run_id, db_path)[-1].id,
            {
                "status": AIJobStatus.SKIPPED.value,
                "error_code": "UNSUPPORTED_FILE_TYPE" if not df.is_supported else "CANCELLED_BY_USER",
                "error_message": df.skip_reason or "کاربر غیرفعال کرد",
            },
            db_path,
        )

    db.update_ai_batch_run(run_id, {
        "total_files": len(enabled_files) + len(skipped_files),
        "skipped_files": len(skipped_files),
    }, db_path)

    return run_id
