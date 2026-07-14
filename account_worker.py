"""
account_worker.py – Per-account async worker running inside its own Process.

Each AccountWorker:
  - Opens one NotebookLM client for its dedicated profile.
  - Manages an asyncio.Queue of job IDs.
  - Runs max_concurrent_jobs slot-workers simultaneously.
  - Each slot picks the next job as soon as it finishes the current one.
  - Sends WorkerEvents through a shared multiprocessing.Queue (single-writer).
  - Full idempotency: each step is skipped if already completed.

Entry point: run_account_worker()
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from multiprocessing import Queue as MPQueue
from pathlib import Path
from typing import Any, Optional

import database as db
from log_utils import install_redacting_filter
from models import Job, JobStatus, WorkerEvent
from notebook_service import (
    NotebookLMAuthError,
    NotebookLMRateLimitError,
    NotebookLMError,
    get_client,
)
from shared_source_service import (
    get_sources_for_notebook,
    is_source_already_uploaded,
    record_source_upload,
    record_source_upload_failed,
)
from settings import (
    AUDIO_SOURCE_SETTLE_SECONDS,
    DB_PATH,
    RETRY_DELAYS,
    USE_FAKE_CLIENT,
)

logger = logging.getLogger(__name__)


def _runtime_setting(db_path: Path, key: str, default: str) -> str:
    """Read an optional runtime setting without making workers depend on it."""
    try:
        return db.get_setting(key, default, db_path)
    except Exception:
        logger.debug("Using default for unavailable setting %s.", key)
        return default


# ─── Entry point ───────────────────────────────────────────────────────────────

def run_account_worker(
    account_id: int,
    profile_name: str,
    job_ids: list[int],
    event_queue: MPQueue,
    db_path_str: str,
    use_fake: bool,
    max_concurrency: int,
    fail_auth_after: Optional[int] = None,
) -> None:
    """Synchronous entry point called by multiprocessing.Process."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [%(name)s] [{profile_name}] %(message)s",
    )
    install_redacting_filter()

    db_path = Path(db_path_str)
    worker = AccountWorker(
        account_id=account_id,
        profile_name=profile_name,
        job_ids=job_ids,
        event_queue=event_queue,
        db_path=db_path,
        use_fake=use_fake,
        max_concurrency=max_concurrency,
        fail_auth_after=fail_auth_after,
    )
    asyncio.run(worker.run())


# ─── AccountWorker ─────────────────────────────────────────────────────────────

class AccountWorker:
    def __init__(
        self,
        account_id: int,
        profile_name: str,
        job_ids: list[int],
        event_queue: MPQueue,
        db_path: Path,
        use_fake: bool,
        max_concurrency: int = 3,
        fail_auth_after: Optional[int] = None,
    ):
        self.account_id   = account_id
        self.profile_name = profile_name
        self.initial_job_ids = list(job_ids)
        self.event_queue  = event_queue
        self.db_path      = db_path
        self.use_fake     = use_fake
        self.max_concurrency = max_concurrency
        self._fail_auth_after = fail_auth_after
        self._client      = None
        self._stop_event  = asyncio.Event()
        self._job_queue: asyncio.Queue[Optional[int]] = asyncio.Queue()

    async def run(self) -> None:
        logger.info("Worker starting: %d jobs, concurrency=%d", len(self.initial_job_ids), self.max_concurrency)

        results: list = []
        try:
            async with get_client(
                self.profile_name,
                use_fake=self.use_fake,
                fail_auth_after=self._fail_auth_after,
            ) as client:
                self._client = client

                for job_id in self.initial_job_ids:
                    await self._job_queue.put(job_id)
                for _ in range(self.max_concurrency):
                    await self._job_queue.put(None)  # poison pills

                slot_tasks = [
                    asyncio.create_task(self._slot_worker(i + 1))
                    for i in range(self.max_concurrency)
                ]
                results = await asyncio.gather(*slot_tasks, return_exceptions=True)

        except NotebookLMAuthError as exc:
            logger.error("[%s] Auth failed during client startup: %s", self.profile_name, exc)
            self._emit_account_event("AUTH_EXPIRED", str(exc))
            for job_id in self.initial_job_ids:
                self._emit_job_event(job_id, JobStatus.FAILED, error_message=f"Auth failed on startup: {exc}")
            return
        except NotebookLMRateLimitError as exc:
            logger.error(
                "[%s] Rate limited during client startup: %s",
                self.profile_name,
                exc,
            )
            self._emit_account_event("RATE_LIMITED", str(exc))
            for job_id in self.initial_job_ids:
                self._emit_job_event(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"Rate limited on startup: {exc}",
                )
            return
        except Exception as exc:
            logger.error("[%s] Worker failed to initialize client: %s", self.profile_name, exc)
            for job_id in self.initial_job_ids:
                self._emit_job_event(job_id, JobStatus.FAILED, error_message=f"Worker init failed: {exc}")
            return

        for exc in results:
            if isinstance(exc, Exception):
                logger.error("Unhandled slot exception: %s", exc)

        logger.info("Worker finished.")

    async def _slot_worker(self, slot_id: int) -> None:
        while True:
            job_id = await self._job_queue.get()
            if job_id is None:
                self._job_queue.task_done()
                break

            if self._stop_event.is_set():
                self._job_queue.task_done()
                continue

            try:
                await self._process_with_retry(job_id, slot_id)
            except Exception as exc:
                logger.exception("Slot %d unhandled error job %d: %s", slot_id, job_id, exc)
            finally:
                self._job_queue.task_done()

    async def _process_with_retry(self, job_id: int, slot_id: int) -> None:
        try:
            retry_count = max(
                0,
                int(
                    _runtime_setting(
                        self.db_path,
                        "retry_count",
                        str(len(RETRY_DELAYS)),
                    )
                ),
            )
        except (TypeError, ValueError):
            retry_count = len(RETRY_DELAYS)
        delays = list(RETRY_DELAYS[:retry_count])
        while len(delays) < retry_count:
            delays.append(min((delays[-1] if delays else 10) * 2, 300))
        attempt = 0

        while True:
            try:
                await self._process_job(job_id)
                return
            except NotebookLMAuthError as exc:
                self._emit_account_event("AUTH_EXPIRED", str(exc))
                self._emit_job_event(job_id, JobStatus.FAILED, error_message=str(exc))
                self._stop_event.set()
                return
            except NotebookLMRateLimitError as exc:
                self._emit_account_event("RATE_LIMITED", str(exc))
                self._emit_job_event(job_id, JobStatus.FAILED, error_message=str(exc))
                self._stop_event.set()
                return
            except Exception as exc:
                attempt += 1
                if attempt > len(delays):
                    logger.error("Job %d failed after %d attempts: %s", job_id, attempt, exc)
                    self._emit_job_event(job_id, JobStatus.FAILED, error_message=str(exc))
                    return
                wait = delays[attempt - 1]
                logger.warning("Job %d attempt %d failed, retry in %ds: %s", job_id, attempt, wait, exc)
                await asyncio.sleep(wait)

    # ── Core job processor (fully idempotent) ────────────────────────────────

    async def _process_job(self, job_id: int) -> None:
        job = db.get_job(job_id, self.db_path)
        if job is None:
            raise ValueError(f"Job {job_id} not found.")

        # Already completed
        if job.status == JobStatus.COMPLETED and job.downloaded_audio_path:
            if Path(job.downloaded_audio_path).exists():
                logger.info("Job %d already complete – skipping.", job_id)
                return

        # ── Step 1: Create Notebook ──────────────────────────────────────
        notebook_id = job.notebook_id
        if not notebook_id:
            self._emit_job_event(job_id, JobStatus.CREATING_NOTEBOOK)
            notebook_id = await self._client.create_notebook(
                self._notebook_name(job)
            )
            self._emit_job_event(job_id, JobStatus.CREATING_NOTEBOOK, notebook_id=notebook_id)

        # ── Step 2: Upload and confirm shared sources first ──────────────
        self._emit_job_event(
            job_id, JobStatus.UPLOADING_SHARED_SOURCES,
            notebook_id=notebook_id,
        )
        shared_source_ids: list[str] = []
        shared_sources = get_sources_for_notebook(
            job.project_id, job.chunk_id, self.account_id, self.db_path
        )
        for ss in shared_sources:
            existing_sid = is_source_already_uploaded(
                job_id, ss.id, ss.file_hash, self.db_path
            )
            try:
                if existing_sid:
                    logger.info(
                        "Shared source %d already uploaded for job %d; confirming readiness.",
                        ss.id,
                        job_id,
                    )
                    ss_source_id = existing_sid
                else:
                    ss_source_id = await self._client.upload_file(notebook_id, ss.file_path)
                await self._client.wait_for_source(notebook_id, ss_source_id)
                record_source_upload(
                    job_id, ss.id, ss.file_hash, ss_source_id, self.db_path
                )
                shared_source_ids.append(ss_source_id)
            except Exception as exc:
                record_source_upload_failed(
                    job_id, ss.id, ss.file_hash, str(exc), self.db_path
                )
                logger.error("Shared source %d was not ready: %s", ss.id, exc)
                raise

        # ── Step 3: Upload the chunk's main source ───────────────────────
        main_source_id = job.main_source_id
        if not main_source_id:
            self._emit_job_event(
                job_id,
                JobStatus.UPLOADING_MAIN_SOURCE,
                notebook_id=notebook_id,
            )
            chunks = db.get_chunks_for_project(job.project_id, self.db_path)
            pdf_path = next((c.pdf_path for c in chunks if c.id == job.chunk_id), "")
            if not pdf_path:
                raise ValueError(f"Chunk PDF not found for job {job_id}")
            main_source_id = await self._client.upload_file(notebook_id, pdf_path)
            self._emit_job_event(
                job_id,
                JobStatus.UPLOADING_MAIN_SOURCE,
                notebook_id=notebook_id,
                main_source_id=main_source_id,
            )

        # ── Step 4: Confirm the main source is ready ─────────────────────
        self._emit_job_event(job_id, JobStatus.WAITING_FOR_SOURCES, notebook_id=notebook_id)
        await self._client.wait_for_source(notebook_id, main_source_id)
        all_source_ids = [main_source_id, *shared_source_ids]

        # ── Step 5: Let NotebookLM settle, then generate audio ───────────
        artifact_id = job.artifact_id
        if not artifact_id:
            settle_seconds = (
                0.0 if self.use_fake else AUDIO_SOURCE_SETTLE_SECONDS
            )
            logger.info(
                "Job %d sources are ready; waiting %.0f seconds before audio generation.",
                job_id,
                settle_seconds,
            )
            await asyncio.sleep(settle_seconds)
            self._emit_job_event(job_id, JobStatus.GENERATING_AUDIO, notebook_id=notebook_id)
            prompt = self._get_prompt(job)
            if not job.prompt_rendered:
                db.update_job(job_id, {"prompt_rendered": prompt}, self.db_path)
            project = db.get_project(job.project_id, self.db_path)
            language = project.language if project else "fa"
            artifact_id = await self._client.generate_audio(
                notebook_id,
                source_ids=all_source_ids,
                prompt=prompt,
                language=language,
            )
            self._emit_job_event(
                job_id, JobStatus.GENERATING_AUDIO,
                notebook_id=notebook_id, artifact_id=artifact_id,
            )

        # ── Step 6/7: Wait for and download audio ────────────────────────
        audio_path = self._build_audio_path(job)
        audio_exists = (
            Path(audio_path).exists() and Path(audio_path).stat().st_size > 0
        )
        if not audio_exists:
            self._emit_job_event(
                job_id, JobStatus.WAITING_FOR_AUDIO,
                notebook_id=notebook_id, artifact_id=artifact_id,
            )
            await self._client.wait_for_audio(notebook_id, artifact_id)
            self._emit_job_event(
                job_id, JobStatus.DOWNLOADING_AUDIO,
                notebook_id=notebook_id, artifact_id=artifact_id,
            )
            await self._client.download_audio(notebook_id, artifact_id, audio_path)

        # ── Done ─────────────────────────────────────────────────────────
        self._emit_job_event(
            job_id, JobStatus.COMPLETED,
            notebook_id=notebook_id,
            main_source_id=main_source_id,
            artifact_id=artifact_id,
            downloaded_audio_path=audio_path,
        )
        cleanup_notebooks = (
            _runtime_setting(self.db_path, "cleanup_notebooks", "1") == "1"
        )
        if cleanup_notebooks and not self.use_fake:
            await self._client.delete_notebook(notebook_id)
            logger.info("Deleted completed NotebookLM notebook %s.", notebook_id)
        logger.info("Job %d completed → %s", job_id, audio_path)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _notebook_name(self, job: Job) -> str:
        return f"Part {job.chunk_index:03d} - Pages {job.start_page:03d}-{job.end_page:03d}"

    def _get_prompt(self, job: Job) -> str:
        # Use rendered prompt if available
        if job.prompt_rendered:
            return job.prompt_rendered
        project = db.get_project(job.project_id, self.db_path)
        if project and project.prompt_template:
            return self._render_prompt(project.prompt_template, job, project)
        from settings import DEFAULT_AUDIO_PROMPT
        return DEFAULT_AUDIO_PROMPT

    def _render_prompt(self, template: str, job: Job, project: Any) -> str:
        return (
            template
            .replace("{{project_name}}", project.name)
            .replace("{{original_filename}}", project.original_filename)
            .replace("{{chunk_index}}", str(job.chunk_index))
            .replace("{{start_page}}", str(job.start_page))
            .replace("{{end_page}}", str(job.end_page))
            .replace("{{page_count}}", str(job.end_page - job.start_page + 1))
            .replace("{{account_name}}", self.profile_name)
        )

    def _build_audio_path(self, job: Job) -> str:
        project = db.get_project(job.project_id, self.db_path)
        if project and project.output_dir:
            audio_dir = Path(project.output_dir) / "audio_original"
        else:
            from settings import PROJECTS_DIR
            audio_dir = PROJECTS_DIR / f"project_{job.project_id}" / "audio_original"
        audio_dir.mkdir(parents=True, exist_ok=True)
        return str(audio_dir / f"{job.chunk_index:03d}_pages_{job.start_page:03d}_{job.end_page:03d}.m4a")

    def _emit_job_event(
        self,
        job_id: int,
        status: JobStatus,
        *,
        notebook_id: Optional[str] = None,
        main_source_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        downloaded_audio_path: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self._emit({
            "event": "job_status",
            "account_id": self.account_id,
            "job_id": job_id,
            "status": status.value,
            "current_step": status.value,
            "notebook_id": notebook_id,
            "main_source_id": main_source_id,
            "artifact_id": artifact_id,
            "downloaded_audio_path": downloaded_audio_path,
            "error_message": error_message,
        })

    def _emit_account_event(self, status: str, message: str) -> None:
        self._emit({
            "event": "account_status",
            "account_id": self.account_id,
            "account_status": status,
            "message": message,
        })

    def _emit(self, payload: dict) -> None:
        try:
            self.event_queue.put_nowait(payload)
        except Exception as exc:
            logger.warning("Failed to send event: %s", exc)
