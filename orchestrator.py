"""
orchestrator.py – Main process coordinator.

ParallelJobOrchestrator:
  1. Reads project allocations from DB (per-account quotas).
  2. Spawns one multiprocessing.Process per account (with correct concurrency).
  3. All processes start simultaneously.
  4. A daemon thread consumes events from the shared Queue and writes to SQLite.
  5. On AUTH_EXPIRED / RATE_LIMITED, reassigns pending jobs in FLEXIBLE mode.
"""
from __future__ import annotations

import logging
import queue
import threading
from multiprocessing import Process, Queue as MPQueue
from pathlib import Path
from typing import Any, Optional

import database as db
from account_worker import run_account_worker
from allocation_service import (
    distribute_chunks,
    apply_distribution,
    get_jobs_per_account,
    reassign_from_failed_account,
)
from models import AccountStatus, AllocationMode, JobStatus, ProjectStatus
from settings import DB_PATH, USE_FAKE_CLIENT

logger = logging.getLogger(__name__)


class ParallelJobOrchestrator:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        use_fake: bool = USE_FAKE_CLIENT,
    ):
        self.db_path  = db_path
        self.use_fake = use_fake

        self._event_queue: MPQueue       = MPQueue()
        self._processes: dict[int, list[Process]] = {}   # account_id → [Process, ...]
        self._project_id: Optional[int]  = None
        self._stop_flag   = threading.Event()
        self._consumer_thread: Optional[threading.Thread] = None
        self._overflow_job_ids: set[int] = set()

    # ─── Public API ─────────────────────────────────────────────────────────

    def start(self, project_id: int) -> None:
        """Distribute chunks and launch all account processes."""
        if self.is_running():
            raise RuntimeError("Orchestrator is already running.")
        self._project_id = project_id
        self._stop_flag.clear()
        self._overflow_job_ids.clear()

        project = db.get_project(project_id, self.db_path)
        if not project:
            raise ValueError(f"Project {project_id} not found.")

        allocation_mode = project.allocation_mode

        # Distribute chunks → create jobs (idempotent)
        distribution = distribute_chunks(project_id, allocation_mode, self.db_path)
        apply_distribution(project_id, distribution, self.db_path)

        # Gather jobs per account
        jobs_by_account = get_jobs_per_account(project_id, self.db_path)

        if not jobs_by_account:
            logger.info("No pending jobs for project %d.", project_id)
            jobs = db.get_jobs_for_project(project_id, self.db_path)
            if any(job.status == JobStatus.FAILED for job in jobs):
                db.update_project_status(project_id, ProjectStatus.FAILED, self.db_path)
            elif jobs and all(
                job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED)
                for job in jobs
            ):
                db.update_project_status(project_id, ProjectStatus.COMPLETED, self.db_path)
            else:
                db.update_project_status(project_id, ProjectStatus.FAILED, self.db_path)
            return

        allocations = db.get_allocations_for_project(project_id, self.db_path)
        alloc_map   = {a.account_id: a for a in allocations}

        db.update_project_status(project_id, ProjectStatus.RUNNING, self.db_path)

        self._processes.clear()
        for account_id, jobs in jobs_by_account.items():
            if not jobs:
                continue

            account = db.get_account(account_id, self.db_path)
            if not account:
                continue

            alloc = alloc_map.get(account_id)
            max_concurrency = alloc.max_concurrent_jobs if alloc else 3

            job_ids = [j.id for j in jobs]

            p = Process(
                target=run_account_worker,
                args=(
                    account_id,
                    account.profile_name,
                    job_ids,
                    self._event_queue,
                    str(self.db_path),
                    self.use_fake,
                    max_concurrency,
                ),
                daemon=True,
                name=f"worker-{account.profile_name}",
            )
            self._processes.setdefault(account_id, []).append(p)

        for procs in self._processes.values():
            for p in procs:
                p.start()
        logger.info("Started %d account processes.", sum(len(v) for v in self._processes.values()))

        self._start_event_consumer()

    def stop(self, update_project_status: bool = True) -> None:
        self._stop_flag.set()
        for procs in self._processes.values():
            for p in procs:
                if p.is_alive():
                    p.terminate()
        for procs in self._processes.values():
            for p in procs:
                p.join(timeout=5)
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=3)
        if update_project_status and self._project_id:
            db.update_project_status(self._project_id, ProjectStatus.STOPPED, self.db_path)

    def pause_new_jobs(self) -> None:
        self._stop_flag.set()

    def is_running(self) -> bool:
        return any(p.is_alive() for procs in self._processes.values() for p in procs)

    def is_finished(self) -> bool:
        """Return true once workers and their event consumer have exited."""
        consumer_done = (
            self._consumer_thread is None or not self._consumer_thread.is_alive()
        )
        return not self.is_running() and consumer_done

    def retry_failed_jobs(self, project_id: int) -> None:
        failed = db.get_failed_jobs(project_id, self.db_path)
        for job in failed:
            db.reset_job_for_retry(job.id, path=self.db_path)
        if failed:
            db.update_project_status(project_id, ProjectStatus.PENDING, self.db_path)
            # If the previous consumer is fully gone this object can restart
            # immediately. Otherwise the long-lived runner will evict it after
            # the consumer drains and launch a fresh orchestrator next poll.
            if self.is_finished():
                self.start(project_id)

    # ─── Event consumer (daemon thread) ─────────────────────────────────────

    def _start_event_consumer(self) -> None:
        self._consumer_thread = threading.Thread(
            target=self._consume_events,
            name="event-consumer",
            daemon=True,
        )
        self._consumer_thread.start()

    def _consume_events(self) -> None:
        logger.info("Event consumer started.")
        while not self._stop_flag.is_set() or self.is_running():
            try:
                event: dict[str, Any] = self._event_queue.get(timeout=1.0)
            except Exception:
                if not self.is_running() and self._event_queue.empty():
                    break
                continue
            self._handle_event(event)

        while True:
            try:
                event = self._event_queue.get_nowait()
                self._handle_event(event)
            except Exception:
                break

        logger.info("Event consumer finished.")
        self._check_project_completion()

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("event")
        if etype == "job_status":
            self._handle_job_status(event)
        elif etype == "account_status":
            self._handle_account_status(event)

    def _handle_job_status(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if job_id is None:
            return
        status_str = event.get("status")
        try:
            status = JobStatus(status_str)
        except (ValueError, TypeError):
            logger.warning("Unknown job status '%s'", status_str)
            return

        db.update_job_status(
            job_id, status,
            current_step=event.get("current_step"),
            notebook_id=event.get("notebook_id"),
            main_source_id=event.get("main_source_id"),
            artifact_id=event.get("artifact_id"),
            downloaded_audio_path=event.get("downloaded_audio_path"),
            error_message=event.get("error_message"),
            increment_attempt=(status == JobStatus.CREATING_NOTEBOOK),
            path=self.db_path,
        )

        if status == JobStatus.COMPLETED and self._project_id:
            account_id = event.get("account_id")
            if account_id:
                db.increment_allocation_counter(
                    self._project_id, account_id, "completed_jobs_count", self.db_path
                )

        if status == JobStatus.FAILED and self._project_id:
            account_id = event.get("account_id")
            if account_id:
                db.increment_allocation_counter(
                    self._project_id, account_id, "failed_jobs_count", self.db_path
                )

    def _handle_account_status(self, event: dict[str, Any]) -> None:
        account_id = event.get("account_id")
        status_str = event.get("account_status")
        if not account_id or not status_str:
            return
        try:
            new_status = AccountStatus(status_str)
        except ValueError:
            return

        db.update_account_auth_status(account_id, new_status, self.db_path)
        logger.warning("Account %d → %s", account_id, status_str)

        if new_status in (AccountStatus.AUTH_EXPIRED, AccountStatus.RATE_LIMITED):
            if self._project_id:
                for process in self._processes.get(account_id, []):
                    if process.is_alive():
                        process.terminate()
                project = db.get_project(self._project_id, self.db_path)
                mode = project.allocation_mode if project else AllocationMode.EXACT
                before = {
                    job.id for job in db.get_jobs_for_project(
                        self._project_id, self.db_path
                    )
                    if job.account_id == account_id
                    and job.status in (JobStatus.PENDING, JobStatus.ASSIGNED)
                }
                reassigned = reassign_from_failed_account(
                    self._project_id, account_id, mode, self.db_path
                )
                if reassigned > 0:
                    self._start_overflow_processes(
                        self._project_id, account_id, before
                    )

    def _start_overflow_processes(
        self,
        project_id: int,
        failed_account_id: int,
        reassigned_job_ids: set[int],
    ) -> None:
        """Start additional processes for jobs reassigned from a failed account."""
        jobs_by_account = get_jobs_per_account(project_id, self.db_path)
        for account_id, jobs in jobs_by_account.items():
            jobs = [
                job for job in jobs
                if job.id in reassigned_job_ids
                and job.id not in self._overflow_job_ids
            ]
            if account_id == failed_account_id or not jobs:
                continue
            account = db.get_account(account_id, self.db_path)
            if not account:
                continue
            alloc_list = db.get_allocations_for_project(project_id, self.db_path)
            alloc = next((a for a in alloc_list if a.account_id == account_id), None)
            max_concurrency = alloc.max_concurrent_jobs if alloc else 3

            p = Process(
                target=run_account_worker,
                args=(
                    account_id,
                    account.profile_name,
                    [j.id for j in jobs],
                    self._event_queue,
                    str(self.db_path),
                    self.use_fake,
                    max_concurrency,
                ),
                daemon=True,
                name=f"worker-{account.profile_name}-overflow",
            )
            self._processes.setdefault(account_id, []).append(p)
            self._overflow_job_ids.update(j.id for j in jobs)
            p.start()
            logger.info("Started overflow process for account %d.", account_id)

    def _check_project_completion(self) -> None:
        if self._project_id is None:
            return
        jobs = db.get_jobs_for_project(self._project_id, self.db_path)
        if not jobs:
            return
        statuses = {j.status for j in jobs}
        terminal = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        if statuses <= terminal:
            if JobStatus.FAILED in statuses:
                db.update_project_status(self._project_id, ProjectStatus.FAILED, self.db_path)
            else:
                db.update_project_status(self._project_id, ProjectStatus.COMPLETED, self.db_path)
            logger.info("Project %d finished with statuses: %s", self._project_id, statuses)
