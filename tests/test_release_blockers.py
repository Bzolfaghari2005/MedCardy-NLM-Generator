import asyncio
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import database as db
from account_worker import AccountWorker
from allocation_service import (
    apply_distribution,
    distribute_chunks,
    validate_project_preflight,
)
from models import (
    AccountStatus,
    AttachMode,
    JobStatus,
    ProjectStatus,
    SourceScope,
)
from notebook_service import (
    NotebookLMRateLimitError,
    RealNotebookClient,
)
from orchestrator import ParallelJobOrchestrator
from runner import _reconcile_orchestrators


def _project_db(tmp_path: Path) -> tuple[Path, int, int, int]:
    db_path = tmp_path / "test.sqlite3"
    db.init_db(db_path)
    account_id = db.create_account("active", path=db_path)
    db.update_account_auth_status(account_id, AccountStatus.ACTIVE, db_path)
    project_id = db.create_project("Project", "project", path=db_path)
    pdf_path = tmp_path / "chunk.pdf"
    pdf_path.write_bytes(b"%PDF")
    chunk_id = db.create_chunk(
        project_id, 1, 1, 1, str(pdf_path), path=db_path
    )
    db.upsert_allocation(project_id, account_id, 1, 1, path=db_path)
    return db_path, project_id, account_id, chunk_id


def test_distribution_and_retry_reset_are_idempotent(tmp_path: Path) -> None:
    db_path, project_id, account_id, _ = _project_db(tmp_path)
    distribution = distribute_chunks(project_id, path=db_path)

    apply_distribution(project_id, distribution, db_path)
    apply_distribution(project_id, distribution, db_path)

    jobs = db.get_jobs_for_project(project_id, db_path)
    allocation = db.get_allocations_for_project(project_id, db_path)[0]
    assert len(jobs) == 1
    assert allocation.assigned_jobs_count == 1

    job = jobs[0]
    db.update_job_status(
        job.id,
        JobStatus.FAILED,
        artifact_id="stale-artifact",
        error_message="rate limited",
        path=db_path,
    )
    assert db.get_pending_jobs(project_id, db_path) == []

    db.reset_job_for_retry(job.id, path=db_path)
    reset = db.get_job(job.id, db_path)
    assert reset is not None
    assert reset.status == JobStatus.PENDING
    assert reset.artifact_id is None
    assert reset.error_message is None


def test_shared_source_attachment_is_unique(tmp_path: Path) -> None:
    db_path, project_id, _, _ = _project_db(tmp_path)
    shared_path = tmp_path / "shared.md"
    shared_path.write_text("shared", encoding="utf-8")
    source_id = db.create_shared_source(
        scope=SourceScope.PROJECT,
        display_name="Shared",
        file_path=str(shared_path),
        original_filename="shared.md",
        file_hash="hash",
        mime_type="text/markdown",
        file_size=6,
        project_id=project_id,
        path=db_path,
    )

    first = db.attach_shared_source_to_project(
        project_id, source_id, AttachMode.ALL_NOTEBOOKS, path=db_path
    )
    second = db.attach_shared_source_to_project(
        project_id, source_id, AttachMode.ALL_NOTEBOOKS, path=db_path
    )

    assert first == second
    assert len(db.get_project_source_links(project_id, db_path)) == 1


def test_project_preflight_reports_missing_pdf_and_active_deficit(
    tmp_path: Path,
) -> None:
    db_path, project_id, account_id, chunk_id = _project_db(tmp_path)
    db.update_chunk(chunk_id, {"pdf_path": str(tmp_path / "missing.pdf")}, db_path)
    db.upsert_allocation(project_id, account_id, 0, 1, path=db_path)

    report = validate_project_preflight(project_id, db_path)

    assert report["can_start"] is False
    assert report["missing_chunk_ids"] == [chunk_id]
    assert report["allocation_deficit"] == 1
    assert report["active_account_count"] == 1
    assert report["source_counts"][chunk_id] == 1


def test_artifact_failures_are_rate_limits(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = RealNotebookClient("test")
        artifacts = SimpleNamespace(
            generate_audio=AsyncMock(
                return_value=SimpleNamespace(
                    task_id="", status="failed", error="daily quota"
                )
            ),
            wait_for_completion=AsyncMock(return_value=None),
            download_audio=AsyncMock(
                side_effect=RuntimeError("No completed Audio Overview found")
            ),
        )
        client._client = SimpleNamespace(artifacts=artifacts)

        try:
            await client.generate_audio("nb", ["source"], "prompt")
            raise AssertionError("empty task id should fail")
        except NotebookLMRateLimitError:
            pass

        try:
            await client.wait_for_audio("nb", "artifact")
            raise AssertionError("empty poll result should fail")
        except NotebookLMRateLimitError:
            pass

        try:
            await client.download_audio(
                "nb", "artifact", str(tmp_path / "audio.m4a")
            )
            raise AssertionError("missing completed audio should fail")
        except NotebookLMRateLimitError:
            pass

    asyncio.run(exercise())


def test_worker_does_not_generically_retry_rate_limits(tmp_path: Path) -> None:
    worker = AccountWorker(
        account_id=7,
        profile_name="limited",
        job_ids=[11],
        event_queue=queue.Queue(),
        db_path=tmp_path / "unused.sqlite3",
        use_fake=True,
    )
    worker._process_job = AsyncMock(
        side_effect=NotebookLMRateLimitError("daily quota")
    )

    asyncio.run(worker._process_with_retry(11, 1))

    assert worker._process_job.await_count == 1
    events = []
    while not worker.event_queue.empty():
        events.append(worker.event_queue.get_nowait())
    assert any(
        event.get("account_status") == "RATE_LIMITED" for event in events
    )
    assert any(
        event.get("job_id") == 11 and event.get("status") == "FAILED"
        for event in events
    )


def test_runner_evicts_finished_and_stops_paused_without_status_rewrite(
    tmp_path: Path,
) -> None:
    class FakeOrchestrator:
        def __init__(self, finished: bool = False):
            self.finished = finished
            self.started = []
            self.stop_args = []

        def start(self, project_id: int) -> None:
            self.started.append(project_id)

        def stop(self, update_project_status: bool = True) -> None:
            self.stop_args.append(update_project_status)

        def is_finished(self) -> bool:
            return self.finished

    db_module = SimpleNamespace(update_project_status=lambda *args: None)
    running = SimpleNamespace(id=1, status=ProjectStatus.RUNNING)
    finished = FakeOrchestrator(finished=True)
    orchestrators = {1: finished}
    created = []

    def factory():
        orchestrator = FakeOrchestrator()
        created.append(orchestrator)
        return orchestrator

    _reconcile_orchestrators(
        [running],
        orchestrators,
        db_module=db_module,
        orchestrator_factory=factory,
        db_path=tmp_path / "db.sqlite3",
    )
    assert orchestrators == {}

    _reconcile_orchestrators(
        [running],
        orchestrators,
        db_module=db_module,
        orchestrator_factory=factory,
        db_path=tmp_path / "db.sqlite3",
    )
    assert created[0].started == [1]

    paused = SimpleNamespace(id=1, status=ProjectStatus.PAUSED)
    _reconcile_orchestrators(
        [paused],
        orchestrators,
        db_module=db_module,
        orchestrator_factory=factory,
        db_path=tmp_path / "db.sqlite3",
    )
    assert created[0].stop_args == [False]
    assert orchestrators == {}


def test_retry_failed_jobs_marks_project_pending_and_restarts_when_idle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    orchestrator = ParallelJobOrchestrator(
        db_path=tmp_path / "unused.sqlite3", use_fake=True
    )
    failed = [SimpleNamespace(id=4), SimpleNamespace(id=5)]
    reset_ids = []
    statuses = []
    starts = []
    monkeypatch.setattr(
        "orchestrator.db.get_failed_jobs", lambda *args: failed
    )
    monkeypatch.setattr(
        "orchestrator.db.reset_job_for_retry",
        lambda job_id, **kwargs: reset_ids.append(job_id),
    )
    monkeypatch.setattr(
        "orchestrator.db.update_project_status",
        lambda project_id, status, path: statuses.append(status),
    )
    monkeypatch.setattr(orchestrator, "is_finished", lambda: True)
    monkeypatch.setattr(orchestrator, "start", lambda project_id: starts.append(project_id))

    orchestrator.retry_failed_jobs(9)

    assert reset_ids == [4, 5]
    assert statuses == [ProjectStatus.PENDING]
    assert starts == [9]
