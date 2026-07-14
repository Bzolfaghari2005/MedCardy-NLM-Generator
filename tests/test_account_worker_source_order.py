import asyncio
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from account_worker import AccountWorker
from models import JobStatus


def test_shared_sources_are_ready_before_main_source_and_audio(tmp_path: Path) -> None:
    trace: list[str] = []
    job = SimpleNamespace(
        id=1,
        project_id=10,
        chunk_id=20,
        notebook_id=None,
        main_source_id=None,
        artifact_id=None,
        downloaded_audio_path=None,
        status=JobStatus.PENDING,
        prompt_rendered="prompt",
        chunk_index=1,
        start_page=1,
        end_page=2,
    )
    shared = SimpleNamespace(id=30, file_hash="hash", file_path="shared.pdf")
    chunk = SimpleNamespace(id=20, pdf_path="main.pdf")
    project = SimpleNamespace(language="fa", output_dir=str(tmp_path))

    class RecordingClient:
        async def create_notebook(self, name: str) -> str:
            trace.append("create")
            return "notebook"

        async def upload_file(self, notebook_id: str, file_path: str) -> str:
            source_id = "shared-id" if file_path == "shared.pdf" else "main-id"
            trace.append(f"upload:{source_id}")
            return source_id

        async def wait_for_source(self, notebook_id: str, source_id: str) -> None:
            trace.append(f"ready:{source_id}")

        async def generate_audio(
            self,
            notebook_id: str,
            source_ids: list[str],
            prompt: str,
            language: str,
        ) -> str:
            trace.append(f"generate:{','.join(source_ids)}")
            return "artifact"

        async def wait_for_audio(self, notebook_id: str, artifact_id: str) -> None:
            trace.append("audio-ready")

        async def download_audio(
            self, notebook_id: str, artifact_id: str, output_path: str
        ) -> None:
            Path(output_path).write_bytes(b"audio")

    async def record_sleep(seconds: float) -> None:
        trace.append(f"settle:{seconds}")

    worker = AccountWorker(
        account_id=1,
        profile_name="test",
        job_ids=[1],
        event_queue=queue.Queue(),
        db_path=tmp_path / "test.db",
        use_fake=False,
    )
    worker._client = RecordingClient()

    with (
        patch("account_worker.db.get_job", return_value=job),
        patch("account_worker.db.get_chunks_for_project", return_value=[chunk]),
        patch("account_worker.db.get_project", return_value=project),
        patch("account_worker.get_sources_for_notebook", return_value=[shared]),
        patch("account_worker.is_source_already_uploaded", return_value=None),
        patch("account_worker.record_source_upload"),
        patch("account_worker._runtime_setting", return_value="0"),
        patch("account_worker.AUDIO_SOURCE_SETTLE_SECONDS", 60.0),
        patch("account_worker.asyncio.sleep", side_effect=record_sleep),
    ):
        asyncio.run(worker._process_job(job.id))

    assert trace[:6] == [
        "create",
        "upload:shared-id",
        "ready:shared-id",
        "upload:main-id",
        "ready:main-id",
        "settle:60.0",
    ]
    assert trace[6] == "generate:main-id,shared-id"
