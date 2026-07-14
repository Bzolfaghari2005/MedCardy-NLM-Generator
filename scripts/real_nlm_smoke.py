"""Minimal paid/quota-consuming smoke test for the real NotebookLM pipeline."""
from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import sys
import tempfile
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database as db
from account_worker import AccountWorker
from log_utils import install_redacting_filter
from models import (
    AccountStatus,
    AllocationMode,
    AttachMode,
    SourceScope,
)
from notebook_service import get_client


def _create_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "NLM release smoke test. Summarize that the real audio pipeline works.",
    )
    document.save(path)
    document.close()


async def run(profile: str) -> None:
    with tempfile.TemporaryDirectory(prefix="nlm_real_smoke_") as temp:
        root = Path(temp)
        db_path = root / "smoke.sqlite3"
        output_dir = root / "output"
        chunk_path = root / "main.pdf"
        shared_path = root / "shared.md"
        _create_pdf(chunk_path)
        shared_path.write_text(
            "# Shared release context\n"
            "This is NLM version 1.0.0 and this source must be included.",
            encoding="utf-8",
        )

        db.init_db(db_path)
        account_id = db.create_account(profile, display_name="Real smoke", path=db_path)
        db.update_account_auth_status(account_id, AccountStatus.ACTIVE, db_path)
        project_id = db.create_project(
            name="NLM 1.0.0 real smoke",
            slug="nlm-real-smoke",
            prompt_template=(
                "Create a very short audio overview confirming both sources were read."
            ),
            language="en",
            allocation_mode=AllocationMode.EXACT,
            output_dir=str(output_dir),
            path=db_path,
        )
        chunk_id = db.create_chunk(
            project_id=project_id,
            chunk_index=1,
            start_page=1,
            end_page=1,
            pdf_path=str(chunk_path),
            pdf_hash="smoke-main",
            path=db_path,
        )
        db.upsert_allocation(project_id, account_id, 1, 1, path=db_path)
        shared_id = db.create_shared_source(
            scope=SourceScope.PROJECT,
            display_name="Release context",
            file_path=str(shared_path),
            original_filename=shared_path.name,
            file_hash="smoke-shared",
            mime_type="text/markdown",
            file_size=shared_path.stat().st_size,
            project_id=project_id,
            path=db_path,
        )
        db.attach_shared_source_to_project(
            project_id,
            shared_id,
            AttachMode.ALL_NOTEBOOKS,
            path=db_path,
        )
        job_id = db.create_job(project_id, chunk_id, account_id, path=db_path)

        worker = AccountWorker(
            account_id=account_id,
            profile_name=profile,
            job_ids=[job_id],
            event_queue=queue.Queue(),
            db_path=db_path,
            use_fake=False,
            max_concurrency=1,
        )
        async with get_client(profile, use_fake=False) as client:
            worker._client = client
            await worker._process_job(job_id)

        audio_files = list((output_dir / "audio_original").glob("*.m4a"))
        if len(audio_files) != 1 or audio_files[0].stat().st_size == 0:
            raise RuntimeError("Real smoke test did not produce a valid M4A file.")
        upload = db.get_source_upload(job_id, shared_id, db_path)
        if upload is None or upload.source_id is None:
            raise RuntimeError("Shared-source upload was not recorded.")

        print(
            "REAL_NLM_SMOKE_OK "
            f"profile={profile} audio_bytes={audio_files[0].stat().st_size} "
            f"shared_source_id={upload.source_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="account_01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    install_redacting_filter()
    asyncio.run(run(args.profile))


if __name__ == "__main__":
    main()
