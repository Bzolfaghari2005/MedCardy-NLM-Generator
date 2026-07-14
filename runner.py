"""
runner.py – Standalone job runner.

Run with:   python runner.py [--project-id N] [--fake]

Responsibilities:
  - Write PID to data/runtime/runner.pid at startup.
  - Initialize DB.
  - For each RUNNING / PENDING project, start the orchestrator.
  - Optionally run conversion and transcription jobs after audio download.
  - Poll DB every few seconds to pick up newly created projects.
  - Remove PID file on exit.

Streamlit launches this via subprocess.Popen; it runs independently.
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from log_utils import install_redacting_filter

# Windows multiprocessing guard
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

from settings import (
    DB_PATH, LOG_LEVEL, RUNNER_LOG_FILE, RUNNER_PID_FILE,
    USE_FAKE_CLIENT,
)

# ─── Logging ──────────────────────────────────────────────────────────────────

RUNNER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RUNNER_LOG_FILE), encoding="utf-8"),
    ],
)
install_redacting_filter()
logger = logging.getLogger("runner")
_post_processing_projects: set[int] = set()
_post_processing_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# PID management
# ══════════════════════════════════════════════════════════════════════════════

def _write_pid() -> None:
    RUNNER_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNNER_PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    RUNNER_PID_FILE.unlink(missing_ok=True)


def is_runner_alive() -> bool:
    """Check if another runner process is already active."""
    if not RUNNER_PID_FILE.exists():
        return False
    try:
        pid = int(RUNNER_PID_FILE.read_text().strip())
        if pid == os.getpid():
            return False
        # Check if process is actually running
        if sys.platform == "win32":
            import ctypes
            SYNCHRONIZE = 0x100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _maybe_convert_audio(project_id: int, path: Path) -> None:
    """Create audio_conversion records for completed jobs that have audio."""
    import database as db
    from models import JobStatus

    project = db.get_project(project_id, path)
    if not project or not project.auto_convert_to_mp3:
        return

    jobs = db.get_jobs_for_project(project_id, path)
    for job in jobs:
        if job.status != JobStatus.COMPLETED:
            continue
        if not job.downloaded_audio_path:
            continue
        if job.converted_mp3_path and Path(job.converted_mp3_path).exists():
            continue

        audio_path = Path(job.downloaded_audio_path)
        if not audio_path.exists():
            continue

        mp3_dir = Path(project.output_dir) / "audio_mp3"
        mp3_path = mp3_dir / (audio_path.stem + ".mp3")

        from audio_service import convert_to_mp3
        ok, msg = convert_to_mp3(
            audio_path, mp3_path,
            bitrate=project.mp3_bitrate,
            overwrite=False,
            keep_original=project.keep_original_audio,
        )
        if ok:
            db.update_job(job.id, {"converted_mp3_path": str(mp3_path)}, path)
            logger.info("Converted %s → %s", audio_path.name, mp3_path.name)
        else:
            logger.warning("Conversion failed for job %d: %s", job.id, msg)


def _maybe_transcribe(project_id: int, path: Path) -> None:
    """Create transcription records for completed jobs if auto_transcribe is on."""
    import database as db
    from models import JobStatus, TranscriptionStatus

    project = db.get_project(project_id, path)
    if not project or not project.auto_transcribe:
        return

    jobs = db.get_jobs_for_project(project_id, path)
    for job in jobs:
        if job.status != JobStatus.COMPLETED:
            continue

        # Prefer MP3 for transcription, fallback to M4A
        audio_path = None
        if job.converted_mp3_path and Path(job.converted_mp3_path).exists():
            audio_path = Path(job.converted_mp3_path)
        elif job.downloaded_audio_path and Path(job.downloaded_audio_path).exists():
            audio_path = Path(job.downloaded_audio_path)

        if not audio_path:
            continue

        if job.transcript_txt_path and Path(job.transcript_txt_path).exists():
            continue

        from shared_source_service import compute_file_hash
        from transcription_service import run_transcription_job

        file_hash = compute_file_hash(audio_path)
        tr_id = db.create_transcription(
            input_path=str(audio_path),
            input_hash=file_hash,
            model_name=project.whisper_model,
            language=project.whisper_language,
            project_id=project_id,
            job_id=job.id,
            path=path,
        )
        ok = run_transcription_job(tr_id, path=path)
        if ok:
            # Update job with transcript path
            trs = db.list_transcriptions(project_id, path)
            tr = next((t for t in trs if t.id == tr_id), None)
            if tr and tr.output_txt_path:
                db.update_job(job.id, {"transcript_txt_path": tr.output_txt_path}, path)


def _start_post_processing(project_id: int, path: Path) -> None:
    """Run conversion/transcription without blocking the runner poll loop."""
    with _post_processing_lock:
        if project_id in _post_processing_projects:
            return
        _post_processing_projects.add(project_id)

    def _run() -> None:
        try:
            _maybe_convert_audio(project_id, path)
            _maybe_transcribe(project_id, path)
        except Exception:
            logger.exception("Post-processing failed for project %d.", project_id)
        finally:
            with _post_processing_lock:
                _post_processing_projects.discard(project_id)

    threading.Thread(
        target=_run,
        daemon=True,
        name=f"post-process-{project_id}",
    ).start()


# ══════════════════════════════════════════════════════════════════════════════
# Main runner loop
# ══════════════════════════════════════════════════════════════════════════════

def _reconcile_orchestrators(
    projects,
    orchestrators: dict,
    *,
    db_module,
    orchestrator_factory,
    db_path: Path,
) -> None:
    """Bring the in-memory orchestrators in sync with current project state."""
    from models import ProjectStatus

    visible_ids = {project.id for project in projects}
    for stale_id in set(orchestrators) - visible_ids:
        orchestrators.pop(stale_id).stop(update_project_status=False)

    for project in projects:
        orchestrator = orchestrators.get(project.id)
        active = project.status in (ProjectStatus.RUNNING, ProjectStatus.PENDING)

        if orchestrator is not None and not active:
            orchestrator.stop(update_project_status=False)
            orchestrators.pop(project.id, None)
            if project.status in (ProjectStatus.COMPLETED, ProjectStatus.FAILED):
                _start_post_processing(project.id, db_path)
            continue

        if orchestrator is not None and orchestrator.is_finished():
            # The consumer has already written the final project state. Evict
            # now and use a fresh DB snapshot on the next poll before restarting.
            orchestrators.pop(project.id, None)
            continue

        if active and orchestrator is None:
            orchestrator = orchestrator_factory()
            orchestrators[project.id] = orchestrator
            try:
                orchestrator.start(project.id)
                logger.info("Started orchestrator for project %d.", project.id)
            except Exception as exc:
                orchestrators.pop(project.id, None)
                logger.error("Failed to start project %d: %s", project.id, exc)
                db_module.update_project_status(
                    project.id, ProjectStatus.FAILED, db_path
                )


def run(project_id: int | None = None, use_fake: bool = USE_FAKE_CLIENT) -> None:
    import database as db
    from orchestrator import ParallelJobOrchestrator

    db.init_db(DB_PATH)
    ffmpeg_override = db.get_setting("ffmpeg_path", "", DB_PATH).strip()
    if ffmpeg_override:
        os.environ["FFMPEG_PATH"] = ffmpeg_override
    configured_level = db.get_setting("log_level", LOG_LEVEL, DB_PATH).upper()
    level = getattr(logging, configured_level, logging.INFO)
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)

    orchestrators: dict[int, ParallelJobOrchestrator] = {}

    logger.info("Runner started. PID=%d use_fake=%s", os.getpid(), use_fake)

    try:
        while True:
            projects = db.list_projects(DB_PATH)
            if project_id is not None:
                projects = [project for project in projects if project.id == project_id]
            _reconcile_orchestrators(
                projects,
                orchestrators,
                db_module=db,
                orchestrator_factory=lambda: ParallelJobOrchestrator(
                    db_path=DB_PATH, use_fake=use_fake
                ),
                db_path=DB_PATH,
            )

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Runner interrupted.")
    finally:
        for orch in orchestrators.values():
            try:
                orch.stop()
            except Exception:
                pass
        logger.info("Runner stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLM Job Runner")
    parser.add_argument("--project-id", type=int, default=None)
    parser.add_argument("--fake", action="store_true", default=False)
    args = parser.parse_args()

    if is_runner_alive():
        print("Runner is already running. Check data/runtime/runner.pid")
        sys.exit(1)

    _write_pid()
    atexit.register(_remove_pid)

    # Handle Ctrl+C / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Signal %d received, shutting down.", signum)
        _remove_pid()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    use_fake = args.fake or USE_FAKE_CLIENT
    run(project_id=args.project_id, use_fake=use_fake)
