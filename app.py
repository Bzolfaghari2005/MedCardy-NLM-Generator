"""
app.py – Streamlit UI for NLM Audio Generator.

Run with: streamlit run app.py
"""
from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

multiprocessing.freeze_support()

# ─── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="NLM Audio Generator",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

import database as db
from database import init_db
from models import (
    AccountStatus, AccountType, AllocationMode,
    AttachMode, JobStatus, ProjectStatus, SourceScope,
)
from settings import (
    APP_VERSION,
    DB_PATH,
    RUNNER_LOG_FILE,
    RUNNER_PID_FILE,
    UI_REFRESH_INTERVAL,
    daily_job_quota_for,
)

init_db(DB_PATH)

# Reset any transcriptions that were left in RUNNING state after a crash/restart
def _reset_stale_transcriptions() -> None:
    from models import TranscriptionStatus
    try:
        stale = [
            t for t in db.list_transcriptions(path=DB_PATH)
            if t.status == TranscriptionStatus.RUNNING
        ]
        for t in stale:
            db.update_transcription(t.id, {
                "status": TranscriptionStatus.PENDING.value,
                "progress": 0.0,
                "started_at": None,
            }, DB_PATH)
    except Exception:
        pass

_reset_stale_transcriptions()


# ══════════════════════════════════════════════════════════════════════════════
# Background transcription worker  (module-level — survives st.rerun())
# ══════════════════════════════════════════════════════════════════════════════

_tr_lock = threading.Lock()

# tr_id → current progress float [0.0 – 1.0]  (written by worker thread)
_tr_live_progress: dict[int, float] = {}

# active worker thread (one at a time)
_tr_worker_thread: Optional[threading.Thread] = None


def _transcription_worker(
    tr_ids: list[int],
    output_srt: bool,
    output_vtt: bool,
    output_json: bool,
    beam_size: int,
    include_header: bool,
    include_timestamps: bool,
) -> None:
    """Background thread — process each transcription job sequentially."""
    from transcription_service import transcribe_file
    from models import TranscriptionStatus
    from settings import TRANSCRIPTIONS_DIR

    for tr_id in tr_ids:
        transcriptions = db.list_transcriptions(path=DB_PATH)
        tr = next((t for t in transcriptions if t.id == tr_id), None)
        if tr is None:
            continue
        if tr.status == TranscriptionStatus.COMPLETED:
            with _tr_lock:
                _tr_live_progress[tr_id] = 1.0
            continue

        db.update_transcription(tr_id, {
            "status": TranscriptionStatus.RUNNING.value,
            "started_at": datetime.utcnow().isoformat(),
        }, DB_PATH)

        input_path = Path(tr.input_path)

        # Build output dir — use project folder when available
        if tr.project_id:
            proj = db.get_project(tr.project_id, DB_PATH)
            out_dir = Path(proj.output_dir) / "transcripts" if proj else TRANSCRIPTIONS_DIR
        else:
            out_dir = TRANSCRIPTIONS_DIR

        # DB progress throttle: only write when crossing a 5 % boundary
        _last_db_pct: list[int] = [-1]

        def _progress_cb(p: float, _id: int = tr_id) -> None:
            with _tr_lock:
                _tr_live_progress[_id] = p
            pct = int(p * 20)          # 5 % buckets
            if pct > _last_db_pct[0]:
                _last_db_pct[0] = pct
                try:
                    db.update_transcription(_id, {"progress": round(p, 3)}, DB_PATH)
                except Exception:
                    pass

        try:
            result = transcribe_file(
                input_path=input_path,
                output_dir=out_dir,
                model_name=tr.model_name,
                language=tr.language or "",
                device=tr.device,
                compute_type=tr.compute_type,
                beam_size=beam_size,
                output_txt=True,
                output_srt=output_srt,
                output_vtt=output_vtt,
                output_json=output_json,
                include_header=include_header,
                include_timestamps=include_timestamps,
                progress_callback=_progress_cb,
            )
            with _tr_lock:
                _tr_live_progress[tr_id] = 1.0
            db.update_transcription(tr_id, {
                "status": TranscriptionStatus.COMPLETED.value,
                "output_txt_path": result.get("txt_path"),
                "output_srt_path": result.get("srt_path"),
                "output_vtt_path": result.get("vtt_path"),
                "output_json_path": result.get("json_path"),
                "progress": 1.0,
                "completed_at": datetime.utcnow().isoformat(),
            }, DB_PATH)
        except Exception as exc:
            db.update_transcription(tr_id, {
                "status": TranscriptionStatus.FAILED.value,
                "error_message": str(exc)[:500],
                "completed_at": datetime.utcnow().isoformat(),
            }, DB_PATH)


def _submit_transcription_batch(
    tr_ids: list[int],
    output_srt: bool,
    output_vtt: bool,
    output_json: bool,
    beam_size: int,
    include_header: bool,
    include_timestamps: bool,
) -> None:
    """Launch a background worker thread for a batch of transcription IDs."""
    global _tr_worker_thread
    if _tr_worker_thread is not None and _tr_worker_thread.is_alive():
        # Don't allow two workers simultaneously; caller should check first
        return
    _tr_worker_thread = threading.Thread(
        target=_transcription_worker,
        args=(tr_ids, output_srt, output_vtt, output_json,
              beam_size, include_header, include_timestamps),
        daemon=True,
        name="TranscriptionWorker",
    )
    _tr_worker_thread.start()


def _is_transcription_worker_alive() -> bool:
    return _tr_worker_thread is not None and _tr_worker_thread.is_alive()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "Dashboard":          "dashboard",
    "Projects":           "projects",
    "New Project":        "new_project",
    "Accounts":           "accounts",
    "Source Library":     "shared_sources",
    "Run Queue":          "queue",
    "Files":              "files",
    "Audio Conversion":   "audio_conv",
    "Transcription":      "transcribe",
    "AI Folder":          "ai_folder",
    "Word Booklet":       "booklet",
    "Settings":           "settings",
    "Logs":               "logs",
}

PAGE_ICONS = {
    "Dashboard": "🏠", "Projects": "📁", "New Project": "➕",
    "Accounts": "👤", "Source Library": "📚",
    "Run Queue": "▶", "Files": "🗂", "Audio Conversion": "🎵",
    "Transcription": "📝", "AI Folder": "🤖",
    "Word Booklet": "📖",
    "Settings": "⚙", "Logs": "📋",
}


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🎙 NLM Audio Generator")
        st.caption(f"Version {APP_VERSION}")
        st.divider()
        for label, key in PAGES.items():
            icon = PAGE_ICONS.get(label, "")
            if st.button(f"{icon} {label}", width="stretch", key=f"nav_{key}"):
                st.session_state["page"] = key
        st.divider()
        _runner_status_widget()
    return st.session_state.get("page", "dashboard")


def _runner_status_widget() -> None:
    alive = _is_runner_alive()
    if alive:
        st.success("Runner: Running")
    else:
        st.warning("Runner: Stopped")
        if st.button("▶ Start Runner", key="sidebar_start_runner"):
            _launch_runner()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Runner helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_runner_alive() -> bool:
    if not RUNNER_PID_FILE.exists():
        return False
    try:
        pid = int(RUNNER_PID_FILE.read_text().strip())
        if sys.platform == "win32":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def _launch_runner(fake: bool = False) -> None:
    if _is_runner_alive():
        return
    args = [sys.executable, str(Path(__file__).parent / "runner.py")]
    if fake:
        args.append("--fake")
    RUNNER_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNNER_LOG_FILE, "a", encoding="utf-8") as error_log:
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=error_log,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            ),
        )
    time.sleep(0.5)


def _stop_runner() -> None:
    if not RUNNER_PID_FILE.exists():
        return
    try:
        pid = int(RUNNER_PID_FILE.read_text().strip())
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            import signal as sig
            os.kill(pid, sig.SIGTERM)
    except Exception:
        pass
    RUNNER_PID_FILE.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Status icons / helpers
# ══════════════════════════════════════════════════════════════════════════════

_JOB_ICON = {
    JobStatus.PENDING: "⏳", JobStatus.ASSIGNED: "📋",
    JobStatus.CREATING_NOTEBOOK: "📓", JobStatus.UPLOADING_MAIN_SOURCE: "📤",
    JobStatus.UPLOADING_SHARED_SOURCES: "📤", JobStatus.WAITING_FOR_SOURCES: "🔄",
    JobStatus.GENERATING_AUDIO: "🎙", JobStatus.WAITING_FOR_AUDIO: "⏱",
    JobStatus.DOWNLOADING_AUDIO: "⬇", JobStatus.CONVERTING_AUDIO: "🔀",
    JobStatus.TRANSCRIBING_AUDIO: "📝", JobStatus.COMPLETED: "✅",
    JobStatus.FAILED: "❌", JobStatus.CANCELLED: "🚫",
}

_ACC_ICON = {
    AccountStatus.ACTIVE: "🟢", AccountStatus.AUTH_EXPIRED: "🔴",
    AccountStatus.RATE_LIMITED: "🟡", AccountStatus.DISABLED: "⚫",
    AccountStatus.CHECKING: "🔵", AccountStatus.ERROR: "🔴",
    AccountStatus.LOGIN_REQUIRED: "🟠",
}

# Supported output languages
LANGUAGES = {
    "fa": "Persian",
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "tr": "Turkish",
    "ru": "Russian",
    "es": "Spanish",
    "zh": "Chinese",
    "ja": "Japanese",
}

DEFAULT_PROMPTS = {
    "fa": (
        "لطفاً یک مرور صوتی جامع و کامل از این سند به زبان فارسی تهیه کن. "
        "مفاهیم کلیدی، نکات اصلی و جمع‌بندی مهم را به شکل واضح و جذاب بیان کن."
    ),
    "en": (
        "Please create a comprehensive audio overview of this document in English. "
        "Explain the key concepts, main points and important summaries clearly and engagingly."
    ),
    "ar": (
        "يرجى إنشاء نظرة عامة صوتية شاملة لهذه الوثيقة باللغة العربية. "
        "اشرح المفاهيم الرئيسية والنقاط الأساسية والملخصات المهمة بوضوح وجاذبية."
    ),
}


def _elapsed(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    delta = int((datetime.utcnow() - dt).total_seconds())
    if delta < 60: return f"{delta}s"
    m, s = divmod(delta, 60)
    if m < 60: return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _fmt_size(b: int) -> str:
    if b < 1024: return f"{b} B"
    elif b < 1024**2: return f"{b/1024:.1f} KB"
    elif b < 1024**3: return f"{b/1024**2:.1f} MB"
    return f"{b/1024**3:.2f} GB"


# ══════════════════════════════════════════════════════════════════════════════
# Page: Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard() -> None:
    st.title("🏠 Dashboard")

    accounts = db.list_accounts(DB_PATH)
    projects = db.list_projects(DB_PATH)

    active_accounts  = [a for a in accounts if a.enabled and a.auth_status == AccountStatus.ACTIVE]
    running_projects = [p for p in projects if p.status == ProjectStatus.RUNNING]

    all_jobs: list = []
    for p in projects:
        all_jobs.extend(db.get_jobs_for_project(p.id, DB_PATH))

    running_jobs   = [j for j in all_jobs if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.PENDING)]
    pending_jobs   = [j for j in all_jobs if j.status == JobStatus.PENDING]
    completed_jobs = [j for j in all_jobs if j.status == JobStatus.COMPLETED]
    failed_jobs    = [j for j in all_jobs if j.status == JobStatus.FAILED]

    cols = st.columns(5)
    cols[0].metric("Active Accounts", len(active_accounts))
    cols[1].metric("Running Jobs", len(running_jobs))
    cols[2].metric("Pending", len(pending_jobs))
    cols[3].metric("Completed", len(completed_jobs))
    cols[4].metric("Failed", len(failed_jobs))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recent Projects")
        if projects:
            rows = []
            for p in projects[:5]:
                pjobs = [j for j in all_jobs if j.project_id == p.id]
                rows.append({
                    "Name": p.name,
                    "Status": p.status.value,
                    "Chunks": len(pjobs),
                    "Done": len([j for j in pjobs if j.status == JobStatus.COMPLETED]),
                    "Language": p.language,
                })
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No projects yet. Create one from the sidebar.")

    with col2:
        st.subheader("Account Status")
        if accounts:
            rows = []
            for a in accounts:
                icon = _ACC_ICON.get(a.auth_status, "⚪")
                rows.append({
                    "Account": a.display_name or a.profile_name,
                    "Status": f"{icon} {a.auth_status.value}",
                    "Enabled": "Yes" if a.enabled else "No",
                    "Type": a.account_type.value,
                })
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No accounts configured. Add them in the Accounts page.")

    if _is_runner_alive():
        time.sleep(UI_REFRESH_INTERVAL)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Projects
# ══════════════════════════════════════════════════════════════════════════════

def page_projects() -> None:
    st.title("📁 Projects")

    if st.button("➕ New Project", type="primary"):
        st.session_state["page"] = "new_project"
        st.rerun()

    projects = db.list_projects(DB_PATH)
    if not projects:
        st.info("No projects yet. Click «New Project» to get started.")
        return

    for proj in projects:
        jobs = db.get_jobs_for_project(proj.id, DB_PATH)
        completed = len([j for j in jobs if j.status == JobStatus.COMPLETED])
        total = len(jobs)

        status_icon = {
            ProjectStatus.PENDING: "⏳", ProjectStatus.RUNNING: "▶",
            ProjectStatus.COMPLETED: "✅", ProjectStatus.FAILED: "❌",
            ProjectStatus.STOPPED: "⏸", ProjectStatus.PAUSED: "⏸",
        }.get(proj.status, "?")

        with st.expander(f"{status_icon} **{proj.name}**  [{completed}/{total} chunks]", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.write(f"**Status:** {proj.status.value}")
            c2.write(f"**Chunks:** {total}")
            c3.write(f"**Language:** {LANGUAGES.get(proj.language, proj.language)}")
            c4.write(f"**Created:** {proj.created_at.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**PDF:** `{proj.original_filename}`")

            btn_cols = st.columns(6)
            if btn_cols[0].button("▶ Resume", key=f"resume_{proj.id}"):
                _ensure_runner_and_set_running(proj.id)
                st.rerun()
            if btn_cols[1].button("⏸ Pause", key=f"stop_{proj.id}"):
                db.update_project_status(proj.id, ProjectStatus.STOPPED, DB_PATH)
                st.rerun()
            if btn_cols[2].button("📁 Files", key=f"files_{proj.id}"):
                st.session_state["selected_project_id"] = proj.id
                st.session_state["page"] = "files"
                st.rerun()
            if btn_cols[3].button("▶ Queue", key=f"queue_{proj.id}"):
                st.session_state["selected_project_id"] = proj.id
                st.session_state["page"] = "queue"
                st.rerun()
            if btn_cols[4].button("📦 ZIP", key=f"zip_{proj.id}"):
                _export_zip(proj.id)
            if btn_cols[5].button("🗑 Delete", key=f"del_{proj.id}"):
                st.session_state[f"confirm_del_{proj.id}"] = True

            if st.session_state.get(f"confirm_del_{proj.id}"):
                st.warning("Are you sure you want to delete this project?")
                d1, d2, d3 = st.columns(3)
                if d1.button("Record only", key=f"del_rec_{proj.id}"):
                    db.delete_project(proj.id, DB_PATH)
                    st.rerun()
                if d2.button("Record + Files", key=f"del_files_{proj.id}"):
                    from file_service import delete_project_files
                    delete_project_files(proj.id, DB_PATH)
                    db.delete_project(proj.id, DB_PATH)
                    st.rerun()
                if d3.button("Cancel", key=f"cancel_del_{proj.id}"):
                    st.session_state[f"confirm_del_{proj.id}"] = False
                    st.rerun()


def _export_zip(project_id: int) -> None:
    from file_service import create_project_zip
    zip_path = create_project_zip(project_id, path=DB_PATH)
    if zip_path and zip_path.exists():
        with open(zip_path, "rb") as f:
            st.download_button(
                "⬇ Download ZIP",
                data=f.read(),
                file_name=zip_path.name,
                mime="application/zip",
                key=f"dl_zip_{project_id}_{int(time.time())}",
            )
    else:
        st.error("Failed to create ZIP archive.")


def _ensure_runner_and_set_running(project_id: int) -> bool:
    from allocation_service import validate_project_preflight

    report = validate_project_preflight(project_id, DB_PATH)
    if not report["can_start"]:
        db.update_project_status(project_id, ProjectStatus.PENDING, DB_PATH)
        for issue in report["issues"]:
            st.error(issue)
        return False
    for warning in report["warnings"]:
        st.warning(warning)
    db.update_project_status(project_id, ProjectStatus.RUNNING, DB_PATH)
    if not _is_runner_alive():
        _launch_runner()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Page: New Project
# ══════════════════════════════════════════════════════════════════════════════

def page_new_project() -> None:
    st.title("➕ New Project")
    saved_settings = db.get_all_settings(DB_PATH)

    with st.form("new_project_form"):
        st.subheader("Project Info")
        proj_name = st.text_input("Project Name", placeholder="Anatomy – Chapter 1")

        st.subheader("Output Language")
        lang_keys   = list(LANGUAGES.keys())
        lang_labels = list(LANGUAGES.values())
        saved_language = saved_settings.get("default_language", "fa")
        lang_idx = (
            lang_keys.index(saved_language) if saved_language in lang_keys else 0
        )
        sel_lang    = st.selectbox(
            "Audio generation language (default: Persian)",
            options=lang_keys,
            index=lang_idx,
            format_func=lambda k: LANGUAGES.get(k, k),
        )
        st.caption("This sets the language for the NotebookLM Audio Overview output.")

        st.subheader("PDF Upload")
        uploaded = st.file_uploader("Select PDF file", type=["pdf"])

        st.subheader("Page Range Split")
        split_mode = st.radio("Split mode", ["Manual", "Automatic"], horizontal=True)
        ranges_text   = ""
        pages_per_chunk = 15
        if split_mode == "Manual":
            ranges_text = st.text_area(
                "Page ranges (one per line: start-end)",
                placeholder="1-10\n11-25\n26-40",
                height=150,
            )
        else:
            pages_per_chunk = st.number_input("Pages per chunk", min_value=1, value=15)

        st.subheader("Audio Prompt Template")
        default_prompt = DEFAULT_PROMPTS.get(sel_lang, DEFAULT_PROMPTS["en"])
        prompt = st.text_area("Prompt template", value=default_prompt, height=100,
                              help="Placeholders: {{project_name}}, {{chunk_index}}, {{start_page}}, {{end_page}}")

        st.subheader("Conversion Settings")
        c1, c2, c3 = st.columns(3)
        auto_mp3 = c1.toggle("Auto-convert to MP3", value=False)
        bitrate_options = ["64k", "96k", "128k", "160k", "192k"]
        saved_bitrate = saved_settings.get("default_bitrate", "128k")
        mp3_bitrate = c2.selectbox(
            "MP3 bitrate",
            bitrate_options,
            index=(
                bitrate_options.index(saved_bitrate)
                if saved_bitrate in bitrate_options
                else 2
            ),
        )
        keep_m4a = c3.toggle(
            "Keep original M4A",
            value=saved_settings.get("keep_m4a", "1") == "1",
        )
        auto_transcribe = st.toggle("Auto-transcribe with Whisper", value=False)

        st.subheader("Account Allocation")
        accounts = db.list_accounts(DB_PATH)
        if not accounts:
            st.warning("No accounts configured. Add them in the Accounts page first.")
        alloc_data: list[dict] = []
        for acc in accounts:
            if not acc.enabled:
                continue
            c1b, c2b, c3b = st.columns([3, 2, 2])
            c1b.write(f"**{acc.display_name or acc.profile_name}**")
            en = c2b.checkbox("Enable", value=True, key=f"alloc_en_{acc.id}")
            mj = c3b.number_input(
                "Max jobs",
                min_value=0,
                value=acc.default_project_job_limit,
                key=f"alloc_jobs_{acc.id}",
                help="Daily quota and parallel concurrency for this account",
            )
            alloc_data.append({"account_id": acc.id, "enabled": en, "max_jobs": mj})

        alloc_mode_label = st.radio(
            "Allocation mode",
            ["EXACT – each account gets exactly its quota",
             "FLEXIBLE – overflow allowed across accounts"],
            index=0,
        )

        st.subheader("Shared Sources")
        st.caption(
            "Optionally attach shared documents to every notebook in this project. "
            "Global sources are picked from the library; project sources are uploaded fresh and belong only to this project."
        )
        global_sources_all = db.list_global_shared_sources(DB_PATH)
        enabled_global = [s for s in global_sources_all if s.enabled]
        selected_global_ids: list[int] = []
        if enabled_global:
            selected_global_ids = st.multiselect(
                "Global sources to attach (all notebooks)",
                options=[s.id for s in enabled_global],
                format_func=lambda sid: next(
                    (s.display_name for s in enabled_global if s.id == sid), str(sid)
                ),
                key="new_proj_global_sources",
            )
        else:
            st.caption("No global sources in library yet. You can add them from the Source Library page.")

        new_source_files = st.file_uploader(
            "Upload project-specific sources (PDF, TXT, MD, DOCX)",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="new_proj_source_files",
        )

        submitted = st.form_submit_button("Create Project & Start", type="primary")

    if submitted:
        _create_project(
            proj_name, sel_lang, uploaded,
            split_mode, ranges_text, pages_per_chunk,
            prompt, auto_mp3, mp3_bitrate, keep_m4a, auto_transcribe,
            alloc_data, alloc_mode_label,
            selected_global_ids, new_source_files or [],
        )


def _create_project(
    name, language, uploaded,
    split_mode, ranges_text, pages_per_chunk,
    prompt, auto_mp3, mp3_bitrate, keep_m4a, auto_transcribe,
    alloc_data, allocation_mode_str,
    selected_global_ids: list | None = None,
    new_source_files: list | None = None,
) -> None:
    from file_service import slugify
    from pdf_service import (
        auto_split_ranges, parse_page_ranges, validate_ranges,
        split_pdf, get_pdf_page_count, compute_pdf_hash, chunk_hashes,
    )
    from settings import PROJECTS_DIR

    if not name.strip():
        st.error("Project name cannot be empty."); return
    if not uploaded:
        st.error("Please upload a PDF file."); return
    selected_allocations = [
        item for item in alloc_data
        if item["enabled"] and int(item["max_jobs"]) > 0
    ]
    if not selected_allocations:
        st.error("Select at least one account with a job quota greater than zero.")
        return
    inactive_accounts = []
    for item in selected_allocations:
        account = db.get_account(item["account_id"], DB_PATH)
        if not account or account.auth_status != AccountStatus.ACTIVE:
            inactive_accounts.append(
                account.display_name if account else str(item["account_id"])
            )
    if inactive_accounts:
        st.error(
            "Check authentication before starting. Inactive accounts: "
            + ", ".join(inactive_accounts)
        )
        return

    slug = slugify(name)
    project_dir = PROJECTS_DIR / slug
    project_dir.mkdir(parents=True, exist_ok=True)

    original_dir = project_dir / "original"
    original_dir.mkdir(exist_ok=True)
    pdf_path = original_dir / "original.pdf"
    pdf_bytes = uploaded.read()
    pdf_path.write_bytes(pdf_bytes)

    with st.spinner("Reading PDF..."):
        try:
            total_pages = get_pdf_page_count(pdf_path)
        except Exception as exc:
            st.error(f"Failed to read PDF: {exc}"); return

    st.info(f"PDF has **{total_pages}** pages.")

    if split_mode == "Manual":
        if not ranges_text.strip():
            st.error("Please enter page ranges."); return
        try:
            ranges = parse_page_ranges(ranges_text)
        except ValueError as e:
            st.error(str(e)); return
    else:
        ranges = auto_split_ranges(total_pages, int(pages_per_chunk))

    validation = validate_ranges(ranges, total_pages)
    if not validation["valid"]:
        for err in validation["errors"]:
            st.error(err)
        return
    for w in validation["warnings"]:
        st.warning(w)
    total_quota = sum(int(item["max_jobs"]) for item in selected_allocations)
    if total_quota < len(ranges):
        import shutil
        shutil.rmtree(project_dir, ignore_errors=True)
        st.error(
            f"Account quota is insufficient: {total_quota} jobs for "
            f"{len(ranges)} chunks. Increase allocation before creating the project."
        )
        return

    st.subheader("Split Preview")
    st.dataframe(validation["preview"], width="stretch", hide_index=True)

    allocation_mode = AllocationMode.EXACT if "EXACT" in allocation_mode_str else AllocationMode.FLEXIBLE

    with st.spinner(f"Splitting PDF into {len(ranges)} chunks..."):
        chunks_dir = project_dir / "chunks"
        try:
            chunk_paths = split_pdf(pdf_path, ranges, chunks_dir)
        except Exception as exc:
            st.error(f"PDF split failed: {exc}"); return

    pdf_hash = compute_pdf_hash(pdf_path)
    project_id = db.create_project(
        name=name.strip(),
        slug=slug,
        original_filename=uploaded.name,
        original_pdf_path=str(pdf_path),
        total_pages=total_pages,
        prompt_template=prompt.strip(),
        language=language,
        allocation_mode=allocation_mode,
        output_dir=str(project_dir),
        path=DB_PATH,
    )
    saved_settings = db.get_all_settings(DB_PATH)
    db.update_project(project_id, {
        "original_pdf_hash": pdf_hash,
        "auto_convert_to_mp3": int(auto_mp3),
        "mp3_bitrate": mp3_bitrate,
        "keep_original_audio": int(keep_m4a),
        "auto_transcribe": int(auto_transcribe),
        "whisper_model": saved_settings.get("whisper_model", "small"),
        "whisper_language": saved_settings.get("whisper_language", "fa"),
    }, DB_PATH)

    from shared_source_service import (
        add_project_source,
        attach_global_sources_to_project,
        is_supported_format,
    )
    if selected_global_ids:
        attach_global_sources_to_project(project_id, selected_global_ids, path=DB_PATH)
        st.info(f"Attached {len(selected_global_ids)} global source(s) to project.")
    for f in (new_source_files or []):
        if is_supported_format(f.name):
            add_project_source(f.read(), f.name, project_id, path=DB_PATH)
        else:
            st.warning(f"Skipped unsupported file: {f.name}")
    if new_source_files:
        supported = [f for f in new_source_files if is_supported_format(f.name)]
        if supported:
            st.info(f"Uploaded {len(supported)} project-specific source(s).")

    c_hashes = chunk_hashes(chunk_paths)
    for i, ((start, end), cp, ch) in enumerate(zip(ranges, chunk_paths, c_hashes), start=1):
        db.create_chunk(project_id, i, start, end, str(cp), ch, DB_PATH)

    for a in alloc_data:
        if a["max_jobs"] > 0:
            max_jobs = int(a["max_jobs"])
            db.upsert_allocation(
                project_id=project_id,
                account_id=a["account_id"],
                max_jobs_for_project=max_jobs,
                max_concurrent_jobs=max_jobs,
                enabled=a["enabled"],
                path=DB_PATH,
            )

    from allocation_service import validate_allocations
    report = validate_allocations(project_id, DB_PATH)
    for issue in report.get("issues", []):
        st.warning(issue)
    if report.get("deficit", 0) > 0:
        st.error(
            f"Project cannot start: {report['deficit']} chunk(s) are unassigned."
        )
        return
    st.success(
        f"✅ {len(ranges)} chunks | Total quota: {report['total_quota']} | "
        f"{report['enabled_accounts']} account(s) enabled"
    )

    if not _ensure_runner_and_set_running(project_id):
        return

    st.success(f"Project «{name}» created with {len(ranges)} chunks. Runner started.")
    st.session_state["selected_project_id"] = project_id
    st.session_state["page"] = "queue"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Accounts
# ══════════════════════════════════════════════════════════════════════════════

def page_accounts() -> None:
    st.title("👤 NotebookLM Accounts")
    st.info("📌 This app never stores your email, password, cookies, or tokens.")

    accounts = db.list_accounts(DB_PATH)

    for acc in accounts:
        icon = _ACC_ICON.get(acc.auth_status, "⚪")
        with st.expander(f"{icon} **{acc.display_name or acc.profile_name}**  `{acc.auth_status.value}`", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Profile:** `{acc.profile_name}`")
            c2.write(f"**Type:** {acc.account_type.value} ({acc.default_project_job_limit} jobs)")
            c3.write(f"**Enabled:** {'Yes' if acc.enabled else 'No'}")
            if acc.description:
                st.caption(acc.description)
            if acc.last_auth_check_at:
                st.caption(f"Last checked: {acc.last_auth_check_at.strftime('%Y-%m-%d %H:%M')}")

            st.subheader("Login commands")
            from account_service import get_login_commands
            cmds = get_login_commands(acc.profile_name)
            st.code("\n".join(cmds), language="bash")

            btn1, btn2, btn3, btn4, btn5 = st.columns(5)
            if btn1.button("🔍 Check Auth", key=f"check_{acc.id}"):
                import asyncio
                from account_service import check_account_auth
                with st.spinner("Checking..."):
                    try:
                        status = asyncio.run(check_account_auth(acc, DB_PATH))
                        st.info(f"Status: {status.value}")
                    except Exception as exc:
                        st.error(str(exc))
                st.rerun()

            if btn2.button("🍪 Edge Cookies", key=f"cookies_{acc.id}", help="کوکی‌های Edge لاگین‌شده رو مستقیم بخون (بدون باز شدن مرورگر)"):
                from account_service import login_with_edge_cookies
                with st.spinner("در حال خواندن کوکی‌های Edge..."):
                    ok, msg = login_with_edge_cookies(acc.profile_name)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            if btn3.button("🌐 Open Edge Login", key=f"login_{acc.id}", help="یک پنجره Edge برای لاگین گوگل باز کن"):
                from account_service import open_login_browser
                ok, msg = open_login_browser(acc.profile_name)
                if ok: st.success(msg)
                else: st.error(msg)

            if btn4.button("Disable" if acc.enabled else "Enable", key=f"toggle_{acc.id}"):
                db.update_account(acc.id, {"enabled": int(not acc.enabled)}, DB_PATH)
                st.rerun()

            if btn5.button("🗑 Delete", key=f"del_acc_{acc.id}", type="secondary"):
                st.session_state[f"confirm_del_acc_{acc.id}"] = True

            if st.session_state.get(f"confirm_del_acc_{acc.id}"):
                from account_service import get_deletion_risks, delete_account_record
                risks = get_deletion_risks(acc.id, DB_PATH)
                if risks["has_active_jobs"]:
                    st.error(f"Cannot delete: {risks['active_job_count']} active job(s) exist.")
                else:
                    st.warning("Confirm deletion:")
                    d1, d2, d3 = st.columns(3)
                    if d1.button("Just disable", key=f"da_dis_{acc.id}"):
                        db.update_account(acc.id, {"enabled": 0}, DB_PATH); st.rerun()
                    if d2.button("Delete record", key=f"da_rec_{acc.id}"):
                        delete_account_record(acc.id, DB_PATH); st.rerun()
                    if d3.button("Cancel", key=f"da_can_{acc.id}"):
                        st.session_state[f"confirm_del_acc_{acc.id}"] = False; st.rerun()

    st.divider()
    st.subheader("Add New Account")
    acc_type = st.selectbox(
        "Account Type",
        [t.value for t in AccountType],
        key="add_acc_type",
        help="FREE default: 3 jobs · PAID default: 20 jobs (quota & concurrency)",
    )
    _acc_type = AccountType(acc_type)
    _jobs_default = daily_job_quota_for(_acc_type)

    with st.form("add_account_form"):
        c1, c2 = st.columns(2)
        profile_name = c1.text_input("Profile Name", placeholder="account_01")
        display_name = c2.text_input("Display Name", placeholder="My Study Account")
        description  = st.text_input("Description (optional)")
        max_jobs = st.number_input(
            "Max jobs",
            min_value=1,
            value=_jobs_default,
            help="Daily quota and parallel concurrency (FREE: 3, PAID: 20)",
        )

        if st.form_submit_button("Add Account"):
            if not profile_name.strip():
                st.error("Profile name cannot be empty.")
            else:
                db.create_account(
                    profile_name=profile_name.strip(),
                    display_name=display_name.strip(),
                    description=description.strip(),
                    account_type=_acc_type,
                    default_job_limit=int(max_jobs),
                    default_concurrency=int(max_jobs),
                    path=DB_PATH,
                )
                st.success(f"Account '{profile_name}' added.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Source Library
# ══════════════════════════════════════════════════════════════════════════════

def page_shared_sources() -> None:
    st.title("📚 Source Library")

    st.markdown("""
**Two types of shared sources:**
- 🌐 **Global Sources** — stored in the global library; can be manually attached to any project's notebooks.
- 📁 **Project Sources** — belong exclusively to one project; automatically added to every notebook in that project only.
    """)

    tab_global, tab_project = st.tabs(["🌐 Global Sources", "📁 Project Sources"])

    # ── Global tab ──────────────────────────────────────────────────────────
    with tab_global:
        st.subheader("Global Source Library")
        st.caption("These files can be attached to any project. Each project decides which global sources to include.")
        global_sources = db.list_global_shared_sources(DB_PATH)
        _render_source_list(global_sources, SourceScope.GLOBAL)

        st.divider()
        st.subheader("Upload Global Source")
        _add_source_form(scope=SourceScope.GLOBAL, project_id=None)

        # Attach global sources to a project
        projects = db.list_projects(DB_PATH)
        if projects and global_sources:
            st.divider()
            st.subheader("Attach Global Source to Project")
            col1, col2, col3 = st.columns(3)
            sel_proj = col1.selectbox(
                "Project",
                options=[p.id for p in projects],
                format_func=lambda pid: next((p.name for p in projects if p.id == pid), str(pid)),
                key="attach_global_proj",
            )
            sel_src = col2.selectbox(
                "Global Source",
                options=[s.id for s in global_sources],
                format_func=lambda sid: next((s.display_name for s in global_sources if s.id == sid), str(sid)),
                key="attach_global_src",
            )
            attach_mode_label = col3.selectbox(
                "Attach mode",
                options=["All Notebooks", "Disabled"],
                key="attach_global_mode",
            )
            if st.button("Attach to Project", key="do_attach_global"):
                mode = AttachMode.ALL_NOTEBOOKS if attach_mode_label == "All Notebooks" else AttachMode.DISABLED
                db.attach_shared_source_to_project(sel_proj, sel_src, mode, path=DB_PATH)
                st.success("Source attached to project.")
                st.rerun()

    # ── Project tab ──────────────────────────────────────────────────────────
    with tab_project:
        st.subheader("Project-Specific Sources")
        st.caption("These files belong to one project only and are automatically added to every notebook in that project.")

        projects = db.list_projects(DB_PATH)
        if not projects:
            st.info("No projects yet.")
        else:
            selected_proj_id = st.selectbox(
                "Select Project",
                options=[p.id for p in projects],
                format_func=lambda pid: next((p.name for p in projects if p.id == pid), str(pid)),
                key="proj_src_selector",
            )

            if selected_proj_id:
                proj_sources = db.list_project_shared_sources(selected_proj_id, DB_PATH)
                if proj_sources:
                    _render_source_list(proj_sources, SourceScope.PROJECT)
                else:
                    st.info("No project-specific sources for this project.")

                st.divider()
                st.subheader("Upload Project Source")
                st.caption(f"This file will only be used in notebooks belonging to the selected project.")
                _add_source_form(scope=SourceScope.PROJECT, project_id=selected_proj_id)

            # Show which global sources are also attached
            st.divider()
            if selected_proj_id:
                _show_attached_global_sources(selected_proj_id)


def _show_attached_global_sources(project_id: int) -> None:
    """Display global sources currently attached to this project."""
    from database import get_active_sources_for_project
    active = get_active_sources_for_project(project_id, DB_PATH)
    global_active = [s for s in active if s.scope == SourceScope.GLOBAL]

    st.subheader("Global Sources Attached to This Project")
    if not global_active:
        st.caption("No global sources attached.")
    else:
        rows = []
        for s in global_active:
            rows.append({
                "Name": s.display_name,
                "File": s.original_filename,
                "Size": _fmt_size(s.file_size),
                "Enabled": "✅" if s.enabled else "❌",
            })
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_source_list(sources, scope: SourceScope) -> None:
    if not sources:
        st.caption("No sources found.")
        return

    rows = []
    for s in sources:
        rows.append({
            "Name": s.display_name,
            "File": s.original_filename,
            "Size": _fmt_size(s.file_size),
            "Type": s.mime_type.split("/")[-1],
            "Enabled": "✅" if s.enabled else "❌",
        })
    st.dataframe(rows, width="stretch", hide_index=True)

    for s in sources:
        with st.expander(f"⚙ Edit: {s.display_name}", expanded=False):
            new_name    = st.text_input("Display name", value=s.display_name, key=f"ss_name_{s.id}")
            new_desc    = st.text_input("Description",  value=s.description,  key=f"ss_desc_{s.id}")
            new_enabled = st.toggle("Enabled", value=s.enabled, key=f"ss_en_{s.id}")
            c1, c2, c3  = st.columns(3)
            if c1.button("Save", key=f"ss_save_{s.id}"):
                db.update_shared_source(s.id, {
                    "display_name": new_name,
                    "description": new_desc,
                    "enabled": int(new_enabled),
                }, DB_PATH)
                st.success("Saved."); st.rerun()
            if c2.button("Delete", key=f"ss_del_{s.id}"):
                db.delete_shared_source(s.id, DB_PATH); st.rerun()
            if c3.button("Replace file", key=f"ss_replace_{s.id}"):
                st.session_state[f"replace_{s.id}"] = True
            if st.session_state.get(f"replace_{s.id}"):
                new_file = st.file_uploader("New file", key=f"ss_newfile_{s.id}")
                if new_file:
                    from shared_source_service import replace_source_file
                    replace_source_file(s.id, new_file.read(), new_file.name, DB_PATH)
                    st.success("File replaced."); st.rerun()


def _add_source_form(scope: SourceScope, project_id: Optional[int]) -> None:
    key_suffix = f"{scope.value}_{project_id or 'g'}"
    with st.form(f"add_source_{key_suffix}"):
        uploaded = st.file_uploader(
            "Source file (PDF, TXT, MD, DOCX)",
            type=["pdf","txt","md","docx"],
            key=f"src_file_{key_suffix}",
        )
        c1, c2 = st.columns(2)
        disp_name = c1.text_input("Display name", key=f"src_name_{key_suffix}")
        desc      = c2.text_input("Description",  key=f"src_desc_{key_suffix}")

        if st.form_submit_button("Upload Source"):
            if not uploaded:
                st.error("No file selected.")
            else:
                from shared_source_service import (
                    add_global_source, add_project_source, is_supported_format
                )
                if not is_supported_format(uploaded.name):
                    st.error(f"Format '{Path(uploaded.name).suffix}' not supported.")
                    return
                if scope == SourceScope.GLOBAL:
                    s = add_global_source(uploaded.read(), uploaded.name, disp_name, desc, DB_PATH)
                else:
                    s = add_project_source(uploaded.read(), uploaded.name, project_id, disp_name, desc, DB_PATH)
                st.success(f"Source '{s.display_name}' uploaded.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Run Queue
# ══════════════════════════════════════════════════════════════════════════════

def page_queue() -> None:
    st.title("▶ Run Queue")

    projects = db.list_projects(DB_PATH)
    if not projects:
        st.info("No projects yet."); return

    project_id = st.session_state.get("selected_project_id")
    proj_map = {p.id: f"{p.name} ({p.status.value})" for p in projects}
    if project_id not in proj_map:
        project_id = projects[0].id

    selected_id = st.selectbox(
        "Project",
        options=list(proj_map.keys()),
        format_func=lambda pid: proj_map[pid],
        index=list(proj_map.keys()).index(project_id) if project_id in proj_map else 0,
    )
    st.session_state["selected_project_id"] = selected_id

    proj = db.get_project(selected_id, DB_PATH)
    if not proj: return

    jobs       = db.get_jobs_for_project(selected_id, DB_PATH)
    accounts   = db.list_accounts(DB_PATH)
    allocations = db.get_allocations_for_project(selected_id, DB_PATH)

    # Control buttons
    btn_cols = st.columns(6)
    if btn_cols[0].button("▶ Start / Resume", type="primary"):
        _ensure_runner_and_set_running(selected_id); st.rerun()
    if btn_cols[1].button("⏸ Pause"):
        db.update_project_status(selected_id, ProjectStatus.PAUSED, DB_PATH); st.rerun()
    if btn_cols[2].button("⏹ Stop"):
        db.update_project_status(selected_id, ProjectStatus.STOPPED, DB_PATH); st.rerun()
    if btn_cols[3].button("🔁 Retry failed"):
        failed = db.get_failed_jobs(selected_id, DB_PATH)
        for j in failed:
            db.reset_job_for_retry(j.id, path=DB_PATH)
        if failed:
            _ensure_runner_and_set_running(selected_id)
        st.rerun()
    if btn_cols[4].button("🔀 Convert all M4A"):
        _batch_convert_project(selected_id)
    if btn_cols[5].button("📝 Transcribe all"):
        _batch_transcribe_project(selected_id)

    st.divider()

    running = [j for j in jobs if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.PENDING)]
    pending  = [j for j in jobs if j.status == JobStatus.PENDING]
    done     = [j for j in jobs if j.status == JobStatus.COMPLETED]
    failed   = [j for j in jobs if j.status == JobStatus.FAILED]

    m = st.columns(5)
    m[0].metric("Total", len(jobs))
    m[1].metric("Active", len(running))
    m[2].metric("Pending", len(pending))
    m[3].metric("Completed", len(done))
    m[4].metric("Failed", len(failed))

    # Source summary
    from shared_source_service import estimate_sources_per_notebook
    src_info = estimate_sources_per_notebook(selected_id, DB_PATH)
    if src_info:
        src_cols = st.columns(4)
        src_cols[0].metric("Sources / Notebook", src_info["total"])
        src_cols[1].metric("Main PDF", src_info["main_pdf"])
        src_cols[2].metric("Shared Sources", src_info["shared_sources"])
        if src_info["error"]:
            st.error(f"⚠ {src_info['message']}")
        elif src_info["warning"]:
            st.warning(f"⚠ {src_info['message']}")

    # Allocation table
    st.subheader("Account Allocations")
    acc_map = {a.id: a for a in accounts}
    alloc_rows = []
    for alloc in allocations:
        acc = acc_map.get(alloc.account_id)
        alloc_rows.append({
            "Account": acc.display_name if acc else str(alloc.account_id),
            "Enabled": "✅" if alloc.enabled else "❌",
            "Max jobs": alloc.max_jobs_for_project,
            "Assigned": alloc.assigned_jobs_count,
            "Completed": alloc.completed_jobs_count,
            "Failed": alloc.failed_jobs_count,
        })
    if alloc_rows:
        st.dataframe(alloc_rows, width="stretch", hide_index=True)

    # Jobs table
    st.subheader("Jobs")
    job_rows = []
    for j in sorted(jobs, key=lambda x: x.chunk_index):
        acc = acc_map.get(j.account_id)
        icon = _JOB_ICON.get(j.status, "?")
        job_rows.append({
            "#": j.chunk_index,
            "Pages": j.page_label,
            "Account": acc.display_name if acc else "—",
            "Status": f"{icon} {j.status.value}",
            "Step": j.current_step or "—",
            "NB ID": (j.notebook_id or "—")[:12],
            "Tries": j.attempt_count,
            "Elapsed": _elapsed(j.started_at),
            "M4A": "✅" if j.downloaded_audio_path and Path(j.downloaded_audio_path).exists() else "",
            "MP3": "✅" if j.converted_mp3_path and Path(j.converted_mp3_path).exists() else "",
            "Txt": "✅" if j.transcript_txt_path and Path(j.transcript_txt_path).exists() else "",
            "Error": (j.error_message or "")[:50],
        })
    if job_rows:
        st.dataframe(job_rows, width="stretch", hide_index=True)

    # Download audio
    with_mp3   = [j for j in done if j.converted_mp3_path and Path(j.converted_mp3_path).exists()]
    with_audio = [j for j in done if j.downloaded_audio_path and Path(j.downloaded_audio_path).exists()]
    audio_list = with_mp3 or with_audio
    if audio_list:
        st.subheader("Download Audio Files")
        dl_cols = st.columns(4)
        for i, j in enumerate(audio_list):
            ap = Path(j.converted_mp3_path or j.downloaded_audio_path)
            with dl_cols[i % 4]:
                with open(ap, "rb") as f:
                    st.download_button(
                        f"⬇ Chunk {j.chunk_index}",
                        data=f.read(),
                        file_name=ap.name,
                        mime="audio/mpeg",
                        key=f"dl_audio_{j.id}_{int(time.time())}",
                    )

    if _is_runner_alive() or any(
        j.status in (JobStatus.PENDING, JobStatus.ASSIGNED) or
        j.status not in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        for j in jobs
    ):
        time.sleep(UI_REFRESH_INTERVAL)
        st.rerun()


def _batch_convert_project(project_id: int) -> None:
    from audio_service import convert_to_mp3
    proj = db.get_project(project_id, DB_PATH)
    if not proj: return
    jobs = db.get_jobs_for_project(project_id, DB_PATH)
    converted = 0
    for j in jobs:
        if j.status != JobStatus.COMPLETED or not j.downloaded_audio_path: continue
        ap = Path(j.downloaded_audio_path)
        if not ap.exists(): continue
        mp3_path = Path(proj.output_dir) / "audio_mp3" / (ap.stem + ".mp3")
        ok, _ = convert_to_mp3(ap, mp3_path, proj.mp3_bitrate, keep_original=proj.keep_original_audio)
        if ok:
            db.update_job(j.id, {"converted_mp3_path": str(mp3_path)}, DB_PATH)
            converted += 1
    st.success(f"Converted {converted} file(s) to MP3.")


def _batch_transcribe_project(project_id: int) -> None:
    from shared_source_service import compute_file_hash
    from transcription_service import run_transcription_job
    proj = db.get_project(project_id, DB_PATH)
    if not proj: return
    jobs = db.get_jobs_for_project(project_id, DB_PATH)
    count = 0
    for j in jobs:
        if j.status != JobStatus.COMPLETED: continue
        ap_str = j.converted_mp3_path or j.downloaded_audio_path
        if not ap_str: continue
        ap = Path(ap_str)
        if not ap.exists() or (j.transcript_txt_path and Path(j.transcript_txt_path).exists()): continue
        fh = compute_file_hash(ap)
        tr_id = db.create_transcription(
            input_path=str(ap), input_hash=fh,
            model_name=proj.whisper_model, language=proj.whisper_language,
            project_id=project_id, job_id=j.id, path=DB_PATH,
        )
        ok = run_transcription_job(tr_id, path=DB_PATH)
        if ok:
            trs = db.list_transcriptions(project_id, DB_PATH)
            tr = next((t for t in trs if t.id == tr_id), None)
            if tr and tr.output_txt_path:
                db.update_job(j.id, {"transcript_txt_path": tr.output_txt_path}, DB_PATH)
            count += 1
    st.success(f"Created {count} transcript(s).")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Files
# ══════════════════════════════════════════════════════════════════════════════

def page_files() -> None:
    st.title("🗂 Project Files")
    from file_service import list_project_files, create_project_zip, delete_temp_files

    projects = db.list_projects(DB_PATH)
    if not projects:
        st.info("No projects yet."); return

    project_id = st.session_state.get("selected_project_id", projects[0].id)
    proj_map   = {p.id: p.name for p in projects}
    selected_id = st.selectbox(
        "Project",
        options=list(proj_map.keys()),
        format_func=lambda pid: proj_map.get(pid, str(pid)),
        index=list(proj_map.keys()).index(project_id) if project_id in proj_map else 0,
    )
    st.session_state["selected_project_id"] = selected_id

    proj = db.get_project(selected_id, DB_PATH)
    if not proj: return

    from file_service import project_storage_report
    report = project_storage_report(selected_id, DB_PATH)
    if report:
        labels = list(report.keys())
        cols   = st.columns(len(labels))
        for i, k in enumerate(labels):
            cols[i].metric(k, report[k]["size_fmt"])

    st.divider()

    ba_cols = st.columns(4)
    if ba_cols[0].button("📦 Create ZIP"):
        zip_path = create_project_zip(selected_id, path=DB_PATH)
        if zip_path:
            with open(zip_path, "rb") as f:
                st.download_button("⬇ Download ZIP", f.read(),
                                   file_name=zip_path.name, mime="application/zip",
                                   key=f"zip_dl_{int(time.time())}")
    if ba_cols[1].button("🗑 Delete temp files"):
        n = delete_temp_files(selected_id, DB_PATH)
        st.info(f"Deleted {n} temp file(s).")
    if ba_cols[2].button("🔀 Convert all M4A"):
        _batch_convert_project(selected_id)
    if ba_cols[3].button("📝 Transcribe all"):
        _batch_transcribe_project(selected_id)

    st.divider()

    file_cats = list_project_files(selected_id, DB_PATH)
    for category, files in file_cats.items():
        with st.expander(f"{category} ({len(files)} file(s))", expanded=len(files) > 0):
            if not files:
                st.caption("Empty"); continue
            for finfo in files:
                c1, c2, c3 = st.columns([4, 2, 2])
                c1.write(finfo["name"])
                c2.write(finfo["size"])
                fp = Path(finfo["path"])
                if fp.exists():
                    with open(fp, "rb") as f:
                        c3.download_button(
                            "⬇", data=f.read(), file_name=fp.name,
                            key=f"dl_file_{fp.name}_{int(time.time()*1000)}",
                        )


# ══════════════════════════════════════════════════════════════════════════════
# Page: Audio Conversion Tool
# ══════════════════════════════════════════════════════════════════════════════

def page_audio_conversion() -> None:
    st.title("🎵 Audio Conversion Tool")
    from audio_service import is_ffmpeg_available, batch_convert
    from settings import AUDIO_CONV_DIR

    ok, ffmpeg_msg = is_ffmpeg_available()
    if ok:
        st.success(f"FFmpeg: {ffmpeg_msg}")
    else:
        st.error(f"FFmpeg not found: {ffmpeg_msg}")
        with st.expander("FFmpeg Installation Guide (Windows)"):
            st.code("""
# Option 1 – winget (recommended):
winget install ffmpeg

# Option 2 – Manual:
# 1. Download from https://ffmpeg.org/download.html
# 2. Extract to C:\\ffmpeg
# 3. Add C:\\ffmpeg\\bin to the PATH environment variable
# 4. Or set the FFMPEG_PATH env variable to the ffmpeg.exe path
# 5. Or enter the path in Settings > FFmpeg path
            """, language="bash")
        return

    st.subheader("Convert Audio Files")
    uploaded_files = st.file_uploader(
        "Select audio files",
        type=["m4a","wav","mp3","aac","flac","ogg","opus","mp4"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns(2)
    bitrate    = c1.selectbox("MP3 bitrate", ["64k","96k","128k","160k","192k"], index=2)
    keep_orig  = c2.toggle("Keep original files", value=True)

    if uploaded_files and st.button("Start Conversion", type="primary"):
        input_paths = []
        for uf in uploaded_files:
            tmp = AUDIO_CONV_DIR / uf.name
            tmp.write_bytes(uf.read())
            input_paths.append(tmp)

        with st.spinner("Converting..."):
            results = batch_convert(input_paths, AUDIO_CONV_DIR,
                                    bitrate=bitrate, overwrite=True, keep_original=keep_orig)

        for r in results:
            if r["success"]:
                st.success(f"✅ {Path(r['input']).name} → {Path(r['output']).name}")
                out = Path(r["output"])
                if out.exists():
                    with open(out, "rb") as f:
                        st.download_button(
                            f"⬇ {out.name}", data=f.read(),
                            file_name=out.name, mime="audio/mpeg",
                            key=f"dl_conv_{out.name}_{int(time.time()*1000)}",
                        )
            else:
                st.error(f"❌ {Path(r['input']).name}: {r['message']}")


# ══════════════════════════════════════════════════════════════════════════════
# Page: Transcription Tool
# ══════════════════════════════════════════════════════════════════════════════

_TR_STATUS_ICON = {
    "PENDING":   "⏳",
    "RUNNING":   "🔄",
    "COMPLETED": "✅",
    "FAILED":    "❌",
}


def page_transcription() -> None:
    from transcription_service import is_whisper_available
    from shared_source_service import compute_file_hash
    from models import TranscriptionStatus
    from settings import TRANSCRIPTIONS_DIR

    st.title("📝 Speech-to-Text  —  Whisper")
    saved_settings = db.get_all_settings(DB_PATH)

    # ── Status banner ──────────────────────────────────────────────────────
    whisper_ok, whisper_msg = is_whisper_available()
    if whisper_ok:
        st.success(f"Whisper ready  ·  {whisper_msg}")
    else:
        st.error(f"Whisper not available: {whisper_msg}")
        st.code("pip install faster-whisper", language="bash")

    worker_alive = _is_transcription_worker_alive()
    if worker_alive:
        st.info("⚙ Transcription worker is running in the background.")

    # ── Settings ───────────────────────────────────────────────────────────
    with st.expander("⚙ Whisper Settings", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        model_options = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        saved_model = saved_settings.get("whisper_model", "small")
        model_name = sc1.selectbox(
            "Model",
            model_options,
            index=model_options.index(saved_model) if saved_model in model_options else 2,
            help="Larger models are more accurate but slower and use more RAM.",
        )
        language_options = ["auto", "fa", "en", "ar", "de", "fr", "tr", "ru", "zh", "ja"]
        saved_whisper_language = saved_settings.get("whisper_language", "fa") or "auto"
        language = sc2.selectbox(
            "Language",
            language_options,
            index=(
                language_options.index(saved_whisper_language)
                if saved_whisper_language in language_options
                else 0
            ),
            help="'auto' lets Whisper detect the language automatically.",
        )
        device_options = ["auto", "cpu", "cuda"]
        saved_device = saved_settings.get("whisper_device", "auto")
        device = sc3.selectbox(
            "Device",
            device_options,
            index=device_options.index(saved_device) if saved_device in device_options else 0,
        )
        sc4, sc5 = st.columns(2)
        compute_options = ["auto", "int8", "int8_float16", "float16", "float32"]
        saved_compute = saved_settings.get("whisper_compute", "auto")
        compute = sc4.selectbox(
            "Compute type",
            compute_options,
            index=(
                compute_options.index(saved_compute)
                if saved_compute in compute_options
                else 0
            ),
        )
        beam_size = sc5.number_input("Beam size", min_value=1, max_value=10, value=5)

        oc1, oc2, oc3 = st.columns(3)
        out_srt  = oc1.toggle("SRT output")
        out_vtt  = oc2.toggle("VTT output")
        out_json = oc3.toggle("JSON output")
        timestamps  = st.toggle("Include timestamps in TXT")
        include_hdr = st.toggle("Include metadata header in TXT", value=True)

    # ── File upload + submit ───────────────────────────────────────────────
    st.divider()
    st.subheader("Upload Audio Files")
    uploaded_files = st.file_uploader(
        "Select one or more audio files",
        type=["m4a", "mp3", "wav", "aac", "flac", "ogg", "opus", "mp4"],
        accept_multiple_files=True,
        help="Files are saved locally and processed by the background worker.",
    )

    can_submit = bool(uploaded_files) and whisper_ok and not worker_alive
    submit_help = (
        "Worker is busy — wait for it to finish or clear completed jobs first."
        if worker_alive else
        "Upload files above to enable submission."
    )
    if st.button(
        "▶ Submit Transcription Jobs",
        type="primary",
        disabled=not can_submit,
        help=submit_help,
    ):
        lang_param = "" if language == "auto" else language
        new_tr_ids: list[int] = []
        TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            dest = TRANSCRIPTIONS_DIR / uf.name
            dest.write_bytes(uf.read())
            fh = compute_file_hash(dest)
            tr_id = db.create_transcription(
                input_path=str(dest),
                input_hash=fh,
                model_name=model_name,
                language=lang_param,
                device=device,
                compute_type=compute,
                path=DB_PATH,
            )
            new_tr_ids.append(tr_id)

        _submit_transcription_batch(
            tr_ids=new_tr_ids,
            output_srt=out_srt,
            output_vtt=out_vtt,
            output_json=out_json,
            beam_size=int(beam_size),
            include_header=include_hdr,
            include_timestamps=timestamps,
        )
        st.success(f"Submitted {len(new_tr_ids)} job(s) to the background worker.")
        st.rerun()

    # ── Transcription queue ────────────────────────────────────────────────
    st.divider()
    st.subheader("Transcription Queue")

    col_r, col_c = st.columns([6, 2])
    with col_c:
        if st.button("🗑 Clear completed", width="stretch"):
            all_trs_all = db.list_transcriptions(path=DB_PATH)
            for t in all_trs_all:
                if t.status == TranscriptionStatus.COMPLETED:
                    try:
                        db.delete_transcription(t.id, DB_PATH)
                    except Exception:
                        pass
            st.rerun()

    all_trs = db.list_transcriptions(path=DB_PATH)
    # Filter out any archived/unknown status gracefully
    visible_trs = [
        t for t in all_trs
        if t.status in (
            TranscriptionStatus.PENDING,
            TranscriptionStatus.RUNNING,
            TranscriptionStatus.COMPLETED,
            TranscriptionStatus.FAILED,
        )
    ]

    if not visible_trs:
        st.info("No transcription jobs yet.  Upload files above and click Submit.")
    else:
        any_active = False
        for tr in visible_trs:
            icon = _TR_STATUS_ICON.get(tr.status.value, "?")
            fname = Path(tr.input_path).name

            with st.container(border=True):
                h1, h2 = st.columns([5, 3])
                h1.markdown(f"**{icon} {fname}**")
                status_label = tr.status.value
                if tr.started_at and tr.completed_at:
                    elapsed = int((tr.completed_at - tr.started_at).total_seconds())
                    status_label += f"  ·  {elapsed}s"
                elif tr.started_at:
                    elapsed = int((datetime.utcnow() - tr.started_at).total_seconds())
                    status_label += f"  ·  {elapsed}s elapsed"
                h2.caption(status_label)

                if tr.status == TranscriptionStatus.RUNNING:
                    any_active = True
                    live_p = _tr_live_progress.get(tr.id, tr.progress or 0.0)
                    st.progress(live_p, text=f"{int(live_p * 100)} %  ·  model: {tr.model_name}")
                elif tr.status == TranscriptionStatus.PENDING:
                    any_active = True
                    st.progress(0.0, text="Waiting in queue…")
                elif tr.status == TranscriptionStatus.COMPLETED:
                    st.progress(1.0, text=f"Done  ·  model: {tr.model_name}")
                    # Download buttons
                    dl_cols = st.columns(4)
                    for col, (attr, label) in zip(
                        dl_cols,
                        [("output_txt_path", "TXT"), ("output_srt_path", "SRT"),
                         ("output_vtt_path", "VTT"),  ("output_json_path", "JSON")],
                    ):
                        fp_str = getattr(tr, attr, None)
                        if fp_str and Path(fp_str).exists():
                            fp = Path(fp_str)
                            with open(fp, "rb") as f:
                                col.download_button(
                                    f"⬇ {label}",
                                    data=f.read(),
                                    file_name=fp.name,
                                    key=f"dl_tr_{tr.id}_{label}_{int(time.time()*1000)}",
                                    width="stretch",
                                )
                elif tr.status == TranscriptionStatus.FAILED:
                    st.error(f"Error: {tr.error_message or 'unknown'}")

        # Auto-refresh while work is in progress
        if any_active or _is_transcription_worker_alive():
            time.sleep(1.5)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Settings
# ══════════════════════════════════════════════════════════════════════════════

def page_settings() -> None:
    st.title("⚙ Settings")

    current = db.get_all_settings(DB_PATH)

    with st.form("settings_form"):
        st.subheader("FFmpeg")
        ffmpeg_path = st.text_input(
            "FFmpeg executable path (optional override)",
            value=current.get("ffmpeg_path", ""),
            placeholder=r"C:\ffmpeg\bin\ffmpeg.exe",
        )

        st.subheader("Default Output Language")
        lang_keys = list(LANGUAGES.keys())
        saved_lang = current.get("default_language", "fa")
        default_lang = st.selectbox(
            "Default language for new projects",
            options=lang_keys,
            index=lang_keys.index(saved_lang) if saved_lang in lang_keys else 0,
            format_func=lambda k: LANGUAGES.get(k, k),
        )
        st.caption("This sets the default audio generation language for new projects. You can override it per-project.")

        st.subheader("Audio")
        bitrate  = st.selectbox(
            "Default MP3 bitrate",
            ["64k","96k","128k","160k","192k"],
            index=["64k","96k","128k","160k","192k"].index(current.get("default_bitrate","128k")),
        )
        keep_m4a = st.toggle("Keep M4A by default", value=current.get("keep_m4a","1") == "1")
        cleanup_notebooks = st.toggle(
            "Delete NotebookLM notebooks after successful download",
            value=current.get("cleanup_notebooks", "1") == "1",
            help="Failed notebooks are kept for troubleshooting.",
        )

        st.subheader("Whisper")
        c1, c2, c3, c4 = st.columns(4)
        model_options = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        saved_model = current.get("whisper_model", "small")
        wmodel = c1.selectbox(
            "Default model",
            model_options,
            index=model_options.index(saved_model) if saved_model in model_options else 2,
        )
        wlang    = c2.selectbox("Default language", lang_keys,
                                index=lang_keys.index(current.get("whisper_language","fa"))
                                      if current.get("whisper_language","fa") in lang_keys else 0,
                                format_func=lambda k: LANGUAGES.get(k, k))
        device_options = ["auto", "cpu", "cuda"]
        saved_device = current.get("whisper_device", "auto")
        wdevice = c3.selectbox(
            "Device",
            device_options,
            index=device_options.index(saved_device) if saved_device in device_options else 0,
        )
        compute_options = ["auto", "int8", "int8_float16", "float16", "float32"]
        saved_compute = current.get("whisper_compute", "auto")
        wcompute = c4.selectbox(
            "Compute type",
            compute_options,
            index=(
                compute_options.index(saved_compute)
                if saved_compute in compute_options
                else 0
            ),
        )
        st.caption(
            "Transcriptions run sequentially to avoid loading multiple Whisper "
            "models into memory."
        )

        st.subheader("Runner")
        retry_cnt = st.number_input("Retry count", min_value=0,
                                    value=int(current.get("retry_count","3")))
        log_options = ["DEBUG", "INFO", "WARNING", "ERROR"]
        saved_log_level = current.get("log_level", "INFO")
        log_level = st.selectbox(
            "Log level",
            log_options,
            index=(
                log_options.index(saved_log_level)
                if saved_log_level in log_options
                else 1
            ),
        )

        if st.form_submit_button("Save Settings"):
            db.set_setting("ffmpeg_path",        ffmpeg_path,          DB_PATH)
            db.set_setting("default_language",   default_lang,         DB_PATH)
            db.set_setting("default_bitrate",    bitrate,              DB_PATH)
            db.set_setting("keep_m4a",           "1" if keep_m4a else "0", DB_PATH)
            db.set_setting(
                "cleanup_notebooks",
                "1" if cleanup_notebooks else "0",
                DB_PATH,
            )
            db.set_setting("whisper_model",      wmodel,               DB_PATH)
            db.set_setting("whisper_language",   wlang,                DB_PATH)
            db.set_setting("whisper_device",     wdevice,              DB_PATH)
            db.set_setting("whisper_compute",    wcompute,             DB_PATH)
            db.set_setting("retry_count",        str(int(retry_cnt)),  DB_PATH)
            db.set_setting("log_level",          log_level,            DB_PATH)
            if ffmpeg_path.strip():
                os.environ["FFMPEG_PATH"] = ffmpeg_path.strip()
            st.success("Settings saved.")

    st.divider()
    st.subheader("System Status")
    from audio_service import is_ffmpeg_available
    from transcription_service import is_whisper_available
    ok_ff, msg_ff = is_ffmpeg_available()
    ok_wh, msg_wh = is_whisper_available()
    c1, c2 = st.columns(2)
    c1.write(f"FFmpeg: {'✅' if ok_ff else '❌'} {msg_ff}")
    c2.write(f"Whisper: {'✅' if ok_wh else '❌'} {msg_wh}")
    st.write(f"Python: {sys.version.split()[0]}")
    st.write(f"Runner PID file: {'present' if RUNNER_PID_FILE.exists() else 'absent'}")

    st.divider()
    st.subheader("Runner Control")
    runner_alive = _is_runner_alive()
    if runner_alive:
        st.success("Runner is running.")
        if st.button("⏹ Stop Runner"):
            _stop_runner(); st.rerun()
    else:
        st.warning("Runner is not running.")
        c1, c2 = st.columns(2)
        if c1.button("▶ Start Runner (real)"):
            _launch_runner(fake=False); st.rerun()
        if c2.button("▶ Start Runner (fake/test)"):
            _launch_runner(fake=True); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: Logs
# ══════════════════════════════════════════════════════════════════════════════

def page_logs() -> None:
    st.title("📋 Logs")
    from settings import RUNNER_LOG_FILE

    if RUNNER_LOG_FILE.exists():
        log_text = RUNNER_LOG_FILE.read_text(encoding="utf-8", errors="replace")
        lines = log_text.splitlines()

        c1, c2 = st.columns(2)
        level_filter = c1.selectbox("Filter level", ["All","ERROR","WARNING","INFO","DEBUG"])
        lines_count  = c2.number_input("Last N lines", min_value=10, max_value=2000, value=200)

        if level_filter != "All":
            lines = [l for l in lines if level_filter in l]

        display = lines[-int(lines_count):]
        st.code("\n".join(display), language="text")
        st.caption(f"Total: {len(lines)} line(s)")

        if st.button("Clear log file"):
            RUNNER_LOG_FILE.write_text(""); st.rerun()
    else:
        st.info("Log file not found. Start the Runner to generate logs.")

    if st.button("🔄 Refresh"):
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    page = _sidebar()

    if page == "dashboard":       page_dashboard()
    elif page == "projects":      page_projects()
    elif page == "new_project":   page_new_project()
    elif page == "accounts":      page_accounts()
    elif page == "shared_sources": page_shared_sources()
    elif page == "queue":         page_queue()
    elif page == "files":         page_files()
    elif page == "audio_conv":    page_audio_conversion()
    elif page == "transcribe":    page_transcription()
    elif page == "ai_folder":
        from ai_folder_page import page_ai_folder
        page_ai_folder()
    elif page == "booklet":
        from booklet_page import page_booklet
        page_booklet()
    elif page == "settings":      page_settings()
    elif page == "logs":          page_logs()
    else:                         page_dashboard()


if __name__ == "__main__":
    main()
