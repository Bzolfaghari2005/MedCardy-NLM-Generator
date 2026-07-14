"""
ai_folder_page.py – Streamlit page for AI Folder Processor.

Imported by app.py. Call page_ai_folder() to render the full page.
"""
from __future__ import annotations

import io
import json
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

import database as db
from ai_api_service import build_provider, resolve_api_key
from ai_batch_runner import AiBatchRunner, RunConfig, create_run_and_jobs
from ai_folder_service import (
    DiscoveredFile,
    FolderScanConfig,
    FolderValidationError,
    scan_folder,
    validate_folder,
)
from models import AIChunkMode, AIConnectionStatus, AIFileGroup, AIJobStatus, AIRunStatus
from prompt_service import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    SUPPORTED_PLACEHOLDERS,
    ensure_default_profile,
    render_prompt,
)
from secret_scanner import mask_api_key
from settings import (
    AI_DEFAULT_MODEL,
    AI_MAX_CONCURRENCY,
    GAPGPT_BASE_URL,
    GAPGPT_CDN_URL,
    DB_PATH,
)

# ── Session state keys ────────────────────────────────────────────────────────

_K_SCAN = "ai_scan_result"
_K_API_KEY = "gapgpt_api_key_temp"
_K_PRIVACY_OK = "ai_privacy_confirmed"
_K_RUNNER_THREAD = "ai_runner_thread"
_K_RUNNER_OBJ = "ai_runner_obj"
_K_CURRENT_RUN = "ai_current_run_id"
_K_UPLOAD_FILES = "ai_upload_files"

# ── Privacy warning text ──────────────────────────────────────────────────────

_PRIVACY_WARNING = """
⚠️ **Privacy warning**

Content from selected files will be sent to an **external API** for processing.

Only send confidential files, personal data, passwords, and access keys if you are sure it is safe to do so.
"""


# ══════════════════════════════════════════════════════════════════════════════
# Main page entry point
# ══════════════════════════════════════════════════════════════════════════════

def page_ai_folder() -> None:
    st.title("🤖 AI Folder Processor")
    st.caption("Batch file analysis with GapGPT / OpenAI-compatible API")

    ensure_default_profile(DB_PATH)

    tabs = st.tabs([
        "📁 Select Folder",
        "📋 Select Files",
        "✍️ Prompt",
        "🔌 Model & API",
        "⚙️ Processing Settings",
        "🔍 Preview",
        "▶️ Run Queue",
        "📊 Results",
    ])

    with tabs[0]:
        _tab_folder_selection()
    with tabs[1]:
        _tab_file_selection()
    with tabs[2]:
        _tab_prompt()
    with tabs[3]:
        _tab_model_api()
    with tabs[4]:
        _tab_processing_settings()
    with tabs[5]:
        _tab_preview()
    with tabs[6]:
        _tab_run_queue()
    with tabs[7]:
        _tab_results()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Folder selection
# ══════════════════════════════════════════════════════════════════════════════

def _tab_folder_selection() -> None:
    st.subheader("Input folder selection")

    c1, c2 = st.columns(2)
    with c1:
        input_path = st.text_input(
            "Input folder path",
            value=st.session_state.get("ai_input_folder", ""),
            placeholder=r"D:\MyFiles\Input",
            key="ai_input_folder_input",
        )
    with c2:
        output_path = st.text_input(
            "Output folder path (optional)",
            value=st.session_state.get("ai_output_folder", ""),
            placeholder="Default: {input}/_ai_results",
            key="ai_output_folder_input",
        )

    st.markdown("**Direct file upload**")
    uploaded = st.file_uploader(
        "Or upload files directly",
        accept_multiple_files=True,
        key="ai_file_uploader",
    )
    if uploaded:
        st.session_state[_K_UPLOAD_FILES] = uploaded
        st.info(f"{len(uploaded)} file(s) uploaded.")

    st.divider()
    st.subheader("Scan filters")

    c1, c2, c3 = st.columns(3)
    recursive = c1.toggle("Process subfolders", value=True, key="ai_recursive")
    include_hidden = c2.toggle("Hidden files", value=False, key="ai_include_hidden")
    scan_secrets = c3.toggle("Scan sensitive files", value=True, key="ai_scan_secrets")

    c1, c2, c3 = st.columns(3)
    max_mb = c1.number_input("Max file size (MB)", min_value=0, value=100, key="ai_max_mb")
    allowed_exts = c2.text_input(
        "Allowed extensions (empty = all)",
        placeholder=".txt .pdf .docx",
        key="ai_allowed_exts",
    )
    blocked_exts = c3.text_input(
        "Blocked extensions",
        placeholder=".exe .dll",
        key="ai_blocked_exts",
    )

    if st.button("🔍 Scan folder", type="primary", key="ai_scan_btn"):
        _do_scan(
            input_path=input_path,
            output_path=output_path,
            recursive=recursive,
            include_hidden=include_hidden,
            scan_secrets=scan_secrets,
            max_mb=float(max_mb),
            allowed_exts=allowed_exts,
            blocked_exts=blocked_exts,
        )

    scan = st.session_state.get(_K_SCAN)
    if scan:
        st.success(f"Scan complete: **{scan.total}** file(s) found in `{scan.root}`")


def _do_scan(
    input_path: str,
    output_path: str,
    recursive: bool,
    include_hidden: bool,
    scan_secrets: bool,
    max_mb: float,
    allowed_exts: str,
    blocked_exts: str,
) -> None:
    # Handle uploaded files mode
    if not input_path.strip() and st.session_state.get(_K_UPLOAD_FILES):
        _handle_upload_mode()
        return

    try:
        root = validate_folder(input_path)
    except FolderValidationError as exc:
        st.error(f"Error: {exc}")
        return

    st.session_state["ai_input_folder"] = str(root)

    # Compute output folder
    if output_path.strip():
        out = Path(output_path.strip())
    else:
        from settings import AI_RESULTS_DIR_NAME
        out = root / AI_RESULTS_DIR_NAME
    st.session_state["ai_output_folder"] = str(out)

    def _parse_exts(raw: str) -> Optional[list[str]]:
        if not raw.strip():
            return None
        return [e.strip().lower() for e in raw.split() if e.strip()]

    cfg = FolderScanConfig(
        root=root,
        recursive=recursive,
        include_hidden=include_hidden,
        scan_secrets=scan_secrets,
        max_file_mb=max_mb,
        allowed_extensions=_parse_exts(allowed_exts),
        blocked_extensions=_parse_exts(blocked_exts),
    )

    with st.spinner("Scanning folder..."):
        result = scan_folder(cfg)

    st.session_state[_K_SCAN] = result

    if result.suspicious:
        st.warning(f"⚠️ {len(result.suspicious)} file(s) flagged as potentially sensitive.")
    st.rerun()


def _handle_upload_mode() -> None:
    """Process uploaded files by saving them to a temp directory."""
    import tempfile
    uploads = st.session_state.get(_K_UPLOAD_FILES, [])
    if not uploads:
        st.error("No files uploaded.")
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="ai_folder_upload_"))
    for uf in uploads:
        dest = tmp_dir / uf.name
        dest.write_bytes(uf.getbuffer())

    st.session_state["ai_input_folder"] = str(tmp_dir)
    from settings import AI_RESULTS_DIR_NAME
    st.session_state["ai_output_folder"] = str(tmp_dir / AI_RESULTS_DIR_NAME)

    cfg = FolderScanConfig(root=tmp_dir, recursive=False, scan_secrets=True)
    with st.spinner("Processing uploaded files..."):
        result = scan_folder(cfg)
    st.session_state[_K_SCAN] = result
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: File selection
# ══════════════════════════════════════════════════════════════════════════════

def _tab_file_selection() -> None:
    scan = st.session_state.get(_K_SCAN)
    if not scan:
        st.info("Scan a folder first in the «Select Folder» tab.")
        return

    st.subheader("Discovered files")

    # Summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total files", scan.total)
    col2.metric("Processable", len(scan.supported))
    col3.metric("Skipped", len(scan.skipped))
    col4.metric("Total size", f"{scan.total_size_mb:.1f} MB")

    col1, col2, col3 = st.columns(3)
    col1.metric("Needs Whisper", scan.needs_whisper)
    col2.metric("Needs Vision", scan.needs_vision)
    col3.metric("Suspicious", len(scan.suspicious))

    if scan.suspicious:
        with st.expander(f"⚠️ {len(scan.suspicious)} suspicious file(s)"):
            for f in scan.suspicious:
                st.warning(f"**{f.filename}**: {f.scan_result.risk_summary if f.scan_result else ''}")

    st.divider()

    # File table with enable/disable
    if not scan.files:
        st.info("No files found.")
        return

    st.markdown("**Disable a file** to exclude it from processing.")

    for i, df in enumerate(scan.files):
        col1, col2, col3, col4, col5 = st.columns([0.5, 3, 2, 1.5, 1.5])
        key = f"ai_file_enable_{i}"
        df.enabled = col1.checkbox("", value=df.enabled, key=key, disabled=bool(df.skip_reason))
        col2.write(f"`{df.relative_path}`")
        col3.write(df.extraction_method)
        col4.write(f"{df.file_size_kb:.1f} KB")
        status_color = "🟢" if df.status_label == "Ready" else "🔴" if df.status_label == "Skipped" else "🟡"
        col5.write(f"{status_color} {df.status_label}")

    if st.button("✅ Select all", key="ai_select_all"):
        for df in scan.files:
            if not df.skip_reason:
                df.enabled = True
        st.rerun()

    if st.button("❌ Deselect all", key="ai_deselect_all"):
        for df in scan.files:
            df.enabled = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Prompt
# ══════════════════════════════════════════════════════════════════════════════

def _tab_prompt() -> None:
    st.subheader("Prompt")

    profiles = db.list_ai_prompt_profiles(DB_PATH)
    profile_names = [p.name for p in profiles]
    profile_map = {p.name: p for p in profiles}

    c1, c2 = st.columns([3, 1])
    selected_name = c1.selectbox("Select prompt profile", options=profile_names, key="ai_prompt_profile")
    profile = profile_map.get(selected_name) if selected_name else None

    with st.expander("➕ Create / edit prompt profile", expanded=not profiles):
        new_name = st.text_input("Profile name", value=profile.name if profile else "", key="ai_new_profile_name")
        new_desc = st.text_input("Description", value=profile.description if profile else "", key="ai_new_profile_desc")
        new_group = st.selectbox(
            "File group",
            ["ALL", "TEXT", "PDF", "OFFICE", "CODE", "IMAGE", "AUDIO"],
            index=0,
            key="ai_new_profile_group",
        )
        new_sys = st.text_area(
            "System Prompt",
            value=profile.system_prompt if profile else DEFAULT_SYSTEM_PROMPT,
            height=150,
            key="ai_new_profile_sys",
        )
        new_user = st.text_area(
            "User Prompt Template",
            value=profile.user_prompt_template if profile else DEFAULT_USER_PROMPT_TEMPLATE,
            height=200,
            key="ai_new_profile_user",
        )

        col1, col2, col3 = st.columns(3)
        if col1.button("💾 Save", key="ai_save_profile"):
            if not new_name.strip():
                st.error("Name is required.")
            else:
                _save_profile(new_name, new_desc, new_group, new_sys, new_user, profile)
                st.rerun()

        if profile and col2.button("⭐ Set as default", key="ai_set_default_profile"):
            db.set_default_ai_prompt_profile(profile.id, DB_PATH)
            st.success("Default profile updated.")
            st.rerun()

        if profile and col3.button("🗑️ Delete", key="ai_delete_profile"):
            db.delete_ai_prompt_profile(profile.id, DB_PATH)
            st.rerun()

    if profile:
        st.divider()
        with st.expander("👁️ Prompt preview"):
            preview_content = st.text_area(
                "Sample content",
                value="This is sample text for preview.\nSecond line.",
                height=80,
                key="ai_prompt_preview_content",
            )
            rendered = render_prompt(
                profile.user_prompt_template,
                filename="sample.txt",
                relative_path="folder/sample.txt",
                extension=".txt",
                extraction_method="direct_text",
                file_content=preview_content,
            )
            st.code(rendered, language="text")

        st.markdown("**Supported placeholders:**")
        st.code("  ".join(SUPPORTED_PLACEHOLDERS))


def _save_profile(
    name: str,
    desc: str,
    group: str,
    sys_prompt: str,
    user_template: str,
    existing,
) -> None:
    if existing and existing.name == name:
        db.update_ai_prompt_profile(existing.id, {
            "description": desc,
            "file_group": group,
            "system_prompt": sys_prompt,
            "user_prompt_template": user_template,
        }, DB_PATH)
        st.success(f"Profile «{name}» updated.")
    else:
        try:
            db.create_ai_prompt_profile(
                name=name,
                description=desc,
                file_group=group,
                system_prompt=sys_prompt,
                user_prompt_template=user_template,
                path=DB_PATH,
            )
            st.success(f"Profile «{name}» created.")
        except Exception as exc:
            st.error(f"Error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Model & API
# ══════════════════════════════════════════════════════════════════════════════

def _tab_model_api() -> None:
    st.subheader("Model & API")

    st.info(
        "**GapGPT** is compatible with the OpenAI library.\n\n"
        "Install: `pip install openai`  \n"
        "If international internet is unreliable: `pip install openai -i https://mirror-pypi.runflare.com/simple`  \n"
        "*(Mirror provided by Runflare; not affiliated with GapGPT.)*"
    )

    col1, col2 = st.columns(2)
    base_url = col1.selectbox(
        "Base URL",
        options=[GAPGPT_BASE_URL, GAPGPT_CDN_URL, "custom"],
        key="ai_base_url_select",
    )
    if base_url == "custom":
        base_url = col1.text_input("Custom URL", key="ai_base_url_custom",
                                    value=GAPGPT_BASE_URL)

    model = col2.text_input("Model name", value=AI_DEFAULT_MODEL, key="ai_model_name")

    st.divider()
    st.subheader("API key")

    # Detect existing key
    existing_key = resolve_api_key(st.session_state.get(_K_API_KEY))
    if existing_key:
        st.success(f"API key found: `{mask_api_key(existing_key)}`")
    else:
        st.warning("API key is not configured.")

    key_input = st.text_input(
        "API key (temporary entry)",
        type="password",
        placeholder="sk-...",
        key="ai_key_input",
    )
    if key_input:
        st.session_state[_K_API_KEY] = key_input
        st.success(f"Key received: `{mask_api_key(key_input)}`")

    st.caption(
        "Key priority: `GAPGPT_API_KEY` env var ← `.env` file ← temporary entry above.\n"
        "The key is never stored in the database, logs, or output files."
    )

    if st.button("🔌 Test connection", key="ai_test_connection"):
        _test_connection(model, base_url)

    st.divider()
    st.subheader("Request parameters")

    c1, c2, c3 = st.columns(3)
    c1.number_input("Concurrency", min_value=1, max_value=20,
                    value=AI_MAX_CONCURRENCY, key="ai_concurrency")
    c2.number_input("Timeout (seconds)", min_value=10, max_value=600,
                    value=180, key="ai_timeout")
    c3.number_input("Retry count", min_value=0, max_value=10,
                    value=3, key="ai_retries")


def _test_connection(model: str, base_url: str) -> None:
    key = resolve_api_key(st.session_state.get(_K_API_KEY))
    if not key:
        st.error("API key is not set.")
        return

    provider = build_provider(key, base_url)
    with st.spinner("Testing connection..."):
        status = provider.test_connection(model=model)

    icons = {
        AIConnectionStatus.CONNECTED.value: "✅",
        AIConnectionStatus.INVALID_API_KEY.value: "🔑",
        AIConnectionStatus.INSUFFICIENT_CREDIT.value: "💳",
        AIConnectionStatus.RATE_LIMITED.value: "⏱️",
        AIConnectionStatus.NETWORK_ERROR.value: "🌐",
        AIConnectionStatus.API_ERROR.value: "❌",
    }
    icon = icons.get(status, "❓")
    if status == AIConnectionStatus.CONNECTED.value:
        st.success(f"{icon} {status}")
    else:
        st.error(f"{icon} {status}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Processing settings
# ══════════════════════════════════════════════════════════════════════════════

def _tab_processing_settings() -> None:
    st.subheader("Processing settings")

    col1, col2 = st.columns(2)
    col1.number_input("Max tokens per chunk", min_value=500, max_value=128000,
                      value=6000, key="ai_chunk_tokens")
    col2.number_input("Token overlap", min_value=0, max_value=1000,
                      value=200, key="ai_chunk_overlap")

    chunk_mode = st.selectbox(
        "Chunking mode",
        options=[AIChunkMode.DIRECT.value, AIChunkMode.CHUNKED.value, AIChunkMode.CHUNKED_MERGE.value],
        index=2,
        key="ai_chunk_mode",
        format_func=lambda v: {
            "DIRECT": "Direct (no chunking)",
            "CHUNKED": "Separate chunks",
            "CHUNKED_MERGE": "Chunk + final merge (recommended)",
        }.get(v, v),
    )

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.toggle("Image processing (Vision)", value=False, key="ai_vision_enabled")
    audio_mode = col2.selectbox(
        "Audio/video processing",
        options=["transcribe_and_send", "transcript_only", "skip"],
        key="ai_audio_mode",
        format_func=lambda v: {
            "transcribe_and_send": "Transcribe + send to API",
            "transcript_only": "Transcribe only",
            "skip": "Skip",
        }.get(v, v),
    )
    col3.toggle("ZIP processing", value=False, key="ai_zip_enabled")

    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.toggle("Force reprocess", value=False, key="ai_force_reprocess",
                help="Reprocess files that were already completed")
    col2.toggle("Save metadata JSON", value=False, key="ai_save_json")
    col3.toggle("Send full path to API", value=False, key="ai_send_abs_path",
                help="By default only the relative path is sent")

    output_fmt = st.selectbox(
        "Output format",
        options=["txt", "txt_json", "txt_markdown"],
        key="ai_output_format",
        format_func=lambda v: {
            "txt": "TXT only",
            "txt_json": "TXT + JSON metadata",
            "txt_markdown": "TXT + Markdown",
        }.get(v, v),
    )
    st.toggle("Response text only (no header)", value=False, key="ai_no_header")

    st.divider()
    st.subheader("Optional limits")
    col1, col2 = st.columns(2)
    col1.number_input("Max files in this run (0 = unlimited)",
                      min_value=0, value=0, key="ai_max_files_limit")
    col2.number_input("Max estimated tokens (0 = unlimited)",
                      min_value=0, value=0, key="ai_max_token_limit")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6: Preview
# ══════════════════════════════════════════════════════════════════════════════

def _tab_preview() -> None:
    st.subheader("Processing preview")

    scan = st.session_state.get(_K_SCAN)
    if not scan:
        st.info("Scan a folder first.")
        return

    supported = [f for f in scan.files if f.enabled and not f.skip_reason]
    skipped = [f for f in scan.files if f.skip_reason or not f.enabled]
    chunk_tokens = st.session_state.get("ai_chunk_tokens", 6000)
    model = st.session_state.get("ai_model_name", AI_DEFAULT_MODEL)

    # Estimate tokens
    from chunking_service import estimate_tokens, needs_chunking
    from settings import AI_CHARS_PER_TOKEN

    needs_chunk_count = sum(
        1 for f in supported
        if f.file_size / AI_CHARS_PER_TOKEN > chunk_tokens
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processable files", len(supported))
    col2.metric("Skipped files", len(skipped))
    col3.metric("Needs chunking (est.)", needs_chunk_count)
    col4.metric("Model", model)

    st.divider()

    # Privacy confirmation
    st.markdown(_PRIVACY_WARNING)
    if not st.session_state.get(_K_PRIVACY_OK):
        confirmed = st.checkbox(
            "✅ I understand file content will be sent to the selected API.",
            key="ai_privacy_checkbox",
        )
        if confirmed:
            st.session_state[_K_PRIVACY_OK] = True
            st.rerun()
        else:
            st.warning("Enable the confirmation above to continue.")
    else:
        st.success("✅ Privacy confirmation: done")
        if st.button("Revoke confirmation", key="ai_revoke_privacy"):
            del st.session_state[_K_PRIVACY_OK]
            st.rerun()

    if supported and st.session_state.get(_K_PRIVACY_OK):
        st.divider()
        st.subheader("Optional pricing (estimate)")
        col1, col2 = st.columns(2)
        price_in = col1.number_input("Cost per 1M input tokens ($)", min_value=0.0, value=0.0,
                                      format="%.4f", key="ai_price_in")
        price_out = col2.number_input("Cost per 1M output tokens ($)", min_value=0.0, value=0.0,
                                       format="%.4f", key="ai_price_out")

        total_size = sum(f.file_size for f in supported)
        est_tokens = int(total_size / AI_CHARS_PER_TOKEN)
        est_cost_in = (est_tokens / 1_000_000) * price_in if price_in else 0
        col1.metric("Estimated input tokens", f"{est_tokens:,}")
        if price_in:
            col2.metric("Estimated cost", f"${est_cost_in:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 7: Run queue
# ══════════════════════════════════════════════════════════════════════════════

def _tab_run_queue() -> None:
    st.subheader("Run queue")

    scan = st.session_state.get(_K_SCAN)
    runner_obj: Optional[AiBatchRunner] = st.session_state.get(_K_RUNNER_OBJ)
    runner_thread: Optional[threading.Thread] = st.session_state.get(_K_RUNNER_THREAD)
    current_run_id: Optional[int] = st.session_state.get(_K_CURRENT_RUN)

    running = runner_thread is not None and runner_thread.is_alive()

    col1, col2, col3, col4 = st.columns(4)

    # Start button
    if col1.button("▶️ Start", type="primary", key="ai_start_run",
                   disabled=running or not scan or not st.session_state.get(_K_PRIVACY_OK)):
        _start_run()
        st.rerun()

    # Pause/stop new jobs
    if col2.button("⏸️ Stop new jobs", key="ai_pause_run", disabled=not running):
        if runner_obj:
            runner_obj.stop()
        st.rerun()

    # Retry failed
    if col3.button("🔄 Retry failed", key="ai_retry_failed",
                   disabled=not current_run_id):
        _retry_failed(current_run_id)
        st.rerun()

    # Open output folder
    if col4.button("📂 Open output", key="ai_open_output"):
        out = st.session_state.get("ai_output_folder", "")
        if out:
            import subprocess
            subprocess.Popen(f'explorer "{out}"', shell=True)

    st.divider()

    # Live stats
    if current_run_id:
        run = db.get_ai_batch_run(current_run_id, DB_PATH)
        if run:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total", run.total_files)
            col2.metric("Completed", run.completed_files)
            col3.metric("Failed", run.failed_files)
            col4.metric("Skipped", run.skipped_files)
            col5.metric("Input tokens", f"{run.actual_input_tokens:,}")
            col6.metric("Output tokens", f"{run.actual_output_tokens:,}")

            st.progress(
                (run.completed_files + run.failed_files + run.skipped_files) / max(run.total_files, 1)
            )
            st.caption(f"Run status: **{run.status.value}**")

            # File-level status table
            jobs = db.list_ai_file_jobs(current_run_id, DB_PATH)
            if jobs:
                rows = []
                for j in jobs:
                    rows.append({
                        "Filename": j.input_filename,
                        "Path": j.relative_path,
                        "Status": j.status.value,
                        "Chunk": f"{j.completed_chunk_count}/{j.chunk_count}" if j.chunk_count else "1",
                        "Attempts": j.attempt_count,
                        "Error": j.error_code or "",
                    })
                st.dataframe(rows, width="stretch")

            # Cancel individual job
            job_names = [j.input_filename for j in jobs if j.status not in
                         (AIJobStatus.COMPLETED, AIJobStatus.CANCELLED, AIJobStatus.SKIPPED)]
            if job_names and runner_obj:
                sel = st.selectbox("Cancel job", [""] + job_names, key="ai_cancel_job_sel")
                if sel and st.button("Cancel this job", key="ai_cancel_job_btn"):
                    for j in jobs:
                        if j.input_filename == sel:
                            runner_obj.cancel_job(j.id)
                    st.rerun()

    if running:
        st.button("🔄 Refresh", key="ai_refresh_queue")

    # Past runs
    st.divider()
    with st.expander("📜 Previous runs"):
        all_runs = db.list_ai_batch_runs(DB_PATH)
        if all_runs:
            for r in all_runs[:10]:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.write(f"**#{r.id}** {r.input_folder}")
                col2.write(f"{r.status.value} | {r.completed_files}/{r.total_files}")
                if col3.button("Resume", key=f"ai_resume_{r.id}",
                               disabled=r.status == AIRunStatus.COMPLETED):
                    _resume_run(r.id)
                    st.rerun()
        else:
            st.info("No runs created yet.")


def _start_run() -> None:
    scan = st.session_state.get(_K_SCAN)
    if not scan:
        st.error("Scan a folder first.")
        return

    key = resolve_api_key(st.session_state.get(_K_API_KEY))
    if not key:
        st.error("API key is not set.")
        return

    model = st.session_state.get("ai_model_name", AI_DEFAULT_MODEL)
    base_url = st.session_state.get("ai_base_url_select", GAPGPT_BASE_URL)
    if base_url == "custom":
        base_url = st.session_state.get("ai_base_url_custom", GAPGPT_BASE_URL)

    input_folder = Path(st.session_state.get("ai_input_folder", ""))
    output_folder = Path(st.session_state.get("ai_output_folder", str(input_folder / "_ai_results")))

    profiles = db.list_ai_prompt_profiles(DB_PATH)
    sel_name = st.session_state.get("ai_prompt_profile")
    profile = next((p for p in profiles if p.name == sel_name), None)
    profile_id = profile.id if profile else (profiles[0].id if profiles else None)

    chunk_mode_val = st.session_state.get("ai_chunk_mode", AIChunkMode.CHUNKED_MERGE.value)

    run_id = create_run_and_jobs(
        input_folder=input_folder,
        output_folder=output_folder,
        discovered_files=scan.files,
        model=model,
        base_url=base_url,
        prompt_profile_id=profile_id,
        config_kwargs={
            "max_concurrency": st.session_state.get("ai_concurrency", 3),
            "timeout_seconds": st.session_state.get("ai_timeout", 180),
            "max_retries": st.session_state.get("ai_retries", 3),
            "chunk_max_tokens": st.session_state.get("ai_chunk_tokens", 6000),
            "chunk_overlap_tokens": st.session_state.get("ai_chunk_overlap", 200),
            "chunk_mode": chunk_mode_val,
            "include_hidden_files": st.session_state.get("ai_include_hidden", False),
            "preserve_directory_structure": True,
        },
        db_path=DB_PATH,
    )

    provider = build_provider(key, base_url)

    cfg = RunConfig(
        run_id=run_id,
        input_folder=input_folder,
        output_folder=output_folder,
        model=model,
        base_url=base_url,
        provider=provider,
        max_concurrency=st.session_state.get("ai_concurrency", 3),
        timeout_seconds=st.session_state.get("ai_timeout", 180),
        max_retries=st.session_state.get("ai_retries", 3),
        chunk_max_tokens=st.session_state.get("ai_chunk_tokens", 6000),
        chunk_overlap_tokens=st.session_state.get("ai_chunk_overlap", 200),
        chunk_mode=AIChunkMode(chunk_mode_val),
        vision_enabled=st.session_state.get("ai_vision_enabled", False),
        audio_mode=st.session_state.get("ai_audio_mode", "transcribe_and_send"),
        zip_enabled=st.session_state.get("ai_zip_enabled", False),
        force_reprocess=st.session_state.get("ai_force_reprocess", False),
        include_absolute_path=st.session_state.get("ai_send_abs_path", False),
        output_format=st.session_state.get("ai_output_format", "txt"),
        output_header=not st.session_state.get("ai_no_header", False),
        db_path=DB_PATH,
    )

    runner = AiBatchRunner(cfg)
    thread = threading.Thread(target=runner.run, daemon=True, name=f"ai_run_{run_id}")
    thread.start()

    st.session_state[_K_RUNNER_OBJ] = runner
    st.session_state[_K_RUNNER_THREAD] = thread
    st.session_state[_K_CURRENT_RUN] = run_id
    st.success(f"Run #{run_id} started.")


def _retry_failed(run_id: Optional[int]) -> None:
    if not run_id:
        return
    jobs = db.list_ai_file_jobs(run_id, DB_PATH)
    for j in jobs:
        if j.status == AIJobStatus.FAILED:
            db.update_ai_file_job(j.id, {
                "status": AIJobStatus.DISCOVERED.value,
                "error_code": None,
                "error_message": None,
                "attempt_count": 0,
            }, DB_PATH)
    st.info("Failed files are ready for retry. Start the run again.")


def _resume_run(run_id: int) -> None:
    """Resume a previous (stopped/failed) run."""
    run = db.get_ai_batch_run(run_id, DB_PATH)
    if not run:
        return

    key = resolve_api_key(st.session_state.get(_K_API_KEY))
    if not key:
        st.error("API key is not set.")
        return

    provider = build_provider(key, run.base_url)

    cfg = RunConfig(
        run_id=run_id,
        input_folder=Path(run.input_folder),
        output_folder=Path(run.output_folder),
        model=run.model,
        base_url=run.base_url,
        provider=provider,
        max_concurrency=run.max_concurrency,
        timeout_seconds=run.timeout_seconds,
        max_retries=run.max_retries,
        chunk_max_tokens=run.chunk_max_tokens,
        chunk_overlap_tokens=run.chunk_overlap_tokens,
        chunk_mode=run.chunk_mode,
        db_path=DB_PATH,
    )

    runner = AiBatchRunner(cfg)
    thread = threading.Thread(target=runner.run, daemon=True, name=f"ai_resume_{run_id}")
    thread.start()

    st.session_state[_K_RUNNER_OBJ] = runner
    st.session_state[_K_RUNNER_THREAD] = thread
    st.session_state[_K_CURRENT_RUN] = run_id
    st.success(f"Run #{run_id} resumed.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 8: Results
# ══════════════════════════════════════════════════════════════════════════════

def _tab_results() -> None:
    st.subheader("Results")

    current_run_id = st.session_state.get(_K_CURRENT_RUN)
    all_runs = db.list_ai_batch_runs(DB_PATH)

    if not all_runs:
        st.info("No runs executed yet.")
        return

    run_options = {f"#{r.id} – {r.input_folder} ({r.status.value})": r.id for r in all_runs}
    selected_label = st.selectbox("Select run", list(run_options.keys()), key="ai_result_run")
    run_id = run_options[selected_label]

    jobs = db.list_ai_file_jobs(run_id, DB_PATH)

    col1, col2 = st.columns(2)
    search = col1.text_input("Search by filename", key="ai_result_search")
    status_filter = col2.selectbox(
        "Status filter",
        ["All"] + [s.value for s in AIJobStatus],
        key="ai_result_status_filter",
    )

    filtered = jobs
    if search:
        filtered = [j for j in filtered if search.lower() in j.input_filename.lower()]
    if status_filter != "All":
        filtered = [j for j in filtered if j.status.value == status_filter]

    st.caption(f"Showing {len(filtered)} of {len(jobs)} file(s)")

    for j in filtered:
        with st.expander(f"{j.input_filename} – {j.status.value}"):
            col1, col2 = st.columns(2)
            col1.write(f"**Path:** `{j.relative_path}`")
            col1.write(f"**Model:** {j.model}")
            col1.write(f"**Input tokens:** {j.input_tokens or 'N/A'}")
            col2.write(f"**Status:** {j.status.value}")
            col2.write(f"**Output tokens:** {j.output_tokens or 'N/A'}")
            col2.write(f"**Chunk count:** {j.chunk_count}")

            if j.error_code:
                st.error(f"Error: `{j.error_code}` – {j.error_message}")

            if j.output_txt_path and Path(j.output_txt_path).exists():
                content = Path(j.output_txt_path).read_text(encoding="utf-8")
                st.text_area("Result", value=content[:3000], height=200,
                             key=f"ai_result_text_{j.id}", disabled=True)
                st.download_button(
                    "⬇️ Download",
                    data=content.encode("utf-8"),
                    file_name=Path(j.output_txt_path).name,
                    mime="text/plain",
                    key=f"ai_download_{j.id}",
                )

    st.divider()

    completed_jobs = [j for j in jobs if j.output_txt_path and Path(j.output_txt_path).exists()]
    if completed_jobs:
        if st.button("📦 Download all results (ZIP)", key="ai_download_all_zip"):
            zip_bytes = _build_zip(completed_jobs)
            st.download_button(
                "⬇️ Download ZIP",
                data=zip_bytes,
                file_name=f"ai_results_run_{run_id}.zip",
                mime="application/zip",
                key="ai_zip_download",
            )


def _build_zip(jobs: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for j in jobs:
            p = Path(j.output_txt_path)
            if p.exists():
                zf.write(p, arcname=j.relative_path.replace("\\", "/") + "_ai_result.txt")
    buf.seek(0)
    return buf.read()
