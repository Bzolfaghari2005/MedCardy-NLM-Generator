"""
booklet_page.py – Streamlit UI for the Word Booklet Maker.

Seven tabs:
  1. Source selection
  2. File ordering
  3. Chapters
  4. Booklet design
  5. Preview
  6. Build Word
  7. Outputs
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from booklet_sort_service import (
    SORT_MODE_LABELS, FileEntry,
    apply_manual_orders, find_duplicate_hashes, find_duplicate_titles,
    move_entry_down, move_entry_to_bottom, move_entry_to_top, move_entry_up,
    scan_folder, sort_entries,
)
from booklet_parser import parse_files
from booklet_service import (
    delete_booklet_project, delete_preset, get_preset,
    list_booklet_projects, list_presets, run_build, save_preset,
    sanitize_filename,
)
from settings import (
    BOOKLET_BODY_FONT_SIZE_PT, BOOKLET_DEFAULT_FONT_ENGLISH,
    BOOKLET_DEFAULT_FONT_HEADING, BOOKLET_DEFAULT_FONT_PERSIAN,
    BOOKLET_H1_FONT_SIZE_PT, BOOKLET_H2_FONT_SIZE_PT,
    BOOKLET_H3_FONT_SIZE_PT, BOOKLET_HEADING_COLOR_HEX,
    BOOKLET_LINE_SPACING, BOOKLET_MARGIN_BOTTOM_CM,
    BOOKLET_MARGIN_LEFT_CM, BOOKLET_MARGIN_RIGHT_CM,
    BOOKLET_MARGIN_TOP_CM, BOOKLET_SPACE_AFTER_PT,
    BOOKLET_TABLE_HEADER_COLOR_HEX, BOOKLETS_DIR,
)

_SOURCE_CUSTOM = "Custom folder"
_SOURCE_AI_BATCH = "AI Batch Run output"
_ON_ERROR_SKIP = "Skip file and continue"
_ON_ERROR_STOP = "Stop booklet build"
_PRESET_SELECT = "— Select —"
_PREVIEW_FIRST = "First chapter"
_PREVIEW_SELECT = "Select a chapter"
_PREVIEW_ALL = "All chapters (HTML)"


# ══════════════════════════════════════════════════════════════════════════════
# Session state keys
# ══════════════════════════════════════════════════════════════════════════════

def _ss(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _init_state():
    _ss("bk_entries", [])
    _ss("bk_source_folder", "")
    _ss("bk_scanned", False)
    _ss("bk_sort_mode", "NATURAL")
    _ss("bk_chapters_parsed", False)
    _ss("bk_chapters", [])
    _ss("bk_settings", _default_settings())
    _ss("bk_output_filename", "booklet.docx")
    _ss("bk_project_name", "New booklet")
    _ss("bk_last_result", None)
    _ss("bk_first_h1_behavior", "USE_AS_TITLE")
    _ss("bk_on_error", "SKIP")
    _ss("bk_save_merged_preview", True)
    _ss("bk_extensions", [".txt", ".md", ".markdown"])
    _ss("bk_recursive", True)
    _ss("bk_number_regex", r"(\d+)")


def _default_settings() -> dict:
    return {
        "title": "",
        "subtitle": "",
        "course_name": "",
        "university_name": "",
        "author_name": "",
        "ai_note": "",
        "logo_path": "",
        "date_str": datetime.now().strftime("%Y-%m-%d"),
        "font_persian": BOOKLET_DEFAULT_FONT_PERSIAN,
        "font_english": BOOKLET_DEFAULT_FONT_ENGLISH,
        "font_heading": BOOKLET_DEFAULT_FONT_HEADING,
        "body_size": BOOKLET_BODY_FONT_SIZE_PT,
        "h1_size": BOOKLET_H1_FONT_SIZE_PT,
        "h2_size": BOOKLET_H2_FONT_SIZE_PT,
        "h3_size": BOOKLET_H3_FONT_SIZE_PT,
        "h4_size": BOOKLET_H3_FONT_SIZE_PT,
        "line_spacing": BOOKLET_LINE_SPACING,
        "space_after_pt": BOOKLET_SPACE_AFTER_PT,
        "margin_top_cm": BOOKLET_MARGIN_TOP_CM,
        "margin_bottom_cm": BOOKLET_MARGIN_BOTTOM_CM,
        "margin_right_cm": BOOKLET_MARGIN_RIGHT_CM,
        "margin_left_cm": BOOKLET_MARGIN_LEFT_CM,
        "heading_color_hex": BOOKLET_HEADING_COLOR_HEX,
        "table_header_color_hex": BOOKLET_TABLE_HEADER_COLOR_HEX,
        "accent_color_hex": "2E75B6",
        "include_cover": True,
        "include_toc": True,
        "include_header": True,
        "include_footer": True,
        "include_page_numbers": True,
        "chapter_numbering_mode": "NONE",
        "first_h1_behavior": "USE_AS_TITLE",
        "show_source_filename": "NONE",
        "overwrite_mode": "NEW_VERSION",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _fmt_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 – Source Selection
# ══════════════════════════════════════════════════════════════════════════════

def _tab_source():
    st.subheader("📂 Source selection")

    source_type = st.radio(
        "Source type",
        [_SOURCE_CUSTOM, _SOURCE_AI_BATCH],
        horizontal=True,
        key="bk_source_type_radio",
    )

    folder_path = ""

    if source_type == _SOURCE_CUSTOM:
        folder_path = st.text_input(
            "Folder path",
            value=st.session_state.get("bk_source_folder", ""),
            placeholder=r"e.g. D:\MedicalNotes\AIResults",
            key="bk_folder_input",
        )
    else:
        # AI Batch Run picker
        try:
            from database import get_db
            from settings import DB_PATH
            with get_db(DB_PATH) as conn:
                runs = conn.execute(
                    "SELECT id, name, output_folder, created_at FROM ai_batch_runs ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
            if runs:
                run_labels = {f"[{r['id']}] {r['name']} — {r['created_at'][:10]}": r for r in runs}
                selected_label = st.selectbox("Select batch run", list(run_labels.keys()),
                                               key="bk_batch_run_sel")
                selected_run = run_labels[selected_label]
                folder_path = selected_run["output_folder"] or ""
                if folder_path:
                    st.info(f"Output path: `{folder_path}`")
            else:
                st.info("No batch runs found.")
        except Exception:
            st.info("AI Folder module is unavailable. Use a custom folder instead.")

    st.divider()
    st.markdown("**Scannable formats**")
    col1, col2, col3, col4, col5 = st.columns(5)
    ext_txt  = col1.checkbox(".txt",      value=True,  key="bk_ext_txt")
    ext_md   = col2.checkbox(".md",       value=True,  key="bk_ext_md")
    ext_mark = col3.checkbox(".markdown", value=True,  key="bk_ext_mark")
    ext_json = col4.checkbox(".json",     value=False, key="bk_ext_json")
    ext_rst  = col5.checkbox(".rst",      value=False, key="bk_ext_rst")

    extensions = []
    if ext_txt:  extensions.append(".txt")
    if ext_md:   extensions.append(".md")
    if ext_mark: extensions.append(".markdown")
    if ext_json: extensions.append(".json")
    if ext_rst:  extensions.append(".rst")
    st.session_state["bk_extensions"] = extensions

    recursive = st.checkbox("Scan subfolders", value=True, key="bk_recursive_chk")
    st.session_state["bk_recursive"] = recursive

    st.divider()
    if st.button("🔍 Scan folder", type="primary", key="bk_scan_btn"):
        if not folder_path:
            st.error("Please enter a folder path.")
            return
        fp = Path(folder_path)
        if not fp.is_dir():
            st.error(f"Folder not found: `{folder_path}`")
            return

        with st.spinner("Scanning files…"):
            entries = scan_folder(
                fp,
                extensions=extensions,
                recursive=recursive,
                number_regex=st.session_state.get("bk_number_regex", r"(\d+)"),
            )
            entries = sort_entries(entries, "NATURAL")

        st.session_state["bk_source_folder"] = folder_path
        st.session_state["bk_entries"] = entries
        st.session_state["bk_scanned"] = True
        st.session_state["bk_chapters_parsed"] = False
        st.session_state["bk_chapters"] = []
        st.rerun()

    if st.session_state.get("bk_scanned"):
        entries: list[FileEntry] = st.session_state["bk_entries"]
        readable = [e for e in entries if e.readable]
        unreadable = [e for e in entries if not e.readable]

        st.success(f"✅ {len(readable)} readable file(s) found.")
        if unreadable:
            st.warning(f"⚠️ {len(unreadable)} corrupt or empty file(s) (shown in table)")

        # Duplicate warnings
        dup_hash = find_duplicate_hashes(entries)
        dup_title = find_duplicate_titles([e for e in entries if e.enabled])
        if dup_hash:
            with st.expander(f"⚠️ {len(dup_hash)} duplicate file group(s) (identical content)"):
                for h, lst in dup_hash.items():
                    st.write(", ".join(e.filename for e in lst))
        if dup_title:
            with st.expander(f"⚠️ {len(dup_title)} duplicate title(s)"):
                for t, lst in dup_title.items():
                    st.write(f"**{t}**: " + ", ".join(e.filename for e in lst))

        # Preview table
        table_data = []
        for e in entries:
            status = "✅" if e.readable else f"❌ {e.error_message}"
            table_data.append({
                "Order": e.sort_order,
                "File": e.filename,
                "Detected title": e.detected_title[:60] or "—",
                "Size": _fmt_size(e.file_size),
                "Modified": _fmt_dt(e.modified_time),
                "Tables": e.table_count,
                "Words": e.word_count,
                "Status": status,
            })
        st.dataframe(table_data, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 – File Ordering
# ══════════════════════════════════════════════════════════════════════════════

def _tab_ordering():
    st.subheader("📋 File order")

    if not st.session_state.get("bk_scanned"):
        st.info("Scan a folder first in the «Source selection» tab.")
        return

    entries: list[FileEntry] = st.session_state["bk_entries"]
    if not entries:
        st.warning("No files found.")
        return

    sort_labels = list(SORT_MODE_LABELS.values())
    sort_keys = list(SORT_MODE_LABELS.keys())

    current_mode = st.session_state.get("bk_sort_mode", "NATURAL")
    current_idx = sort_keys.index(current_mode) if current_mode in sort_keys else 2

    selected_label = st.selectbox("Sort method", sort_labels,
                                   index=current_idx, key="bk_sort_mode_sel")
    selected_mode = sort_keys[sort_labels.index(selected_label)]

    if selected_mode == "EXTRACTED_NUM":
        st.session_state["bk_number_regex"] = st.text_input(
            "Number extraction pattern (regex)",
            value=st.session_state.get("bk_number_regex", r"(\d+)"),
            key="bk_num_regex_input",
        )

    if st.button("Sort", key="bk_sort_btn"):
        st.session_state["bk_sort_mode"] = selected_mode
        sort_entries(entries, selected_mode)
        st.session_state["bk_entries"] = entries
        st.session_state["bk_chapters_parsed"] = False
        st.rerun()

    st.divider()
    st.markdown("**Manual order editing**")

    # Build editable table with up/down buttons
    for i, entry in enumerate(entries):
        col_order, col_en, col_name, col_title, col_up, col_dn, col_top, col_bot = st.columns(
            [1, 0.8, 3, 3, 0.7, 0.7, 0.7, 0.7]
        )
        col_order.markdown(f"**{entry.sort_order}**")
        enabled = col_en.checkbox("", value=entry.enabled, key=f"bk_en_{i}",
                                    label_visibility="collapsed")
        entries[i].enabled = enabled

        col_name.markdown(entry.filename)

        new_title = col_title.text_input(
            "", value=entry.custom_title or entry.detected_title,
            key=f"bk_title_{i}", label_visibility="collapsed",
            placeholder=entry.detected_title or entry.filename,
        )
        entries[i].custom_title = new_title

        if col_up.button("⬆", key=f"bk_up_{i}"):
            move_entry_up(entries, i)
            st.session_state["bk_entries"] = entries
            st.session_state["bk_chapters_parsed"] = False
            st.rerun()
        if col_dn.button("⬇", key=f"bk_dn_{i}"):
            move_entry_down(entries, i)
            st.session_state["bk_entries"] = entries
            st.session_state["bk_chapters_parsed"] = False
            st.rerun()
        if col_top.button("⏫", key=f"bk_top_{i}"):
            move_entry_to_top(entries, i)
            st.session_state["bk_entries"] = entries
            st.session_state["bk_chapters_parsed"] = False
            st.rerun()
        if col_bot.button("⏬", key=f"bk_bot_{i}"):
            move_entry_to_bottom(entries, i)
            st.session_state["bk_entries"] = entries
            st.session_state["bk_chapters_parsed"] = False
            st.rerun()

    st.session_state["bk_entries"] = entries

    date_src_examples = set(e.created_time_source for e in entries)
    if "mtime_fallback" in date_src_examples:
        st.caption("ℹ️ True creation time is unavailable; modified time is used instead.")
    elif "ctime" in date_src_examples:
        st.caption("ℹ️ Creation time read from st_ctime (Windows NTFS).")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 – Chapters
# ══════════════════════════════════════════════════════════════════════════════

def _tab_chapters():
    st.subheader("📚 Chapters")

    if not st.session_state.get("bk_scanned"):
        st.info("Scan a folder first.")
        return

    entries: list[FileEntry] = st.session_state["bk_entries"]
    active = [e for e in entries if e.enabled and e.readable]

    if not active:
        st.warning("No enabled readable files.")
        return

    first_h1_options = {
        "USE_AS_TITLE": "Use as chapter title and remove from body",
        "KEEP_IN_BODY":  "Keep in body",
        "DEMOTE_TO_H2":  "Demote to heading level 2",
    }
    current_beh = st.session_state.get("bk_first_h1_behavior", "USE_AS_TITLE")
    selected_beh_label = st.selectbox(
        "First H1 heading behavior per file",
        list(first_h1_options.values()),
        index=list(first_h1_options.keys()).index(current_beh),
        key="bk_h1_behavior_sel",
    )
    selected_beh = list(first_h1_options.keys())[list(first_h1_options.values()).index(selected_beh_label)]
    st.session_state["bk_first_h1_behavior"] = selected_beh

    on_error_mode = st.radio(
        "When a file fails to parse",
        [_ON_ERROR_SKIP, _ON_ERROR_STOP],
        horizontal=True, key="bk_on_error_radio",
    )
    st.session_state["bk_on_error"] = "SKIP" if on_error_mode == _ON_ERROR_SKIP else "STOP"

    if st.button("🔄 Parse chapters", type="primary", key="bk_parse_btn"):
        paths = [e.source_path for e in active]
        progress_bar = st.progress(0, text="Parsing…")

        def _cb(i, total, name):
            progress_bar.progress((i + 1) / max(total, 1), text=f"Parse: {name}")

        chapters = parse_files(
            paths,
            first_h1_behavior=selected_beh,
            on_error=st.session_state.get("bk_on_error", "SKIP"),
            progress_callback=_cb,
        )
        progress_bar.empty()
        st.session_state["bk_chapters"] = chapters
        st.session_state["bk_chapters_parsed"] = True
        # Update entry metadata from parsed chapters
        for i, (entry, ch) in enumerate(zip(active, chapters)):
            entry.table_count = ch.table_count
        st.rerun()

    if st.session_state.get("bk_chapters_parsed"):
        chapters = st.session_state["bk_chapters"]
        st.success(f"✅ {len(chapters)} chapter(s) parsed.")

        total_words = sum(ch.raw_word_count for ch in chapters)
        total_tables = sum(ch.table_count for ch in chapters)
        invalid_tables = sum(ch.invalid_table_count for ch in chapters)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total words", f"{total_words:,}")
        m2.metric("Valid tables", total_tables)
        m3.metric("Invalid tables", invalid_tables)

        for i, (ch, entry) in enumerate(zip(chapters, active)):
            with st.expander(f"Chapter {i+1}: {ch.title}", expanded=False):
                warns = ch.warnings
                if warns:
                    for w in warns:
                        st.warning(w)
                st.caption(f"Words: {ch.raw_word_count} | Tables: {ch.table_count} | Invalid tables: {ch.invalid_table_count}")
                st.caption(f"Source: `{entry.relative_path}`")
                try:
                    preview_text = ch.source_path.read_text(encoding="utf-8-sig")[:1000]
                    st.text_area("Content preview", preview_text, height=150,
                                  key=f"bk_ch_prev_{i}", disabled=True)
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 – Design
# ══════════════════════════════════════════════════════════════════════════════

def _tab_design():
    st.subheader("🎨 Booklet design")

    settings: dict = st.session_state.get("bk_settings", _default_settings())

    # ── Presets ──────────────────────────────────────────────────────────────
    with st.expander("💾 Presets", expanded=False):
        presets = list_presets()
        if presets:
            preset_labels = {p["name"]: p for p in presets}
            sel_preset = st.selectbox("Load preset", [_PRESET_SELECT] + list(preset_labels.keys()),
                                       key="bk_preset_load_sel")
            if sel_preset != _PRESET_SELECT:
                if st.button("Load", key="bk_load_preset_btn"):
                    p = preset_labels[sel_preset]
                    try:
                        loaded = json.loads(p.get("settings_json", "{}"))
                        settings.update(loaded)
                        st.session_state["bk_settings"] = settings
                        st.success(f"Preset '{sel_preset}' loaded.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        col_pn, col_pd, col_psave = st.columns([2, 3, 1])
        preset_name = col_pn.text_input("New preset name", key="bk_new_preset_name")
        preset_desc = col_pd.text_input("Description", key="bk_new_preset_desc")
        if col_psave.button("Save", key="bk_save_preset_btn"):
            if preset_name.strip():
                save_preset(preset_name.strip(), preset_desc, settings)
                st.success(f"Preset '{preset_name}' saved.")
            else:
                st.error("Enter a preset name.")

    # ── Cover ─────────────────────────────────────────────────────────────────
    with st.expander("📄 Cover page", expanded=True):
        settings["include_cover"] = st.checkbox("Add cover page",
                                                  value=settings.get("include_cover", True),
                                                  key="bk_incl_cover")
        if settings["include_cover"]:
            settings["title"]           = st.text_input("Booklet title",
                                                          value=settings.get("title", ""),
                                                          key="bk_title")
            settings["subtitle"]        = st.text_input("Subtitle",
                                                          value=settings.get("subtitle", ""),
                                                          key="bk_subtitle")
            c1, c2 = st.columns(2)
            settings["course_name"]     = c1.text_input("Course name",
                                                          value=settings.get("course_name", ""),
                                                          key="bk_course")
            settings["university_name"] = c2.text_input("University name",
                                                          value=settings.get("university_name", ""),
                                                          key="bk_uni")
            settings["author_name"]     = st.text_input("Author name",
                                                          value=settings.get("author_name", ""),
                                                          key="bk_author")
            settings["ai_note"]         = st.text_input("Optional note (e.g. generated with AI)",
                                                          value=settings.get("ai_note", ""),
                                                          key="bk_ainote")
            logo_file = st.file_uploader("Logo (optional)", type=["png", "jpg", "jpeg"],
                                           key="bk_logo_upload")
            if logo_file:
                logo_tmp = BOOKLETS_DIR / "_tmp_logo" / logo_file.name
                logo_tmp.parent.mkdir(parents=True, exist_ok=True)
                logo_tmp.write_bytes(logo_file.read())
                settings["logo_path"] = str(logo_tmp)
                st.image(str(logo_tmp), width=120)
            settings["date_str"] = st.text_input("Generation date",
                                                   value=settings.get("date_str", ""),
                                                   key="bk_datestr")

    # ── TOC / Header / Footer ─────────────────────────────────────────────────
    with st.expander("📑 Table of contents & header/footer"):
        c1, c2, c3, c4 = st.columns(4)
        settings["include_toc"]          = c1.checkbox("Table of contents", value=settings.get("include_toc", True), key="bk_toc")
        settings["include_header"]       = c2.checkbox("Header", value=settings.get("include_header", True), key="bk_hdr")
        settings["include_footer"]       = c3.checkbox("Footer", value=settings.get("include_footer", True), key="bk_ftr")
        settings["include_page_numbers"] = c4.checkbox("Page numbers", value=settings.get("include_page_numbers", True), key="bk_pgnum")

        num_mode_labels = {
            "NONE":       "No numbering",
            "CHAPTER_FA": "Chapter 1 (Persian numerals)",
            "SECTION_FA": "Section 1 (Persian numerals)",
            "CHAPTER_EN": "Chapter 1",
            "NUMBER_DOT": "1.",
        }
        num_mode_keys = list(num_mode_labels.keys())
        current_nm = settings.get("chapter_numbering_mode", "NONE")
        selected_nm = st.selectbox(
            "Chapter numbering style",
            list(num_mode_labels.values()),
            index=num_mode_keys.index(current_nm) if current_nm in num_mode_keys else 0,
            key="bk_num_mode",
        )
        settings["chapter_numbering_mode"] = num_mode_keys[list(num_mode_labels.values()).index(selected_nm)]

        src_labels = {
            "NONE":          "Do not show",
            "BELOW_TITLE":   "Below chapter title",
            "END_OF_CHAPTER":"End of chapter",
        }
        src_keys = list(src_labels.keys())
        current_src = settings.get("show_source_filename", "NONE")
        selected_src = st.selectbox(
            "Show source filename",
            list(src_labels.values()),
            index=src_keys.index(current_src) if current_src in src_keys else 0,
            key="bk_src_mode",
        )
        settings["show_source_filename"] = src_keys[list(src_labels.values()).index(selected_src)]

    # ── Fonts ─────────────────────────────────────────────────────────────────
    with st.expander("🔤 Fonts"):
        c1, c2, c3 = st.columns(3)
        settings["font_persian"] = c1.text_input("Persian font",
                                                   value=settings.get("font_persian", BOOKLET_DEFAULT_FONT_PERSIAN),
                                                   key="bk_font_fa")
        settings["font_english"] = c2.text_input("English font",
                                                   value=settings.get("font_english", BOOKLET_DEFAULT_FONT_ENGLISH),
                                                   key="bk_font_en")
        settings["font_heading"] = c3.text_input("Heading font",
                                                   value=settings.get("font_heading", BOOKLET_DEFAULT_FONT_HEADING),
                                                   key="bk_font_hd")
        st.caption("If a font is not installed, a fallback is used and a warning is shown.")

    # ── Font sizes ────────────────────────────────────────────────────────────
    with st.expander("📏 Sizes"):
        c1, c2, c3, c4, c5 = st.columns(5)
        settings["body_size"] = c1.number_input("Body (pt)", min_value=8, max_value=20,
                                                  value=settings.get("body_size", BOOKLET_BODY_FONT_SIZE_PT),
                                                  key="bk_sz_body")
        settings["h1_size"]   = c2.number_input("H1 (pt)",  min_value=10, max_value=36,
                                                  value=settings.get("h1_size", BOOKLET_H1_FONT_SIZE_PT),
                                                  key="bk_sz_h1")
        settings["h2_size"]   = c3.number_input("H2 (pt)",  min_value=10, max_value=30,
                                                  value=settings.get("h2_size", BOOKLET_H2_FONT_SIZE_PT),
                                                  key="bk_sz_h2")
        settings["h3_size"]   = c4.number_input("H3 (pt)",  min_value=10, max_value=24,
                                                  value=settings.get("h3_size", BOOKLET_H3_FONT_SIZE_PT),
                                                  key="bk_sz_h3")
        settings["line_spacing"] = c5.number_input("Line spacing", min_value=1.0, max_value=3.0, step=0.05,
                                                     value=float(settings.get("line_spacing", BOOKLET_LINE_SPACING)),
                                                     key="bk_lsp")

    # ── Margins ────────────────────────────────────────────────────────────────
    with st.expander("📐 Margins (cm)"):
        c1, c2, c3, c4 = st.columns(4)
        settings["margin_top_cm"]    = c1.number_input("Top", min_value=0.5, max_value=5.0, step=0.1,
                                                         value=float(settings.get("margin_top_cm", BOOKLET_MARGIN_TOP_CM)),
                                                         key="bk_mg_top")
        settings["margin_bottom_cm"] = c2.number_input("Bottom", min_value=0.5, max_value=5.0, step=0.1,
                                                         value=float(settings.get("margin_bottom_cm", BOOKLET_MARGIN_BOTTOM_CM)),
                                                         key="bk_mg_bot")
        settings["margin_right_cm"]  = c3.number_input("Right", min_value=0.5, max_value=5.0, step=0.1,
                                                         value=float(settings.get("margin_right_cm", BOOKLET_MARGIN_RIGHT_CM)),
                                                         key="bk_mg_right")
        settings["margin_left_cm"]   = c4.number_input("Left", min_value=0.5, max_value=5.0, step=0.1,
                                                         value=float(settings.get("margin_left_cm", BOOKLET_MARGIN_LEFT_CM)),
                                                         key="bk_mg_left")

    # ── Colors ─────────────────────────────────────────────────────────────────
    with st.expander("🎨 Colors"):
        c1, c2, c3 = st.columns(3)
        settings["heading_color_hex"]       = c1.text_input("Heading color (hex)",
                                                              value=settings.get("heading_color_hex", BOOKLET_HEADING_COLOR_HEX),
                                                              key="bk_clr_hd")
        settings["table_header_color_hex"]  = c2.text_input("Table header color (hex)",
                                                              value=settings.get("table_header_color_hex", BOOKLET_TABLE_HEADER_COLOR_HEX),
                                                              key="bk_clr_tbl")
        settings["accent_color_hex"]        = c3.text_input("Accent color (hex)",
                                                              value=settings.get("accent_color_hex", "2E75B6"),
                                                              key="bk_clr_acc")

    st.session_state["bk_settings"] = settings


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 – Preview
# ══════════════════════════════════════════════════════════════════════════════

def _tab_preview():
    st.subheader("👁 Structural preview")

    if not st.session_state.get("bk_chapters_parsed"):
        st.info("Parse chapters first in the «Chapters» tab.")
        return

    chapters = st.session_state.get("bk_chapters", [])
    entries: list[FileEntry] = [e for e in st.session_state.get("bk_entries", [])
                                 if e.enabled and e.readable]
    settings: dict = st.session_state.get("bk_settings", _default_settings())
    active_chapters = [ch for ch in chapters if ch.blocks or ch.title]

    preview_opts = [_PREVIEW_FIRST, _PREVIEW_SELECT, _PREVIEW_ALL]
    preview_mode = st.radio("Preview mode", preview_opts, horizontal=True, key="bk_prev_mode")

    if preview_mode == _PREVIEW_SELECT:
        ch_labels = [f"{i+1}. {ch.title}" for i, ch in enumerate(active_chapters)]
        sel_ch = st.selectbox("Select chapter", ch_labels, key="bk_prev_ch_sel")
        sel_idx = ch_labels.index(sel_ch) if sel_ch in ch_labels else 0
        chapters_to_show = [active_chapters[sel_idx]] if active_chapters else []
    elif preview_mode == _PREVIEW_FIRST:
        chapters_to_show = active_chapters[:1]
    else:
        chapters_to_show = active_chapters[:10]  # Limit for browser

    from booklet_parser import (
        BLOCK_BULLET_LIST, BLOCK_CODE_BLOCK, BLOCK_HEADING, BLOCK_HR,
        BLOCK_NOTE, BLOCK_NUMBERED_LIST, BLOCK_PARAGRAPH, BLOCK_QUOTE,
        BLOCK_TABLE, InlineRun, TableData, sanitize_for_docx,
    )
    from booklet_parser import _NOTE_EMOJIS

    _NOTE_COLORS = {
        "GOLDEN_TIP": "#FFF9C4", "IMPORTANT": "#E8F5E9",
        "CLINICAL_EXAMPLE": "#E3F2FD", "CLINICAL_PEARL": "#E0F7FA",
        "WARNING": "#FFEBEE", "HIGH_YIELD": "#FBE9E7",
        "DATA": "#ECEFF1", "SUMMARY": "#EFEBE9",
        "DRUG_NOTE": "#F3E5F5", "EXAM_FINDING": "#E0F2F1",
    }

    def _runs_to_html(runs) -> str:
        parts = []
        for r in (runs if isinstance(runs, list) else []):
            if isinstance(r, InlineRun):
                text = r.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if r.code:
                    text = f"<code>{text}</code>"
                if r.bold:
                    text = f"<strong>{text}</strong>"
                if r.italic:
                    text = f"<em>{text}</em>"
                parts.append(text)
            elif isinstance(r, str):
                parts.append(r.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return "".join(parts)

    def _block_to_html(block) -> str:
        t = block.block_type
        if t == BLOCK_HEADING:
            level = min(block.level or 1, 6)
            return f"<h{level} style='color:#{settings.get('heading_color_hex','1F4E79')};direction:rtl'>{_runs_to_html(block.content)}</h{level}>"
        elif t == BLOCK_PARAGRAPH:
            return f"<p style='direction:rtl;text-align:justify'>{_runs_to_html(block.content)}</p>"
        elif t == BLOCK_NOTE:
            note_type = (block.metadata or {}).get("note_type", "IMPORTANT")
            bg = _NOTE_COLORS.get(note_type, "#F5F5F5")
            return f"<div style='background:{bg};padding:8px 12px;border-right:4px solid #888;margin:6px 0;direction:rtl'>{_runs_to_html(block.content)}</div>"
        elif t == BLOCK_BULLET_LIST:
            items_html = "".join(
                f"<li style='direction:rtl'>{_runs_to_html(item)}</li>"
                for item in (block.content or [])
            )
            return f"<ul style='direction:rtl'>{items_html}</ul>"
        elif t == BLOCK_NUMBERED_LIST:
            items_html = "".join(
                f"<li style='direction:rtl'>{_runs_to_html(item)}</li>"
                for item in (block.content or [])
            )
            return f"<ol style='direction:rtl'>{items_html}</ol>"
        elif t == BLOCK_TABLE:
            if isinstance(block.content, TableData):
                td = block.content
                header_bg = f"#{settings.get('table_header_color_hex','2E75B6')}"
                hcells = "".join(f"<th style='background:{header_bg};color:white;padding:6px'>{h}</th>" for h in td.headers)
                rows_html = ""
                for i, row in enumerate(td.rows):
                    bg = "#F5F5F5" if i % 2 == 0 else "white"
                    cells = "".join(f"<td style='padding:5px;background:{bg}'>{c}</td>" for c in row)
                    rows_html += f"<tr>{cells}</tr>"
                return f"<table style='border-collapse:collapse;width:100%;direction:rtl'><thead><tr>{hcells}</tr></thead><tbody>{rows_html}</tbody></table>"
        elif t == BLOCK_CODE_BLOCK:
            pe = block.metadata and block.metadata.get("parse_error")
            label = "[JSON Parse Error] " if pe else ""
            content = str(block.content).replace("&", "&amp;").replace("<", "&lt;")
            return f"<pre style='background:#F5F5F5;padding:8px;font-size:11px;overflow:auto'>{label}{content}</pre>"
        elif t == BLOCK_QUOTE:
            inner = "".join(_block_to_html(b) for b in (block.content if isinstance(block.content, list) else []))
            return f"<blockquote style='border-right:3px solid #ccc;padding-right:12px;margin-right:0;direction:rtl'>{inner}</blockquote>"
        elif t == BLOCK_HR:
            return "<hr/>"
        return ""

    html_parts = ["<div style='font-family:Arial,Tahoma,sans-serif;max-width:800px;margin:auto'>"]

    if settings.get("include_cover") and preview_mode == _PREVIEW_ALL:
        html_parts.append(f"""
        <div style='text-align:center;padding:40px;border:1px solid #ddd;margin-bottom:20px'>
          <h1 style='color:#{settings.get('accent_color_hex','2E75B6')}'>{settings.get('title','Booklet title')}</h1>
          <h3 style='color:#555'>{settings.get('subtitle','')}</h3>
          <p>{settings.get('course_name','')}</p>
          <p>{settings.get('author_name','')}</p>
        </div>
        """)

    for i, ch in enumerate(chapters_to_show):
        num_mode = settings.get("chapter_numbering_mode", "NONE")
        prefix = ""
        n = i + 1
        fa_digits = "۰۱۲۳۴۵۶۷۸۹"
        fa_n = "".join(fa_digits[int(d)] for d in str(n))
        if num_mode == "CHAPTER_FA":   prefix = f"فصل {fa_n}: "
        elif num_mode == "CHAPTER_EN": prefix = f"Chapter {n}: "
        elif num_mode == "NUMBER_DOT": prefix = f"{n}. "

        html_parts.append(
            f"<h1 style='color:#{settings.get('heading_color_hex','1F4E79')};direction:rtl;border-bottom:2px solid #ccc;padding-bottom:6px'>"
            f"{prefix}{ch.title}</h1>"
        )
        for block in ch.blocks:
            html_parts.append(_block_to_html(block))
        html_parts.append("<hr style='margin:30px 0'/>")

    html_parts.append("</div>")
    html_out = "".join(html_parts)
    st.markdown(html_out, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 – Build
# ══════════════════════════════════════════════════════════════════════════════

def _tab_build():
    st.subheader("🔨 Build Word booklet")

    if not st.session_state.get("bk_scanned"):
        st.info("Scan a folder first.")
        return

    entries: list[FileEntry] = st.session_state.get("bk_entries", [])
    active = [e for e in entries if e.enabled and e.readable]

    if not active:
        st.warning("No enabled readable files.")
        return

    settings = st.session_state.get("bk_settings", _default_settings())
    chapters = st.session_state.get("bk_chapters", [])

    # Build summary
    total_words = sum(e.word_count for e in active)
    total_tables = sum(e.table_count for e in active)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active files", len(active))
    c2.metric("Total words", f"{total_words:,}")
    c3.metric("Tables detected", total_tables)
    if chapters:
        invalid = sum(ch.invalid_table_count for ch in chapters)
        c4.metric("Invalid tables", invalid)

    st.divider()
    col_fn, col_pn = st.columns(2)
    output_filename = col_fn.text_input(
        "Output filename",
        value=st.session_state.get("bk_output_filename", "booklet.docx"),
        key="bk_out_fname",
    )
    project_name = col_pn.text_input(
        "Project name",
        value=st.session_state.get("bk_project_name", "New booklet"),
        key="bk_proj_name",
    )
    st.session_state["bk_output_filename"] = output_filename
    st.session_state["bk_project_name"] = project_name

    overwrite_labels = {
        "NEW_VERSION": "Create new version",
        "OVERWRITE":   "Overwrite",
    }
    ow_mode = st.radio("If output file already exists",
                        list(overwrite_labels.values()), horizontal=True, key="bk_ow_mode")
    settings["overwrite_mode"] = list(overwrite_labels.keys())[list(overwrite_labels.values()).index(ow_mode)]

    force_rebuild = st.checkbox("Force rebuild (even if nothing changed)",
                                 value=False, key="bk_force_rebuild")
    save_preview = st.checkbox("Save merged_content.md preview",
                                value=True, key="bk_save_preview")

    st.divider()

    if st.button("🚀 Build Word booklet", type="primary", key="bk_build_btn"):
        if not chapters:
            # Auto-parse if not done yet
            paths = [e.source_path for e in active]
            with st.spinner("Parsing files…"):
                chapters = parse_files(
                    paths,
                    first_h1_behavior=st.session_state.get("bk_first_h1_behavior", "USE_AS_TITLE"),
                    on_error=st.session_state.get("bk_on_error", "SKIP"),
                )
            st.session_state["bk_chapters"] = chapters
            st.session_state["bk_chapters_parsed"] = True

        progress_bar = st.progress(0, text="Starting build…")
        status_text = st.empty()

        def _build_cb(i, total, name):
            pct = int((i + 1) / max(total, 1) * 100)
            progress_bar.progress(pct, text=f"Inserting chapter {i+1} of {total}: {name}")
            status_text.text(f"Processing: {name}")

        active_chapters = [ch for ch in chapters if ch.blocks or ch.title]

        with st.spinner("Building Word file…"):
            result = run_build(
                entries=active,
                chapters=active_chapters,
                settings=settings,
                output_filename=output_filename,
                project_name=project_name,
                source_folder=st.session_state.get("bk_source_folder", ""),
                sort_mode=st.session_state.get("bk_sort_mode", "NATURAL"),
                save_merged_preview=save_preview,
                force_rebuild=force_rebuild,
                progress_callback=_build_cb,
            )

        progress_bar.empty()
        status_text.empty()
        st.session_state["bk_last_result"] = result

        if result["skipped"]:
            st.info("⏭ No changes: previous output is still valid.")
        elif result["success"]:
            st.success("✅ Booklet built successfully!")
            for w in result.get("warnings", []):
                st.warning(w)
        else:
            st.error(f"❌ Error: {result.get('error_code','')}: {result.get('error_message','')}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 7 – Outputs
# ══════════════════════════════════════════════════════════════════════════════

def _tab_outputs():
    st.subheader("📥 Outputs")

    result = st.session_state.get("bk_last_result")
    if not result:
        # Try to find existing projects
        projects = list_booklet_projects()
        if not projects:
            st.info("No booklets built yet.")
            return
        st.markdown("**Previous projects**")
        for proj in projects:
            st.markdown(f"- **{proj['name']}** — {proj['slug']} (last updated: {proj['updated_at'][:10]})")
        return

    docx_path = result.get("docx_path")
    manifest_path = result.get("manifest_path")
    merged_md_path = result.get("merged_md_path")
    stats = result.get("stats", {})

    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chapters", stats.get("chapter_count", "—"))
        c2.metric("Total words", f"{stats.get('word_count', 0):,}")
        c3.metric("Valid tables", stats.get("table_count", "—"))
        c4.metric("Invalid tables", stats.get("invalid_table_count", "—"))

    st.divider()

    # DOCX download
    if docx_path and Path(docx_path).exists():
        st.success(f"📄 Word file: `{Path(docx_path).name}`")
        with open(docx_path, "rb") as f:
            docx_bytes = f.read()
        st.download_button(
            "⬇ Download DOCX",
            data=docx_bytes,
            file_name=Path(docx_path).name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key="bk_dl_docx",
        )
    else:
        st.warning("DOCX file not found.")

    # Manifest download
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "rb") as f:
            manifest_bytes = f.read()
        st.download_button(
            "⬇ Download manifest (JSON)",
            data=manifest_bytes,
            file_name="booklet_manifest.json",
            mime="application/json",
            key="bk_dl_manifest",
        )

    # Merged markdown
    if merged_md_path and Path(merged_md_path).exists():
        with open(merged_md_path, "rb") as f:
            md_bytes = f.read()
        st.download_button(
            "⬇ Download Markdown preview",
            data=md_bytes,
            file_name="merged_content.md",
            mime="text/markdown",
            key="bk_dl_md",
        )

    st.divider()

    if st.button("🔄 Rebuild", key="bk_rebuild_btn"):
        st.session_state["bk_last_result"] = None
        st.rerun()

    # Open output folder
    if docx_path:
        folder = str(Path(docx_path).parent)
        if st.button("📂 Open output folder", key="bk_open_folder_btn"):
            try:
                os.startfile(folder)
            except Exception:
                st.info(f"Path: `{folder}`")


# ══════════════════════════════════════════════════════════════════════════════
# Main page entry point
# ══════════════════════════════════════════════════════════════════════════════

def page_booklet() -> None:
    _init_state()

    st.title("📖 Word Booklet Maker")
    st.markdown("Build a professional Word booklet from AI-generated text files")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📂 Source selection",
        "📋 File order",
        "📚 Chapters",
        "🎨 Booklet design",
        "👁 Preview",
        "🔨 Build Word",
        "📥 Outputs",
    ])

    with tab1:
        _tab_source()
    with tab2:
        _tab_ordering()
    with tab3:
        _tab_chapters()
    with tab4:
        _tab_design()
    with tab5:
        _tab_preview()
    with tab6:
        _tab_build()
    with tab7:
        _tab_outputs()
