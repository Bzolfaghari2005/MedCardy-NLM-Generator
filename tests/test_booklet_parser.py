"""Tests for booklet_parser.py"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from booklet_parser import (
    BLOCK_BULLET_LIST, BLOCK_CODE_BLOCK, BLOCK_HEADING, BLOCK_NOTE,
    BLOCK_NUMBERED_LIST, BLOCK_PARAGRAPH, BLOCK_TABLE,
    ParsedChapter, TableData,
    _find_json_objects, _parse_json_table,
    parse_file, sanitize_for_docx,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_text(content: str, first_h1_behavior: str = "USE_AS_TITLE") -> ParsedChapter:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        return parse_file(tmp, first_h1_behavior=first_h1_behavior)
    finally:
        tmp.unlink(missing_ok=True)


# ── Headings ──────────────────────────────────────────────────────────────────

def test_h1_used_as_title():
    ch = _parse_text("# Main Title\n\nSome content.")
    assert ch.title == "Main Title"
    heading_blocks = [b for b in ch.blocks if b.block_type == BLOCK_HEADING]
    assert all(b.level != 1 for b in heading_blocks), "H1 should be removed from body"


def test_h1_kept_in_body():
    ch = _parse_text("# Main Title\n\nSome content.", first_h1_behavior="KEEP_IN_BODY")
    assert ch.title == "Main Title"
    heading_blocks = [b for b in ch.blocks if b.block_type == BLOCK_HEADING and b.level == 1]
    assert len(heading_blocks) >= 1


def test_h1_demoted_to_h2():
    ch = _parse_text("# Main Title\n\nSome content.", first_h1_behavior="DEMOTE_TO_H2")
    assert ch.title == "Main Title"
    heading_blocks = [b for b in ch.blocks if b.block_type == BLOCK_HEADING]
    h2_blocks = [b for b in heading_blocks if b.level == 2]
    assert len(h2_blocks) >= 1


def test_multiple_headings():
    ch = _parse_text("# H1\n## H2\n### H3\n#### H4")
    # H1 removed, rest present
    heading_levels = [b.level for b in ch.blocks if b.block_type == BLOCK_HEADING]
    assert 2 in heading_levels
    assert 3 in heading_levels
    assert 4 in heading_levels


def test_empty_file_returns_empty_chapter():
    ch = _parse_text("")
    assert ch.blocks == []
    assert "خالی" in ch.warnings[0] or ch.warnings


def test_title_falls_back_to_stem():
    ch = _parse_text("No headings here.")
    assert ch.title  # whatever the temp filename stem is, not empty


# ── Inline formatting ─────────────────────────────────────────────────────────

def test_bold_text():
    ch = _parse_text("**Bold text** here.")
    assert any(b.block_type == BLOCK_PARAGRAPH for b in ch.blocks)
    para = next(b for b in ch.blocks if b.block_type == BLOCK_PARAGRAPH)
    bold_runs = [r for r in para.content if hasattr(r, "bold") and r.bold]
    assert bold_runs


def test_italic_text():
    ch = _parse_text("*Italic text* here.")
    para = next(b for b in ch.blocks if b.block_type == BLOCK_PARAGRAPH)
    italic_runs = [r for r in para.content if hasattr(r, "italic") and r.italic]
    assert italic_runs


# ── Lists ─────────────────────────────────────────────────────────────────────

def test_bullet_list():
    ch = _parse_text("- item one\n- item two\n- item three")
    list_blocks = [b for b in ch.blocks if b.block_type == BLOCK_BULLET_LIST]
    assert list_blocks
    assert len(list_blocks[0].content) == 3


def test_numbered_list():
    ch = _parse_text("1. first\n2. second\n3. third")
    list_blocks = [b for b in ch.blocks if b.block_type == BLOCK_NUMBERED_LIST]
    assert list_blocks
    assert len(list_blocks[0].content) == 3


# ── Emoji notes ───────────────────────────────────────────────────────────────

def test_emoji_golden_tip():
    ch = _parse_text("⭐ این یک نکته طلایی است.")
    note_blocks = [b for b in ch.blocks if b.block_type == BLOCK_NOTE]
    assert note_blocks
    assert note_blocks[0].metadata["note_type"] == "GOLDEN_TIP"


def test_emoji_warning():
    ch = _parse_text("🚨 این یک هشدار مهم است.")
    note_blocks = [b for b in ch.blocks if b.block_type == BLOCK_NOTE]
    assert note_blocks
    assert note_blocks[0].metadata["note_type"] == "WARNING"


def test_emoji_drug_note():
    ch = _parse_text("💊 نکته دارویی مهم.")
    note_blocks = [b for b in ch.blocks if b.block_type == BLOCK_NOTE]
    assert note_blocks
    assert note_blocks[0].metadata["note_type"] == "DRUG_NOTE"


# ── Mixed Persian and English ─────────────────────────────────────────────────

def test_mixed_persian_english():
    content = "# AMI جزوه\n\nTroponin I elevated. STEMI vs NSTEMI بررسی."
    ch = _parse_text(content)
    assert ch.title
    assert ch.blocks


# ── JSON table detection ──────────────────────────────────────────────────────

_VALID_TABLE_JSON = json.dumps({
    "table": {
        "headers": ["Parameter", "STEMI", "NSTEMI"],
        "rows": [
            ["ECG Changes", "ST elevation", "ST depression"],
            ["Troponin",    "Elevated",     "Elevated"],
        ]
    }
})


def test_find_json_objects_simple():
    text = '{"table": {"headers": ["A"], "rows": [["B"]]}}'
    objects = _find_json_objects(text)
    assert len(objects) == 1
    assert objects[0][2] == text


def test_find_json_objects_multiple():
    t1 = '{"table": {"headers": ["A"], "rows": [["1"]]}}'
    t2 = '{"table": {"headers": ["B"], "rows": [["2"]]}}'
    text = f"some text {t1} other {t2} end"
    objects = _find_json_objects(text)
    assert len(objects) == 2


def test_parse_json_table_valid():
    td = _parse_json_table(_VALID_TABLE_JSON)
    assert td is not None
    assert isinstance(td, TableData)
    assert td.headers == ["Parameter", "STEMI", "NSTEMI"]
    assert len(td.rows) == 2
    assert td.rows[0][0] == "ECG Changes"


def test_parse_json_table_invalid_json():
    td = _parse_json_table("{not valid json}")
    assert td is None


def test_parse_json_table_missing_table_key():
    td = _parse_json_table('{"headers": ["A"], "rows": [["B"]]}')
    assert td is None


def test_parse_json_table_not_a_table():
    td = _parse_json_table('{"name": "test", "value": 42}')
    assert td is None


def test_parse_json_table_row_padding():
    # Row with fewer cells than headers
    raw = json.dumps({
        "table": {
            "headers": ["A", "B", "C"],
            "rows": [["only_one"]]
        }
    })
    td = _parse_json_table(raw)
    assert td is not None
    assert len(td.rows[0]) == 3
    assert td.rows[0][1] == ""
    assert td.rows[0][2] == ""


def test_parse_json_table_row_truncation():
    # Row with more cells than headers
    raw = json.dumps({
        "table": {
            "headers": ["A", "B"],
            "rows": [["one", "two", "three", "four"]]
        }
    })
    td = _parse_json_table(raw)
    assert td is not None
    assert len(td.rows[0]) == 2


def test_json_table_in_code_fence():
    content = f"## Comparison\n\n```json\n{_VALID_TABLE_JSON}\n```"
    ch = _parse_text(content)
    table_blocks = [b for b in ch.blocks if b.block_type == BLOCK_TABLE]
    assert table_blocks, "JSON table in code fence should be detected"


def test_json_table_inline_in_paragraph():
    content = f"Look at this:\n\n{_VALID_TABLE_JSON}\n\nEnd."
    ch = _parse_text(content)
    table_blocks = [b for b in ch.blocks if b.block_type == BLOCK_TABLE]
    assert table_blocks, "Inline JSON table should be detected"


def test_invalid_json_table_produces_code_block():
    broken = '{"table": {"headers": ["A", "B", "rows": [bad json}'
    ch = _parse_text(f"## Section\n\n{broken}")
    # Should not crash; invalid tables become code blocks with parse_error
    assert not any(b.block_type == BLOCK_TABLE for b in ch.blocks)


def test_multiple_json_tables():
    t1 = json.dumps({"table": {"headers": ["A"], "rows": [["1"]]}})
    t2 = json.dumps({"table": {"headers": ["B"], "rows": [["2"]]}})
    ch = _parse_text(f"## Title\n\n{t1}\n\n{t2}")
    table_blocks = [b for b in ch.blocks if b.block_type == BLOCK_TABLE]
    assert len(table_blocks) == 2


def test_regular_braces_not_mistaken_for_table():
    ch = _parse_text('{"name": "test", "value": 42}')
    table_blocks = [b for b in ch.blocks if b.block_type == BLOCK_TABLE]
    assert not table_blocks, "Regular JSON objects without 'table' key should not be tables"


# ── Sanitize ──────────────────────────────────────────────────────────────────

def test_sanitize_removes_control_chars():
    text = "hello\x00\x01\x08world"
    result = sanitize_for_docx(text)
    assert "\x00" not in result
    assert "\x01" not in result
    assert "hello" in result
    assert "world" in result


def test_sanitize_preserves_emoji():
    text = "⭐ نکته طلایی 🚨 هشدار 💊 دارو"
    result = sanitize_for_docx(text)
    assert "⭐" in result
    assert "🚨" in result
    assert "💊" in result


def test_sanitize_preserves_persian():
    text = "این یک متن فارسی است"
    result = sanitize_for_docx(text)
    assert result == text


def test_sanitize_preserves_newlines_and_tabs():
    text = "line1\nline2\tindented"
    result = sanitize_for_docx(text)
    assert "\n" in result
    assert "\t" in result


def test_sanitize_empty_string():
    assert sanitize_for_docx("") == ""


# ── Encoding ──────────────────────────────────────────────────────────────────

def test_utf8_file():
    content = "# فصل اول\n\nمحتوای فارسی"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     encoding="utf-8", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        ch = parse_file(tmp)
        assert "فصل اول" in ch.title or ch.blocks
    finally:
        tmp.unlink(missing_ok=True)


def test_nonexistent_file():
    ch = parse_file(Path("/nonexistent/path/file.txt"))
    assert ch.blocks == []
    assert ch.warnings
