"""Tests for booklet_sort_service.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from booklet_sort_service import (
    FileEntry, _natural_key, sort_entries,
    move_entry_up, move_entry_down, move_entry_to_top, move_entry_to_bottom,
    find_duplicate_hashes, find_duplicate_titles,
)


def _entry(filename: str, size: int = 100, mtime: float = 0,
           ctime: float | None = None, number: int | None = None,
           hash: str = "", title: str = "") -> FileEntry:
    mt = datetime.fromtimestamp(mtime) if mtime else datetime.now()
    ct = datetime.fromtimestamp(ctime) if ctime else None
    return FileEntry(
        source_path=Path(filename),
        relative_path=filename,
        filename=filename,
        extension=Path(filename).suffix,
        file_size=size,
        created_time=ct,
        modified_time=mt,
        created_time_source="ctime" if ct else "mtime_fallback",
        extracted_number=number,
        detected_title=title or filename,
        heading_count=0,
        table_count=0,
        word_count=10,
        encoding="utf-8",
        file_hash=hash or filename,
        readable=True,
        error_message="",
    )


# ── Natural sort ──────────────────────────────────────────────────────────────

def test_natural_sort_key_ordering():
    keys = ["part_1.txt", "part_10.txt", "part_2.txt", "part_11.txt", "part_3.txt"]
    sorted_keys = sorted(keys, key=_natural_key)
    assert sorted_keys == ["part_1.txt", "part_2.txt", "part_3.txt", "part_10.txt", "part_11.txt"]


def test_natural_sort_leading_zeros():
    keys = ["001_ami.txt", "010_heart.txt", "002_hf.txt"]
    assert sorted(keys, key=_natural_key) == ["001_ami.txt", "002_hf.txt", "010_heart.txt"]


def test_sort_entries_natural():
    entries = [_entry("part_10.txt"), _entry("part_2.txt"), _entry("part_1.txt")]
    result = sort_entries(entries, "NATURAL")
    assert [e.filename for e in result] == ["part_1.txt", "part_2.txt", "part_10.txt"]


def test_sort_entries_name_asc():
    entries = [_entry("c.txt"), _entry("a.txt"), _entry("b.txt")]
    result = sort_entries(entries, "NAME_ASC")
    assert [e.filename for e in result] == ["a.txt", "b.txt", "c.txt"]


def test_sort_entries_name_desc():
    entries = [_entry("a.txt"), _entry("c.txt"), _entry("b.txt")]
    result = sort_entries(entries, "NAME_DESC")
    assert [e.filename for e in result] == ["c.txt", "b.txt", "a.txt"]


def test_sort_entries_modified_asc():
    entries = [
        _entry("new.txt", mtime=1000),
        _entry("old.txt", mtime=100),
        _entry("mid.txt", mtime=500),
    ]
    result = sort_entries(entries, "MODIFIED_ASC")
    assert [e.filename for e in result] == ["old.txt", "mid.txt", "new.txt"]


def test_sort_entries_modified_desc():
    entries = [
        _entry("old.txt", mtime=100),
        _entry("new.txt", mtime=1000),
    ]
    result = sort_entries(entries, "MODIFIED_DESC")
    assert result[0].filename == "new.txt"


def test_sort_entries_created_asc():
    entries = [
        _entry("b.txt", ctime=200),
        _entry("a.txt", ctime=100),
    ]
    result = sort_entries(entries, "CREATED_ASC")
    assert result[0].filename == "a.txt"


def test_sort_entries_extracted_num():
    entries = [
        _entry("chapter_010.txt", number=10),
        _entry("chapter_002.txt", number=2),
        _entry("chapter_001.txt", number=1),
        _entry("no_number.txt",   number=None),
    ]
    result = sort_entries(entries, "EXTRACTED_NUM")
    assert result[0].filename == "chapter_001.txt"
    assert result[1].filename == "chapter_002.txt"
    assert result[2].filename == "chapter_010.txt"
    assert result[3].filename == "no_number.txt"


def test_sort_entries_manual():
    entries = [_entry("c.txt"), _entry("a.txt"), _entry("b.txt")]
    entries[0].sort_order = 3
    entries[1].sort_order = 1
    entries[2].sort_order = 2
    result = sort_entries(entries, "MANUAL")
    assert [e.filename for e in result] == ["a.txt", "b.txt", "c.txt"]


# ── Manual reorder ────────────────────────────────────────────────────────────

def test_move_entry_up():
    entries = [_entry("a.txt"), _entry("b.txt"), _entry("c.txt")]
    result = move_entry_up(entries, 1)
    assert [e.filename for e in result] == ["b.txt", "a.txt", "c.txt"]


def test_move_entry_up_at_top_noop():
    entries = [_entry("a.txt"), _entry("b.txt")]
    result = move_entry_up(entries, 0)
    assert result[0].filename == "a.txt"


def test_move_entry_down():
    entries = [_entry("a.txt"), _entry("b.txt"), _entry("c.txt")]
    result = move_entry_down(entries, 1)
    assert [e.filename for e in result] == ["a.txt", "c.txt", "b.txt"]


def test_move_entry_down_at_bottom_noop():
    entries = [_entry("a.txt"), _entry("b.txt")]
    result = move_entry_down(entries, 1)
    assert result[-1].filename == "b.txt"


def test_move_to_top():
    entries = [_entry("a.txt"), _entry("b.txt"), _entry("c.txt")]
    result = move_entry_to_top(entries, 2)
    assert result[0].filename == "c.txt"


def test_move_to_bottom():
    entries = [_entry("a.txt"), _entry("b.txt"), _entry("c.txt")]
    result = move_entry_to_bottom(entries, 0)
    assert result[-1].filename == "a.txt"


def test_sort_order_reassigned_after_move():
    entries = [_entry("a.txt"), _entry("b.txt"), _entry("c.txt")]
    entries = move_entry_up(entries, 2)
    for i, e in enumerate(entries):
        assert e.sort_order == i + 1


# ── Duplicate detection ───────────────────────────────────────────────────────

def test_find_duplicate_hashes():
    entries = [
        _entry("a.txt", hash="abc123"),
        _entry("b.txt", hash="abc123"),
        _entry("c.txt", hash="different"),
    ]
    dups = find_duplicate_hashes(entries)
    assert "abc123" in dups
    assert len(dups["abc123"]) == 2
    assert "different" not in dups


def test_find_duplicate_titles():
    entries = [
        _entry("a.txt", title="Heart Failure"),
        _entry("b.txt", title="Heart Failure"),
        _entry("c.txt", title="AMI"),
    ]
    dups = find_duplicate_titles(entries)
    assert "heart failure" in dups
    assert len(dups["heart failure"]) == 2


def test_no_duplicate_hashes():
    entries = [_entry("a.txt", hash="x"), _entry("b.txt", hash="y")]
    assert find_duplicate_hashes(entries) == {}


# ── Persian filenames ─────────────────────────────────────────────────────────

def test_natural_sort_persian_names():
    # Persian filenames should sort without crashing
    entries = [_entry("فصل_۱.txt"), _entry("فصل_۱۰.txt"), _entry("فصل_۲.txt")]
    result = sort_entries(entries, "NAME_ASC")
    assert len(result) == 3
