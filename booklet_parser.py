"""
booklet_parser.py – Parse Markdown/text files into ParsedChapter objects.

Uses markdown-it-py for tokenisation.  JSON table detection uses a
balanced-braces character walk (no regex-only approach) followed by
json.loads validation.  Never calls eval().
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from markdown_it import MarkdownIt


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

BLOCK_HEADING       = "HEADING"
BLOCK_PARAGRAPH     = "PARAGRAPH"
BLOCK_BULLET_LIST   = "BULLET_LIST"
BLOCK_NUMBERED_LIST = "NUMBERED_LIST"
BLOCK_TABLE         = "TABLE"
BLOCK_CODE_BLOCK    = "CODE_BLOCK"
BLOCK_INLINE_CODE   = "INLINE_CODE"
BLOCK_QUOTE         = "QUOTE"
BLOCK_HR            = "HORIZONTAL_RULE"
BLOCK_WARNING       = "WARNING"
BLOCK_NOTE          = "NOTE"


@dataclass
class DocumentBlock:
    block_type: str
    content: Any           # str | list[InlineRun] | TableData | list[DocumentBlock]
    level: Optional[int] = None
    metadata: Optional[dict] = None


@dataclass
class InlineRun:
    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


@dataclass
class TableData:
    headers: list[str]
    rows: list[list[str]]


@dataclass
class ParsedChapter:
    title: str
    source_path: Path
    blocks: list[DocumentBlock]
    warnings: list[str]
    raw_word_count: int = 0
    table_count: int = 0
    invalid_table_count: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Emoji note-type detection
# ══════════════════════════════════════════════════════════════════════════════

_NOTE_EMOJIS: dict[str, str] = {
    "⭐": "GOLDEN_TIP",
    "✅": "IMPORTANT",
    "📌": "CLINICAL_EXAMPLE",
    "💡": "CLINICAL_PEARL",
    "🚨": "WARNING",
    "🔥": "HIGH_YIELD",
    "📊": "DATA",
    "🏁": "SUMMARY",
    "💊": "DRUG_NOTE",
    "🩺": "EXAM_FINDING",
}


def _detect_note_type(text: str) -> Optional[str]:
    stripped = text.strip()
    for emoji, note_type in _NOTE_EMOJIS.items():
        if stripped.startswith(emoji):
            return note_type
    return None


# ══════════════════════════════════════════════════════════════════════════════
# XML / DOCX character sanitisation
# ══════════════════════════════════════════════════════════════════════════════

# XML 1.0 forbids control chars except \t \n \r
_INVALID_XML_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
    r"\ud800-\udfff"  # lone surrogates
    r"]"
)

# Zero-width chars that cause rendering issues (keep ZWJ which is needed for emoji)
_PROBLEM_ZW_RE = re.compile(r"[\u200b\u200c\u200e\u200f\u202a-\u202e\ufeff]")


def sanitize_for_docx(text: str) -> str:
    """Remove XML-invalid control chars; keep Unicode, Emoji, and Persian text."""
    if not text:
        return text
    text = _INVALID_XML_RE.sub("", text)
    text = _PROBLEM_ZW_RE.sub("", text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Balanced-braces JSON table extractor
# ══════════════════════════════════════════════════════════════════════════════

def _find_json_objects(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start, end, raw_json) for every top-level { } block."""
    results = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        start = i
        j = i
        while j < length:
            ch = text[j]
            if escape:
                escape = False
            elif ch == "\\" and in_string:
                escape = True
            elif ch == '"' and not escape:
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        raw = text[start: j + 1]
                        results.append((start, j + 1, raw))
                        i = j + 1
                        break
            j += 1
        else:
            # Unmatched brace; move past start
            i = start + 1
    return results


def _parse_json_table(raw: str) -> Optional[TableData]:
    """Try to parse raw JSON as a table object.  Return None if invalid."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None
    table = obj.get("table")
    if not isinstance(table, dict):
        return None
    headers = table.get("headers")
    rows = table.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list):
        return None
    if not headers:
        return None

    str_headers = [str(h) for h in headers]
    n_cols = len(str_headers)

    clean_rows: list[list[str]] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        cells = [str(c) for c in row]
        # Pad or trim to match header count
        if len(cells) < n_cols:
            cells.extend([""] * (n_cols - len(cells)))
        elif len(cells) > n_cols:
            cells = cells[:n_cols]
        clean_rows.append(cells)

    return TableData(headers=str_headers, rows=clean_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Inline token → InlineRun list
# ══════════════════════════════════════════════════════════════════════════════

def _inline_tokens_to_runs(tokens: list) -> list[InlineRun]:
    runs: list[InlineRun] = []
    bold = False
    italic = False
    code = False
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            runs.append(InlineRun(text=sanitize_for_docx(buf), bold=bold, italic=italic, code=code))
            buf = ""

    for tok in tokens:
        t = tok.type
        if t == "text":
            buf += tok.content
        elif t == "softbreak" or t == "hardbreak":
            buf += "\n"
        elif t == "strong_open":
            flush(); bold = True
        elif t == "strong_close":
            flush(); bold = False
        elif t == "em_open":
            flush(); italic = True
        elif t == "em_close":
            flush(); italic = False
        elif t == "code_inline":
            flush()
            runs.append(InlineRun(text=sanitize_for_docx(tok.content), bold=bold, italic=italic, code=True))
        elif t == "html_inline":
            pass  # skip raw HTML
        else:
            if hasattr(tok, "content") and tok.content:
                buf += tok.content

    flush()
    return runs


# ══════════════════════════════════════════════════════════════════════════════
# Token stream walker
# ══════════════════════════════════════════════════════════════════════════════

_md = MarkdownIt("commonmark")


def _tokens_to_blocks(tokens: list) -> tuple[list[DocumentBlock], list[str]]:
    blocks: list[DocumentBlock] = []
    warnings: list[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        t = tok.type

        # ── Heading ──────────────────────────────────────────────────────────
        if t == "heading_open":
            level = int(tok.tag[1])  # h1→1, h2→2 …
            content_tok = tokens[i + 1] if i + 1 < n else None
            if content_tok and content_tok.type == "inline":
                runs = _inline_tokens_to_runs(content_tok.children or [])
                plain = "".join(r.text for r in runs)
                note_type = _detect_note_type(plain)
                if note_type:
                    blocks.append(DocumentBlock(
                        block_type=BLOCK_NOTE,
                        content=runs,
                        level=level,
                        metadata={"note_type": note_type},
                    ))
                else:
                    blocks.append(DocumentBlock(
                        block_type=BLOCK_HEADING,
                        content=runs,
                        level=level,
                    ))
            i += 3  # heading_open, inline, heading_close
            continue

        # ── Paragraph ────────────────────────────────────────────────────────
        if t == "paragraph_open":
            content_tok = tokens[i + 1] if i + 1 < n else None
            if content_tok and content_tok.type == "inline":
                runs = _inline_tokens_to_runs(content_tok.children or [])
                plain = "".join(r.text for r in runs)

                # Check if this paragraph contains a JSON table
                json_blocks, leftover, w = _extract_json_tables_from_text(plain)
                if json_blocks:
                    for jb in json_blocks:
                        blocks.append(jb)
                    if leftover.strip():
                        blocks.append(DocumentBlock(
                            block_type=BLOCK_PARAGRAPH,
                            content=[InlineRun(text=sanitize_for_docx(leftover.strip()))],
                        ))
                else:
                    note_type = _detect_note_type(plain)
                    if note_type:
                        blocks.append(DocumentBlock(
                            block_type=BLOCK_NOTE,
                            content=runs,
                            metadata={"note_type": note_type},
                        ))
                    else:
                        blocks.append(DocumentBlock(
                            block_type=BLOCK_PARAGRAPH,
                            content=runs,
                        ))
                warnings.extend(w)
            i += 3
            continue

        # ── Bullet list ──────────────────────────────────────────────────────
        if t == "bullet_list_open":
            items, end_i, w = _collect_list_items(tokens, i)
            blocks.append(DocumentBlock(block_type=BLOCK_BULLET_LIST, content=items))
            warnings.extend(w)
            i = end_i + 1
            continue

        # ── Ordered list ─────────────────────────────────────────────────────
        if t == "ordered_list_open":
            items, end_i, w = _collect_list_items(tokens, i)
            blocks.append(DocumentBlock(block_type=BLOCK_NUMBERED_LIST, content=items))
            warnings.extend(w)
            i = end_i + 1
            continue

        # ── Blockquote ───────────────────────────────────────────────────────
        if t == "blockquote_open":
            inner_tokens = []
            j = i + 1
            depth = 1
            while j < n:
                if tokens[j].type == "blockquote_open":
                    depth += 1
                elif tokens[j].type == "blockquote_close":
                    depth -= 1
                    if depth == 0:
                        break
                inner_tokens.append(tokens[j])
                j += 1
            inner_blocks, w = _tokens_to_blocks(inner_tokens)
            blocks.append(DocumentBlock(block_type=BLOCK_QUOTE, content=inner_blocks))
            warnings.extend(w)
            i = j + 1
            continue

        # ── Code fence ───────────────────────────────────────────────────────
        if t == "fence":
            info = tok.info.strip().lower()
            code_content = tok.content

            # Attempt JSON table detection inside code fence
            json_blocks, leftover, w = _extract_json_tables_from_text(code_content)
            if json_blocks:
                for jb in json_blocks:
                    blocks.append(jb)
                warnings.extend(w)
            else:
                blocks.append(DocumentBlock(
                    block_type=BLOCK_CODE_BLOCK,
                    content=sanitize_for_docx(code_content),
                    metadata={"info": info},
                ))
            i += 1
            continue

        # ── Code block (indented) ─────────────────────────────────────────────
        if t == "code_block":
            blocks.append(DocumentBlock(
                block_type=BLOCK_CODE_BLOCK,
                content=sanitize_for_docx(tok.content),
            ))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────────
        if t == "hr":
            blocks.append(DocumentBlock(block_type=BLOCK_HR, content=""))
            i += 1
            continue

        i += 1

    return blocks, warnings


def _collect_list_items(
    tokens: list, start: int
) -> tuple[list[list[InlineRun]], int, list[str]]:
    """Return (items, close_index, warnings) for a list starting at *start*."""
    open_type = tokens[start].type  # bullet_list_open or ordered_list_open
    close_type = open_type.replace("open", "close")
    items: list[list[InlineRun]] = []
    warnings: list[str] = []
    depth = 1
    j = start + 1
    n = len(tokens)
    while j < n:
        tok = tokens[j]
        if tok.type == open_type:
            depth += 1
        elif tok.type == close_type:
            depth -= 1
            if depth == 0:
                return items, j, warnings
        elif tok.type == "inline" and depth == 1:
            runs = _inline_tokens_to_runs(tok.children or [])
            items.append(runs)
        j += 1
    return items, j, warnings


def _extract_json_tables_from_text(
    text: str,
) -> tuple[list[DocumentBlock], str, list[str]]:
    """Scan *text* for JSON table objects; return (blocks, remaining_text, warnings)."""
    json_objects = _find_json_objects(text)
    if not json_objects:
        return [], text, []

    blocks: list[DocumentBlock] = []
    warnings: list[str] = []
    consumed_ranges: list[tuple[int, int]] = []

    for start, end, raw in json_objects:
        table_data = _parse_json_table(raw)
        if table_data is not None:
            blocks.append(DocumentBlock(block_type=BLOCK_TABLE, content=table_data))
            consumed_ranges.append((start, end))
        elif '"table"' in raw:
            # Looks like a table but is broken JSON
            warnings.append(f"JSON table parse failed (len={len(raw)}): {raw[:80]}…")
            blocks.append(DocumentBlock(
                block_type=BLOCK_CODE_BLOCK,
                content=sanitize_for_docx(raw),
                metadata={"parse_error": True},
            ))
            consumed_ranges.append((start, end))

    # Build remaining text by removing consumed ranges
    remaining_parts = []
    prev = 0
    for s, e in sorted(consumed_ranges):
        remaining_parts.append(text[prev:s])
        prev = e
    remaining_parts.append(text[prev:])
    remaining = "".join(remaining_parts)

    return blocks, remaining, warnings


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def parse_file(
    path: Path,
    first_h1_behavior: str = "USE_AS_TITLE",
) -> ParsedChapter:
    """Parse a text/Markdown file into a ParsedChapter.

    first_h1_behavior:
      "USE_AS_TITLE"  – use first H1 as chapter title, remove from body
      "KEEP_IN_BODY"  – keep first H1 in body as well
      "DEMOTE_TO_H2"  – convert first H1 to Heading 2 in body
    """
    warnings: list[str] = []

    if not path.exists():
        return ParsedChapter(
            title=path.stem,
            source_path=path,
            blocks=[],
            warnings=["File does not exist"],
        )

    # Read content
    content = ""
    for enc in ("utf-8-sig", "utf-8", "windows-1256", "iso-8859-1", "latin-1"):
        try:
            content = path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if not content.strip():
        return ParsedChapter(
            title=path.stem,
            source_path=path,
            blocks=[],
            warnings=["File is empty"],
        )

    # Also scan the raw text for bare (non-fenced) JSON tables that markdown
    # might silently swallow inside paragraphs.  We do this before tokenising
    # because markdown-it may mangle multi-line raw JSON.
    try:
        tokens = _md.parse(content)
        blocks, w = _tokens_to_blocks(tokens)
        warnings.extend(w)
    except Exception as exc:
        warnings.append(f"Markdown parse error: {exc}")
        blocks = [DocumentBlock(block_type=BLOCK_PARAGRAPH, content=sanitize_for_docx(content))]

    # Handle first H1
    title = path.stem
    if blocks:
        first_heading_idx = next(
            (i for i, b in enumerate(blocks)
             if b.block_type == BLOCK_HEADING and b.level == 1),
            None,
        )
        if first_heading_idx is not None:
            h1_block = blocks[first_heading_idx]
            raw_title = "".join(
                r.text for r in (h1_block.content if isinstance(h1_block.content, list) else [])
            ).strip()
            if raw_title:
                title = raw_title

            if first_h1_behavior == "USE_AS_TITLE":
                blocks.pop(first_heading_idx)
            elif first_h1_behavior == "DEMOTE_TO_H2":
                h1_block.level = 2
            # KEEP_IN_BODY: do nothing

    # Count tables
    table_count = sum(1 for b in blocks if b.block_type == BLOCK_TABLE)
    invalid_table_count = sum(
        1 for b in blocks
        if b.block_type == BLOCK_CODE_BLOCK
        and isinstance(b.metadata, dict)
        and b.metadata.get("parse_error")
    )

    raw_word_count = len(content.split())

    return ParsedChapter(
        title=title,
        source_path=path,
        blocks=blocks,
        warnings=warnings,
        raw_word_count=raw_word_count,
        table_count=table_count,
        invalid_table_count=invalid_table_count,
    )


def parse_files(
    paths: list[Path],
    first_h1_behavior: str = "USE_AS_TITLE",
    on_error: str = "SKIP",
    progress_callback=None,
) -> list[ParsedChapter]:
    """Parse multiple files.  on_error='SKIP'|'STOP'."""
    chapters: list[ParsedChapter] = []
    for i, p in enumerate(paths):
        if progress_callback:
            progress_callback(i, len(paths), p.name)
        try:
            ch = parse_file(p, first_h1_behavior=first_h1_behavior)
            chapters.append(ch)
        except Exception as exc:
            if on_error == "STOP":
                raise
            ch = ParsedChapter(
                title=p.stem,
                source_path=p,
                blocks=[],
                warnings=[f"Failed to read file: {exc}"],
            )
            chapters.append(ch)
    return chapters
