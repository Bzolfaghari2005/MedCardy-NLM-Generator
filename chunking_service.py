"""
chunking_service.py – Text splitting and hierarchical merge for large files.

Token estimation uses a conservative character-based heuristic.
Chunks are written to .intermediate/ so completed chunks survive restarts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from settings import AI_CHARS_PER_TOKEN, AI_CHUNK_MAX_TOKENS, AI_CHUNK_OVERLAP_TOKENS


# ── Token estimation ──────────────────────────────────────────────────────────

def estimate_tokens(text: str, chars_per_token: float = AI_CHARS_PER_TOKEN) -> int:
    """Conservative token estimate: characters ÷ chars_per_token, rounded up."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token) + 1)


# ── Chunk dataclass ───────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    index: int
    text: str
    token_estimate: int
    chunk_hash: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_hash:
            self.chunk_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_text(
    text: str,
    max_tokens: int = AI_CHUNK_MAX_TOKENS,
    overlap_tokens: int = AI_CHUNK_OVERLAP_TOKENS,
    chars_per_token: float = AI_CHARS_PER_TOKEN,
) -> list[TextChunk]:
    """Split text into overlapping chunks.

    Tries to break at paragraph or sentence boundaries when possible.
    """
    if not text:
        return []

    total_tokens = estimate_tokens(text, chars_per_token)
    if total_tokens <= max_tokens:
        return [TextChunk(index=0, text=text, token_estimate=total_tokens)]

    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    chunks: list[TextChunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunk_text = text[start:]
        else:
            # Try to break at paragraph boundary
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + max_chars // 2:
                end = para_break + 2
            else:
                # Try sentence boundary
                for sep in (".\n", ".\t", ". ", "؟ ", "! "):
                    pos = text.rfind(sep, start, end)
                    if pos > start + max_chars // 2:
                        end = pos + len(sep)
                        break
                # Last resort: line boundary
                else:
                    line_break = text.rfind("\n", start, end)
                    if line_break > start + max_chars // 2:
                        end = line_break + 1
            chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(TextChunk(
                index=idx,
                text=chunk_text,
                token_estimate=estimate_tokens(chunk_text, chars_per_token),
            ))
            idx += 1

        # Advance start, backing up by overlap
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = start + max(1, max_chars // 2)
        start = next_start

    return chunks


def needs_chunking(
    text: str,
    max_tokens: int = AI_CHUNK_MAX_TOKENS,
    chars_per_token: float = AI_CHARS_PER_TOKEN,
) -> bool:
    return estimate_tokens(text, chars_per_token) > max_tokens


# ── Intermediate file helpers ─────────────────────────────────────────────────

def chunk_input_path(intermediate_dir: Path, stem: str, index: int) -> Path:
    return intermediate_dir / stem / f"chunk_{index:03d}_input.txt"


def chunk_output_path(intermediate_dir: Path, stem: str, index: int) -> Path:
    return intermediate_dir / stem / f"chunk_{index:03d}_output.txt"


def merge_output_path(intermediate_dir: Path, stem: str, level: int = 1) -> Path:
    return intermediate_dir / stem / f"merge_level_{level:02d}.txt"


def save_chunk_input(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_chunk_output(path: Path) -> Optional[str]:
    """Return text if file exists and is non-empty, else None."""
    if path.exists() and path.stat().st_size > 0:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def save_chunk_output(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── Merge helpers ─────────────────────────────────────────────────────────────

def build_merge_prompt(
    chunk_results: list[str],
    system_prompt: str,
    base_user_template: str,
    filename: str,
) -> str:
    """Build the merge request user prompt combining chunk results."""
    separator = "\n\n" + "─" * 60 + "\n\n"
    combined = separator.join(
        f"[بخش {i + 1}]\n{r}" for i, r in enumerate(chunk_results)
    )
    return (
        f"نتایج بخش‌های مختلف فایل «{filename}» در زیر آمده است.\n"
        "این نتایج را در یک تحلیل یکپارچه، منسجم و کامل ادغام کن. "
        "اطلاعات تکراری را حذف کن و ساختار منطقی حفظ کن.\n\n"
        f"{combined}"
    )


def hierarchical_merge(
    chunk_results: list[str],
    max_tokens: int = AI_CHUNK_MAX_TOKENS,
    chars_per_token: float = AI_CHARS_PER_TOKEN,
) -> list[list[str]]:
    """Group chunk results into merge batches that fit within max_tokens.

    Returns list of groups; each group should be merged in one API call.
    """
    if not chunk_results:
        return []

    groups: list[list[str]] = []
    current_group: list[str] = []
    current_tokens = 0

    for result in chunk_results:
        t = estimate_tokens(result, chars_per_token)
        if current_tokens + t > max_tokens and current_group:
            groups.append(current_group)
            current_group = [result]
            current_tokens = t
        else:
            current_group.append(result)
            current_tokens += t

    if current_group:
        groups.append(current_group)

    return groups
