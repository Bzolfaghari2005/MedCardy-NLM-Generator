"""
pdf_service.py – PDF splitting and page-range management.

Supports manual range entry, auto-split by page count,
full validation with gap/overlap detection, and preview generation.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Type alias ─────────────────────────────────────────────────────────────────
PageRange = tuple[int, int]   # (start, end)  1-based inclusive


# ══════════════════════════════════════════════════════════════════════════════
# Range parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_page_ranges(text: str) -> list[PageRange]:
    """
    Parse page range text into a list of (start, end) tuples.

    Accepted formats:
        1-10
        1-10, 11-25, 26-40
        1-10
        11-25
        26-40
    """
    if not text.strip():
        raise ValueError("Page range text is empty.")

    ranges: list[PageRange] = []
    # Normalize: replace commas with newlines
    normalized = text.replace(",", "\n")
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        if "-" not in line:
            raise ValueError(f"Invalid format: '{line}'. Expected start-end.")
        parts = line.split("-", 1)
        try:
            start = int(parts[0].strip())
            end   = int(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid numbers in range: '{line}'")
        ranges.append((start, end))

    return ranges


def auto_split_ranges(total_pages: int, pages_per_chunk: int) -> list[PageRange]:
    """Generate ranges automatically given total pages and chunk size."""
    if pages_per_chunk < 1:
        raise ValueError("Pages per chunk must be at least 1.")
    if total_pages < 1:
        raise ValueError("PDF must have at least 1 page.")

    ranges: list[PageRange] = []
    start = 1
    while start <= total_pages:
        end = min(start + pages_per_chunk - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_ranges(
    ranges: list[PageRange],
    total_pages: int,
) -> dict:
    """
    Validate page ranges against total_pages.

    Returns a dict with:
        valid: bool
        errors: list[str]
        warnings: list[str]
        preview: list[dict]  — chunk preview rows
    """
    errors: list[str] = []
    warnings: list[str] = []
    preview: list[dict] = []

    if not ranges:
        errors.append("At least one range is required.")
        return {"valid": False, "errors": errors, "warnings": warnings, "preview": preview}

    seen: set[PageRange] = set()

    for i, (start, end) in enumerate(ranges, start=1):
        if start < 1:
            errors.append(f"Range {i}: start page ({start}) cannot be less than 1.")
        if end > total_pages:
            errors.append(
                f"Range {i}: end page ({end}) exceeds PDF page count ({total_pages})."
            )
        if start > end:
            errors.append(f"Range {i}: start ({start}) is greater than end ({end}).")
        if (start, end) in seen:
            errors.append(f"Range {i}: duplicate ({start}-{end}).")
        seen.add((start, end))
        preview.append({
            "Chunk": i,
            "Start": start,
            "End": end,
            "Pages": max(0, end - start + 1),
            "Filename": f"{i:03d}_pages_{start:03d}_{end:03d}.pdf",
        })

    # Overlap check
    sorted_ranges = sorted(ranges)
    for i in range(len(sorted_ranges) - 1):
        a_start, a_end = sorted_ranges[i]
        b_start, b_end = sorted_ranges[i + 1]
        if b_start <= a_end:
            errors.append(
                f"Overlap: ({a_start}-{a_end}) and ({b_start}-{b_end}) intersect."
            )

    # Gap check
    sorted_ranges2 = sorted(ranges)
    for i in range(len(sorted_ranges2) - 1):
        a_start, a_end = sorted_ranges2[i]
        b_start, b_end = sorted_ranges2[i + 1]
        if b_start > a_end + 1:
            warnings.append(
                f"Gap: pages {a_end + 1} to {b_start - 1} are not covered by any range."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "preview": preview,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PDF info
# ══════════════════════════════════════════════════════════════════════════════

def get_pdf_page_count(pdf_path: Path) -> int:
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count


def compute_pdf_hash(pdf_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Splitting
# ══════════════════════════════════════════════════════════════════════════════

def split_pdf(
    source_pdf: Path,
    ranges: list[PageRange],
    output_dir: Path,
) -> list[Path]:
    """
    Split source_pdf according to ranges. Saves chunks to output_dir.
    Returns list of paths to generated chunk PDFs.
    Page numbers are 1-based.
    """
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(source_pdf))
    total = len(doc)
    paths: list[Path] = []

    for i, (start, end) in enumerate(ranges, start=1):
        if start < 1 or end > total or start > end:
            raise ValueError(
                f"Invalid range ({start}-{end}) for PDF with {total} pages."
            )

        chunk_doc = fitz.open()
        # fitz page indices are 0-based
        chunk_doc.insert_pdf(doc, from_page=start - 1, to_page=end - 1)

        filename = f"{i:03d}_pages_{start:03d}_{end:03d}.pdf"
        out_path = output_dir / filename
        chunk_doc.save(str(out_path))
        chunk_doc.close()
        paths.append(out_path)
        logger.info("Created chunk %d: %s", i, out_path)

    doc.close()
    return paths


def chunk_hashes(chunk_paths: list[Path]) -> list[str]:
    return [compute_pdf_hash(p) for p in chunk_paths]
