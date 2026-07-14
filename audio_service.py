"""
audio_service.py – M4A/audio to MP3 conversion via FFmpeg.

Uses subprocess with list args (no shell=True).
Converts to a .part temp file first, then renames atomically.
"""
from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import database as db
from models import ConversionStatus
from settings import DB_PATH, DEFAULT_MP3_BITRATE, KEEP_ORIGINAL_AUDIO

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_FORMATS = {
    ".m4a", ".wav", ".mp3", ".aac",
    ".flac", ".ogg", ".opus", ".mp4",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core conversion
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_mp3(
    input_path: Path,
    output_path: Path,
    bitrate: str = DEFAULT_MP3_BITRATE,
    overwrite: bool = False,
    keep_original: bool = KEEP_ORIGINAL_AUDIO,
    ffmpeg_exe: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Convert any supported audio file to MP3.

    Returns (success, message).
    Uses a .part temp file to avoid corrupting the output on failure.
    """
    if ffmpeg_exe is None:
        try:
            from settings import get_ffmpeg_executable
            ffmpeg_exe = get_ffmpeg_executable()
        except FileNotFoundError as exc:
            return False, str(exc)

    if not input_path.exists():
        return False, f"Input file not found: {input_path}"

    if output_path.exists() and not overwrite:
        return True, f"MP3 file already exists: {output_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real media extension last so FFmpeg can infer the muxer.
    # A name ending in ".part" makes FFmpeg fail with "Unable to choose an
    # output format" on a standard installation.
    temp_path = output_path.with_name(
        f"{output_path.stem}.part{output_path.suffix}"
    )

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-map_metadata", "0",
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(temp_path),
    ]

    logger.info("Converting: %s → %s", input_path.name, output_path.name)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False, "Conversion timed out (over 10 minutes)."
    except FileNotFoundError:
        return False, f"Failed to run FFmpeg: {ffmpeg_exe}"
    except Exception as exc:
        return False, f"FFmpeg error: {exc}"

    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False, f"FFmpeg exited with error (code {result.returncode}):\n{result.stderr[-500:]}"

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        return False, "Output file is empty or incomplete."

    # Validate duration similarity (optional, best-effort)
    _warn_duration_mismatch(input_path, temp_path, ffmpeg_exe)

    # Atomic rename
    temp_path.replace(output_path)
    logger.info("Conversion OK → %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)

    if not keep_original and input_path != output_path:
        try:
            input_path.unlink()
        except Exception as exc:
            logger.warning("Could not delete original: %s", exc)

    return True, f"Conversion successful: {output_path.name}"


def _warn_duration_mismatch(src: Path, dst: Path, ffmpeg_exe: str) -> None:
    """Best-effort duration comparison via ffprobe."""
    try:
        import shutil
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            ff_dir = Path(ffmpeg_exe).parent
            probe_candidate = ff_dir / "ffprobe.exe"
            if probe_candidate.exists():
                ffprobe = str(probe_candidate)
        if not ffprobe:
            return

        def _dur(path: Path) -> Optional[float]:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                 str(path)],
                capture_output=True, text=True, timeout=10,
            )
            try:
                return float(r.stdout.strip())
            except Exception:
                return None

        src_dur = _dur(src)
        dst_dur = _dur(dst)
        if src_dur and dst_dur:
            diff = abs(src_dur - dst_dur)
            if diff > 2.0:
                logger.warning(
                    "Duration mismatch: src=%.1fs dst=%.1fs (diff=%.1fs)",
                    src_dur, dst_dur, diff,
                )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# DB-tracked conversion
# ══════════════════════════════════════════════════════════════════════════════

def run_conversion_job(
    conv_id: int,
    path: Path = DB_PATH,
) -> bool:
    """Execute a conversion tracked in the DB. Returns True on success."""
    conversions = db.list_audio_conversions(path=path)
    conv = next((c for c in conversions if c.id == conv_id), None)
    if not conv:
        logger.error("Conversion %d not found.", conv_id)
        return False

    db.update_audio_conversion(conv_id, {
        "status": ConversionStatus.RUNNING.value,
        "started_at": datetime.utcnow().isoformat(),
    }, path)

    input_path = Path(conv.input_path)
    output_path = input_path.parent.parent / "audio_mp3" / (input_path.stem + ".mp3")

    # Check if already done
    if output_path.exists() and output_path.stat().st_size > 0:
        db.update_audio_conversion(conv_id, {
            "status": ConversionStatus.COMPLETED.value,
            "output_path": str(output_path),
            "completed_at": datetime.utcnow().isoformat(),
        }, path)
        return True

    success, message = convert_to_mp3(
        input_path,
        output_path,
        bitrate=conv.bitrate,
        overwrite=True,
    )

    now = datetime.utcnow().isoformat()
    if success:
        db.update_audio_conversion(conv_id, {
            "status": ConversionStatus.COMPLETED.value,
            "output_path": str(output_path),
            "completed_at": now,
        }, path)
    else:
        db.update_audio_conversion(conv_id, {
            "status": ConversionStatus.FAILED.value,
            "error_message": message,
            "completed_at": now,
        }, path)

    return success


# ══════════════════════════════════════════════════════════════════════════════
# Batch utility
# ══════════════════════════════════════════════════════════════════════════════

def batch_convert(
    input_files: list[Path],
    output_dir: Path,
    bitrate: str = DEFAULT_MP3_BITRATE,
    overwrite: bool = False,
    keep_original: bool = True,
    ffmpeg_exe: Optional[str] = None,
) -> list[dict]:
    """Convert a list of files and return results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for f in input_files:
        out = output_dir / (f.stem + ".mp3")
        ok, msg = convert_to_mp3(f, out, bitrate, overwrite, keep_original, ffmpeg_exe)
        results.append({"input": str(f), "output": str(out), "success": ok, "message": msg})
    return results


def is_ffmpeg_available() -> tuple[bool, str]:
    """Check if FFmpeg is accessible. Returns (available, path_or_error)."""
    try:
        from settings import get_ffmpeg_executable
        exe = get_ffmpeg_executable()
        result = subprocess.run(
            [exe, "-version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            first_line = result.stdout.splitlines()[0] if result.stdout else ""
            return True, f"{exe} ({first_line})"
        return False, f"FFmpeg error: {result.stderr[:100]}"
    except FileNotFoundError:
        return False, "FFmpeg is not installed."
    except Exception as exc:
        return False, str(exc)
