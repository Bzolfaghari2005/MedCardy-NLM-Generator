"""
transcription_service.py – Local audio transcription via faster-whisper.

Processing is fully local; no audio is sent to external services.
Supports TXT, SRT, VTT, and JSON output.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import database as db
from models import TranscriptionStatus
from settings import (
    DB_PATH, TRANSCRIPTIONS_DIR,
    WHISPER_BEAM_SIZE, WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL,
    WHISPER_VAD_FILTER,
)

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".m4a", ".mp3", ".wav", ".aac",
    ".flac", ".ogg", ".opus", ".mp4",
}


# ══════════════════════════════════════════════════════════════════════════════
# Device detection
# ══════════════════════════════════════════════════════════════════════════════

def resolve_device_and_compute(
    device: str = "auto",
    compute_type: str = "auto",
) -> tuple[str, str]:
    """Resolve 'auto' settings to concrete values."""
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


# ══════════════════════════════════════════════════════════════════════════════
# File hash
# ══════════════════════════════════════════════════════════════════════════════

def compute_audio_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# Core transcription
# ══════════════════════════════════════════════════════════════════════════════

def transcribe_file(
    input_path: Path,
    output_dir: Path,
    model_name: str = WHISPER_MODEL,
    language: str = WHISPER_LANGUAGE,
    device: str = WHISPER_DEVICE,
    compute_type: str = WHISPER_COMPUTE_TYPE,
    beam_size: int = WHISPER_BEAM_SIZE,
    vad_filter: bool = WHISPER_VAD_FILTER,
    output_txt: bool = True,
    output_srt: bool = False,
    output_vtt: bool = False,
    output_json: bool = False,
    include_header: bool = True,
    include_timestamps: bool = False,
    stem_override: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> dict:
    """
    Transcribe audio file locally using faster-whisper.

    progress_callback is called with a float in [0.0, 1.0] as each segment
    completes.  Returns a result dict with paths and detected language.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "faster-whisper is not installed.\n"
            "Install with: pip install faster-whisper"
        )

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    device_resolved, compute_resolved = resolve_device_and_compute(device, compute_type)
    lang_param = language.strip() or None

    logger.info(
        "Transcribing %s [model=%s device=%s compute=%s]",
        input_path.name, model_name, device_resolved, compute_resolved,
    )

    model = WhisperModel(
        model_name,
        device=device_resolved,
        compute_type=compute_resolved,
    )

    segments_raw, info = model.transcribe(
        str(input_path),
        language=lang_param,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    # Consume the generator while tracking progress
    segments = []
    total_duration = max(info.duration or 1.0, 1.0)
    for seg in segments_raw:
        segments.append(seg)
        if progress_callback is not None:
            progress = min(seg.end / total_duration, 1.0)
            try:
                progress_callback(progress)
            except Exception:
                pass

    if progress_callback is not None:
        try:
            progress_callback(1.0)
        except Exception:
            pass

    del model  # free memory

    detected_language = info.language
    duration_s = info.duration
    duration_str = _fmt_duration(duration_s)

    stem = stem_override or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "detected_language": detected_language,
        "duration": duration_str,
        "model": model_name,
        "segment_count": len(segments),
        "txt_path": None,
        "srt_path": None,
        "vtt_path": None,
        "json_path": None,
    }

    full_text = " ".join(seg.text.strip() for seg in segments)

    if output_txt:
        txt_path = output_dir / f"{stem}.txt"
        _write_txt(txt_path, input_path, detected_language, duration_str, model_name,
                   segments, full_text, include_header, include_timestamps)
        result["txt_path"] = str(txt_path)

    if output_srt:
        srt_path = output_dir / f"{stem}.srt"
        _write_srt(srt_path, segments)
        result["srt_path"] = str(srt_path)

    if output_vtt:
        vtt_path = output_dir / f"{stem}.vtt"
        _write_vtt(vtt_path, segments)
        result["vtt_path"] = str(vtt_path)

    if output_json:
        json_path = output_dir / f"{stem}.json"
        _write_json(json_path, segments, detected_language, duration_str, model_name)
        result["json_path"] = str(json_path)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════════

def _write_txt(
    path: Path,
    source_path: Path,
    language: str,
    duration: str,
    model: str,
    segments,
    full_text: str,
    include_header: bool,
    include_timestamps: bool,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if include_header:
            f.write(f"File: {source_path.name}\n")
            f.write(f"Detected language: {language}\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Model: whisper-{model}\n\n")
            f.write("Transcript:\n\n")

        if include_timestamps:
            for seg in segments:
                ts = f"[{_ts(seg.start)} - {_ts(seg.end)}]"
                f.write(f"{ts} {seg.text.strip()}\n")
        else:
            f.write(full_text)
            f.write("\n")


def _write_srt(path: Path, segments) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{_srt_ts(seg.start)} --> {_srt_ts(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")


def _write_vtt(path: Path, segments) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{_vtt_ts(seg.start)} --> {_vtt_ts(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")


def _write_json(
    path: Path, segments, language: str, duration: str, model: str
) -> None:
    data = {
        "language": language,
        "duration": duration,
        "model": model,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Timestamp formatters
# ══════════════════════════════════════════════════════════════════════════════

def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _srt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    return _srt_ts(seconds).replace(",", ".")


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# DB-tracked transcription
# ══════════════════════════════════════════════════════════════════════════════

def run_transcription_job(
    tr_id: int,
    output_srt: bool = False,
    output_vtt: bool = False,
    output_json: bool = False,
    include_header: bool = True,
    include_timestamps: bool = False,
    path: Path = DB_PATH,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> bool:
    """Execute a transcription tracked in the DB. Returns True on success."""
    transcriptions = db.list_transcriptions(path=path)
    tr = next((t for t in transcriptions if t.id == tr_id), None)
    if not tr:
        logger.error("Transcription %d not found.", tr_id)
        return False

    # Idempotency
    if tr.output_txt_path and Path(tr.output_txt_path).exists():
        if tr.status == TranscriptionStatus.COMPLETED:
            return True

    db.update_transcription(tr_id, {
        "status": TranscriptionStatus.RUNNING.value,
        "started_at": datetime.utcnow().isoformat(),
    }, path)

    input_path = Path(tr.input_path)
    # Determine output dir
    if tr.project_id:
        from settings import PROJECTS_DIR
        proj = db.get_project(tr.project_id, path)
        if proj:
            out_dir = Path(proj.output_dir) / "transcripts"
        else:
            out_dir = TRANSCRIPTIONS_DIR
    else:
        out_dir = TRANSCRIPTIONS_DIR

    try:
        result = transcribe_file(
            input_path=input_path,
            output_dir=out_dir,
            model_name=tr.model_name,
            language=tr.language,
            device=tr.device,
            compute_type=tr.compute_type,
            output_txt=True,
            output_srt=output_srt,
            output_vtt=output_vtt,
            output_json=output_json,
            include_header=include_header,
            include_timestamps=include_timestamps,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        logger.error("Transcription %d failed: %s", tr_id, exc)
        db.update_transcription(tr_id, {
            "status": TranscriptionStatus.FAILED.value,
            "error_message": str(exc)[:500],
            "completed_at": datetime.utcnow().isoformat(),
        }, path)
        return False

    now = datetime.utcnow().isoformat()
    db.update_transcription(tr_id, {
        "status": TranscriptionStatus.COMPLETED.value,
        "output_txt_path": result.get("txt_path"),
        "output_srt_path": result.get("srt_path"),
        "output_vtt_path": result.get("vtt_path"),
        "output_json_path": result.get("json_path"),
        "progress": 1.0,
        "completed_at": now,
    }, path)
    return True


def is_whisper_available() -> tuple[bool, str]:
    """Check if faster-whisper is importable."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]
        return True, "faster-whisper is available."
    except ImportError:
        return False, "faster-whisper is not installed.  Run: pip install faster-whisper"
