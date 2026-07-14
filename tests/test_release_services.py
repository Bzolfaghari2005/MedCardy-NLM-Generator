from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import audio_service
import file_service
from log_utils import redact_log_text


def test_mp3_temp_file_keeps_mp3_extension(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    output = tmp_path / "output.mp3"
    source.write_bytes(b"input")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"encoded mp3")
        return SimpleNamespace(returncode=0, stderr="")

    with (
        patch("audio_service.subprocess.run", side_effect=fake_run),
        patch("audio_service._warn_duration_mismatch"),
    ):
        ok, _ = audio_service.convert_to_mp3(
            source,
            output,
            ffmpeg_exe="ffmpeg",
        )

    assert ok
    assert output.read_bytes() == b"encoded mp3"
    assert commands[0][-1].endswith(".part.mp3")


def test_delete_project_files_retries_windows_lock(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "locked.tmp").write_bytes(b"data")
    project = SimpleNamespace(output_dir=str(project_dir))
    real_rmtree = file_service.shutil.rmtree
    attempts = 0

    def flaky_rmtree(path, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("locked")
        return real_rmtree(path, **kwargs)

    with (
        patch("file_service.db.get_project", return_value=project),
        patch("file_service.shutil.rmtree", side_effect=flaky_rmtree),
        patch("file_service.time.sleep"),
    ):
        file_service.delete_project_files(1, tmp_path / "db.sqlite3")

    assert attempts == 2
    assert not project_dir.exists()


def test_delete_temp_files_handles_part_media_names(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    for name in ("old.part", "audio.part.mp3", "download.tmp", "keep.mp3"):
        (project_dir / name).write_bytes(b"x")
    project = SimpleNamespace(output_dir=str(project_dir))

    with patch("file_service.db.get_project", return_value=project):
        deleted = file_service.delete_temp_files(1, tmp_path / "db.sqlite3")

    assert deleted == 3
    assert (project_dir / "keep.mp3").exists()


def test_log_redaction_removes_personal_and_secret_values() -> None:
    value = (
        "GET https://example.test/?authuser=user%40example.com&token=secret "
        "https://drum.usercontent.google.com/download/signed-value "
        "Authorization: Bearer abc.def-123"
    )
    redacted = redact_log_text(value)

    assert "user%40example.com" not in redacted
    assert "token=secret" not in redacted
    assert "abc.def-123" not in redacted
    assert "signed-value" not in redacted
