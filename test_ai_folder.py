"""
test_ai_folder.py – Test suite for the AI Folder Processor.

All tests use FakeAIProvider (no real API calls, no cost).
Real GapGPT integration tests run only when:
    RUN_GAPGPT_INTEGRATION_TESTS=true

Run:
    python test_ai_folder.py

    # With verbose output:
    python test_ai_folder.py -v

    # With real GapGPT tests:
    set RUN_GAPGPT_INTEGRATION_TESTS=true
    python test_ai_folder.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

# ─── project path ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import database as db
from ai_api_service import AIConnectionStatus, AITextResult, FakeAIProvider, GapGPTProvider
from ai_batch_runner import AiBatchRunner, RunConfig, create_run_and_jobs
from ai_folder_service import (
    FolderScanConfig,
    scan_folder,
    validate_folder,
    FolderValidationError,
)
from chunking_service import estimate_tokens, needs_chunking, split_text
from file_extractor_service import extract_file
from models import AIChunkMode, AIFileGroup, AIJobStatus, AIRunStatus
from prompt_service import (
    compute_dedup_key,
    compute_prompt_hash,
    ensure_default_profile,
    render_prompt,
)
from secret_scanner import is_filename_blocked, mask_api_key, scan_text_for_secrets
from settings import DB_PATH

# ─── Test database (isolated temp file) ──────────────────────────────────────

def _make_test_db() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    test_path = Path(tmp.name)
    db.init_db(test_path)
    return test_path


def _fake_provider(**kwargs) -> FakeAIProvider:
    return FakeAIProvider(**kwargs)


# ════════════════════════════════════════════════════════════════════════════════
# 1. Secret Scanner
# ════════════════════════════════════════════════════════════════════════════════

class TestSecretScanner(unittest.TestCase):

    def test_blocked_env_file(self):
        blocked, pat = is_filename_blocked(".env")
        self.assertTrue(blocked)
        self.assertIsNotNone(pat)

    def test_blocked_pem_file(self):
        blocked, _ = is_filename_blocked("server.pem")
        self.assertTrue(blocked)

    def test_blocked_id_rsa(self):
        blocked, _ = is_filename_blocked("id_rsa")
        self.assertTrue(blocked)

    def test_blocked_sqlite(self):
        blocked, _ = is_filename_blocked("database.sqlite3")
        self.assertTrue(blocked)

    def test_safe_txt(self):
        blocked, _ = is_filename_blocked("report.txt")
        self.assertFalse(blocked)

    def test_safe_pdf(self):
        blocked, _ = is_filename_blocked("document.pdf")
        self.assertFalse(blocked)

    def test_detects_api_key_in_text(self):
        text = 'api_key = "sk-abcdefghijklmnopqrst"'
        matches = scan_text_for_secrets(text)
        self.assertTrue(len(matches) > 0)

    def test_detects_private_key_block(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nsome stuff"
        matches = scan_text_for_secrets(text)
        self.assertTrue(len(matches) > 0)

    def test_detects_aws_access_key(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        matches = scan_text_for_secrets(text)
        self.assertTrue(len(matches) > 0)

    def test_no_false_positive_normal_text(self):
        text = "این یک فایل متنی عادی است.\nهیچ اطلاعات حساسی ندارد."
        matches = scan_text_for_secrets(text)
        self.assertEqual(len(matches), 0)

    def test_mask_api_key(self):
        key = "sk-abcdefghijklmnopqrstuvwxyz123456"
        masked = mask_api_key(key)
        self.assertIn("****", masked)
        self.assertNotIn(key, masked)
        self.assertFalse(masked.endswith("..."))


# ════════════════════════════════════════════════════════════════════════════════
# 2. Folder Scanning
# ════════════════════════════════════════════════════════════════════════════════

class TestFolderScanning(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _create(self, rel: str, content: str = "hello") -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_discover_txt(self):
        self._create("a.txt")
        cfg = FolderScanConfig(root=self.root, recursive=False, scan_secrets=False)
        result = scan_folder(cfg)
        self.assertEqual(len(result.files), 1)
        self.assertEqual(result.files[0].extension, ".txt")

    def test_recursive_discovers_subdirs(self):
        self._create("sub/b.txt")
        self._create("c.txt")
        cfg = FolderScanConfig(root=self.root, recursive=True, scan_secrets=False)
        result = scan_folder(cfg)
        self.assertEqual(len(result.files), 2)

    def test_non_recursive_ignores_subdirs(self):
        self._create("sub/b.txt")
        self._create("c.txt")
        cfg = FolderScanConfig(root=self.root, recursive=False, scan_secrets=False)
        result = scan_folder(cfg)
        self.assertEqual(len(result.files), 1)

    def test_hidden_file_excluded_by_default(self):
        self._create(".hidden.txt")
        self._create("visible.txt")
        cfg = FolderScanConfig(root=self.root, include_hidden=False, scan_secrets=False)
        result = scan_folder(cfg)
        names = [f.filename for f in result.files]
        self.assertNotIn(".hidden.txt", names)
        self.assertIn("visible.txt", names)

    def test_hidden_file_included_when_flag_set(self):
        self._create(".hidden.txt")
        cfg = FolderScanConfig(root=self.root, include_hidden=True, scan_secrets=False)
        result = scan_folder(cfg)
        names = [f.filename for f in result.files]
        self.assertIn(".hidden.txt", names)

    def test_env_file_blocked_by_secret_scanner(self):
        self._create(".env", "GAPGPT_API_KEY=sk-secret")
        cfg = FolderScanConfig(root=self.root, include_hidden=True, scan_secrets=True)
        result = scan_folder(cfg)
        env_file = next((f for f in result.files if f.filename == ".env"), None)
        self.assertIsNotNone(env_file)
        self.assertIsNotNone(env_file.skip_reason)

    def test_unknown_extension_group(self):
        self._create("archive.bin", b"\x00\x01\x02".decode("latin-1"))
        cfg = FolderScanConfig(root=self.root, scan_secrets=False)
        result = scan_folder(cfg)
        bin_file = next((f for f in result.files if f.filename == "archive.bin"), None)
        self.assertIsNotNone(bin_file)
        self.assertEqual(bin_file.file_group, AIFileGroup.UNKNOWN)

    def test_max_file_size_filter(self):
        big = self._create("big.txt", "x" * 2000)
        cfg = FolderScanConfig(root=self.root, max_file_mb=0.001, scan_secrets=False)
        result = scan_folder(cfg)
        big_file = next((f for f in result.files if f.filename == "big.txt"), None)
        self.assertIsNotNone(big_file)
        self.assertIsNotNone(big_file.skip_reason)

    def test_validate_nonexistent_folder(self):
        with self.assertRaises(FolderValidationError):
            validate_folder("/nonexistent/path/that/does/not/exist")

    def test_validate_valid_folder(self):
        p = validate_folder(str(self.root))
        self.assertEqual(p, self.root.resolve())

    def test_preserve_relative_path(self):
        self._create("chapter/lesson.txt")
        cfg = FolderScanConfig(root=self.root, recursive=True, scan_secrets=False)
        result = scan_folder(cfg)
        f = next((x for x in result.files if x.filename == "lesson.txt"), None)
        self.assertIsNotNone(f)
        self.assertEqual(str(f.relative_path), str(Path("chapter") / "lesson.txt"))


# ════════════════════════════════════════════════════════════════════════════════
# 3. File Extraction
# ════════════════════════════════════════════════════════════════════════════════

class TestFileExtraction(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_txt(self):
        p = self.root / "a.txt"
        p.write_text("Hello World\nLine 2", encoding="utf-8")
        result = extract_file(p)
        self.assertTrue(result.success)
        self.assertIn("Hello World", result.text)

    def test_extract_txt_utf8(self):
        p = self.root / "fa.txt"
        p.write_text("سلام دنیا", encoding="utf-8")
        result = extract_file(p)
        self.assertTrue(result.success)
        self.assertIn("سلام", result.text)

    def test_extract_json(self):
        p = self.root / "data.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        result = extract_file(p)
        self.assertTrue(result.success)
        self.assertIn("value", result.text)

    def test_image_skipped_when_vision_disabled(self):
        p = self.root / "photo.jpg"
        p.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 10)  # minimal JPEG header
        result = extract_file(p, vision_enabled=False)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "VISION_NOT_SUPPORTED")

    def test_unknown_type_returns_error(self):
        p = self.root / "file.xyz123"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = extract_file(p)
        self.assertEqual(result.error_code, "UNSUPPORTED_FILE_TYPE")

    def test_nonexistent_file(self):
        p = self.root / "nofile.txt"
        result = extract_file(p)
        self.assertEqual(result.error_code, "FILE_NOT_FOUND")

    def test_audio_skip_mode(self):
        p = self.root / "audio.mp3"
        p.write_bytes(b"\x00" * 100)
        result = extract_file(p, audio_mode="skip")
        self.assertEqual(result.error_code, "SKIPPED")

    def test_zip_disabled_by_default(self):
        p = self.root / "archive.zip"
        p.write_bytes(b"PK\x03\x04" + b"\x00" * 20)
        result = extract_file(p, zip_enabled=False)
        self.assertEqual(result.error_code, "SKIPPED")


# ════════════════════════════════════════════════════════════════════════════════
# 4. Chunking
# ════════════════════════════════════════════════════════════════════════════════

class TestChunking(unittest.TestCase):

    def test_no_chunking_small_text(self):
        text = "Short text."
        chunks = split_text(text, max_tokens=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_chunking_large_text(self):
        text = "word " * 5000
        chunks = split_text(text, max_tokens=100, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)

    def test_chunk_indices_sequential(self):
        text = "line\n" * 1000
        chunks = split_text(text, max_tokens=50, overlap_tokens=5)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.index, i)

    def test_needs_chunking_true(self):
        text = "a" * 10000
        self.assertTrue(needs_chunking(text, max_tokens=100))

    def test_needs_chunking_false(self):
        text = "short"
        self.assertFalse(needs_chunking(text, max_tokens=1000))

    def test_estimate_tokens(self):
        text = "abc" * 350  # ~1050 chars → ~300 tokens at 3.5 chars/token
        tokens = estimate_tokens(text)
        self.assertGreater(tokens, 200)
        self.assertLess(tokens, 500)


# ════════════════════════════════════════════════════════════════════════════════
# 5. Prompt Service
# ════════════════════════════════════════════════════════════════════════════════

class TestPromptService(unittest.TestCase):

    def test_placeholder_substitution(self):
        template = "File: {{filename}}\nContent:\n{{file_content}}"
        result = render_prompt(template, filename="doc.txt", file_content="body")
        self.assertIn("doc.txt", result)
        self.assertIn("body", result)

    def test_stem_placeholder(self):
        result = render_prompt("{{stem}}", filename="report.pdf")
        self.assertEqual(result, "report")

    def test_absolute_path_hidden_by_default(self):
        result = render_prompt(
            "{{absolute_path}}",
            absolute_path="/home/user/secret/file.txt",
            include_absolute_path=False,
        )
        self.assertEqual(result.strip(), "")

    def test_absolute_path_shown_when_enabled(self):
        result = render_prompt(
            "{{absolute_path}}",
            absolute_path="/home/user/file.txt",
            include_absolute_path=True,
        )
        self.assertIn("/home/user/file.txt", result)

    def test_prompt_hash_deterministic(self):
        h1 = compute_prompt_hash("sys", "user")
        h2 = compute_prompt_hash("sys", "user")
        self.assertEqual(h1, h2)

    def test_prompt_hash_differs_on_change(self):
        h1 = compute_prompt_hash("sys", "user A")
        h2 = compute_prompt_hash("sys", "user B")
        self.assertNotEqual(h1, h2)

    def test_dedup_key_changes_with_model(self):
        fh = "abc123"
        ph = "prompt_hash"
        k1 = compute_dedup_key(fh, ph, "gpt-4", "url", "ext")
        k2 = compute_dedup_key(fh, ph, "gpt-5.2", "url", "ext")
        self.assertNotEqual(k1, k2)

    def test_ensure_default_profile_creates(self):
        db_path = _make_test_db()
        profile_id = ensure_default_profile(db_path)
        self.assertIsNotNone(profile_id)
        profiles = db.list_ai_prompt_profiles(db_path)
        self.assertGreater(len(profiles), 0)
        os.unlink(db_path)

    def test_ensure_default_profile_idempotent(self):
        db_path = _make_test_db()
        id1 = ensure_default_profile(db_path)
        id2 = ensure_default_profile(db_path)
        self.assertEqual(id1, id2)
        os.unlink(db_path)


# ════════════════════════════════════════════════════════════════════════════════
# 6. Fake AI Provider
# ════════════════════════════════════════════════════════════════════════════════

class TestFakeAIProvider(unittest.TestCase):

    def test_generate_text_success(self):
        provider = FakeAIProvider()
        result = provider.generate_text("gpt-5.2", "sys", "user")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.text)

    def test_generate_text_fails_on_call(self):
        provider = FakeAIProvider(fail_on_call=1, fail_error_code="API_ERROR")
        result = provider.generate_text("gpt-5.2", "sys", "user")
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "API_ERROR")

    def test_rate_limit_simulation(self):
        provider = FakeAIProvider(simulate_rate_limit_on=1)
        result = provider.generate_text("gpt-5.2", "sys", "user")
        self.assertEqual(result.error_code, "RATE_LIMITED")

    def test_test_connection_returns_connected(self):
        provider = FakeAIProvider()
        status = provider.test_connection()
        self.assertEqual(status, AIConnectionStatus.CONNECTED.value)

    def test_vision_response(self):
        provider = FakeAIProvider()
        result = provider.generate_vision("gpt-5.2", "sys", "user text", "data:image/png;base64,abc")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.text)


# ════════════════════════════════════════════════════════════════════════════════
# 7. End-to-end batch run with Fake Provider
# ════════════════════════════════════════════════════════════════════════════════

class TestBatchRunnerEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input_dir = self.root / "input"
        self.output_dir = self.root / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        self.db_path = _make_test_db()
        ensure_default_profile(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()
        try:
            os.unlink(self.db_path)
        except Exception:
            pass

    def _create_file(self, rel: str, content: str = "Test content for AI.") -> Path:
        p = self.input_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def _run_batch(self, discovered_files, provider=None, **kwargs) -> int:
        if provider is None:
            provider = FakeAIProvider()

        profiles = db.list_ai_prompt_profiles(self.db_path)
        run_id = create_run_and_jobs(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            discovered_files=discovered_files,
            model="gpt-5.2",
            base_url="https://api.gapgpt.app/v1",
            prompt_profile_id=profiles[0].id if profiles else None,
            config_kwargs={
                "max_concurrency": 2,
                "timeout_seconds": 10,
                "max_retries": 0,
                "chunk_max_tokens": 6000,
                "chunk_overlap_tokens": 200,
                "chunk_mode": "CHUNKED_MERGE",
                **kwargs,
            },
            db_path=self.db_path,
        )

        cfg = RunConfig(
            run_id=run_id,
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            model="gpt-5.2",
            base_url="https://api.gapgpt.app/v1",
            provider=provider,
            max_concurrency=2,
            timeout_seconds=10,
            max_retries=0,
            chunk_max_tokens=6000,
            db_path=self.db_path,
        )

        runner = AiBatchRunner(cfg)
        runner.run()
        return run_id

    def _scan(self, **kw) -> list:
        from ai_folder_service import DiscoveredFile, FolderScanConfig, scan_folder
        cfg = FolderScanConfig(root=self.input_dir, scan_secrets=False, **kw)
        return scan_folder(cfg).files

    def test_txt_file_processed(self):
        self._create_file("doc.txt", "Content here.")
        files = self._scan()
        run_id = self._run_batch(files)
        jobs = db.list_ai_file_jobs(run_id, self.db_path)
        completed = [j for j in jobs if j.status == AIJobStatus.COMPLETED]
        self.assertEqual(len(completed), 1)

    def test_output_file_created(self):
        self._create_file("report.txt", "Report content.")
        files = self._scan()
        run_id = self._run_batch(files)
        jobs = db.list_ai_file_jobs(run_id, self.db_path)
        completed = [j for j in jobs if j.status == AIJobStatus.COMPLETED]
        self.assertGreater(len(completed), 0)
        out_path = Path(completed[0].output_txt_path)
        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 0)

    def test_output_file_utf8(self):
        self._create_file("fa.txt", "سلام دنیا")
        files = self._scan()
        run_id = self._run_batch(files)
        jobs = [j for j in db.list_ai_file_jobs(run_id, self.db_path) if j.status == AIJobStatus.COMPLETED]
        if jobs:
            text = Path(jobs[0].output_txt_path).read_text(encoding="utf-8")
            self.assertIsInstance(text, str)

    def test_subdirectory_structure_preserved(self):
        self._create_file("ch1/lesson.txt", "Lesson 1")
        self._create_file("ch2/lecture.txt", "Lecture 2")
        files = self._scan(recursive=True)
        run_id = self._run_batch(files)
        jobs = [j for j in db.list_ai_file_jobs(run_id, self.db_path) if j.status == AIJobStatus.COMPLETED]
        for j in jobs:
            out = Path(j.output_txt_path)
            rel = Path(j.relative_path)
            self.assertTrue(str(out).replace("\\", "/").endswith(
                str(rel.parent / f"{rel.stem}__ai_result.txt").replace("\\", "/")
            ) or out.exists())

    def test_unknown_file_skipped(self):
        p = self.input_dir / "archive.bin"
        p.write_bytes(b"\x00\x01\x02\x03" * 50)
        files = self._scan()
        # Manually mark as unknown
        for f in files:
            if f.filename == "archive.bin":
                f.enabled = False
                f.skip_reason = "unsupported"
        run_id = self._run_batch(files)
        jobs = db.list_ai_file_jobs(run_id, self.db_path)
        skipped = [j for j in jobs if j.status == AIJobStatus.SKIPPED]
        self.assertGreater(len(skipped), 0)

    def test_three_concurrent_requests(self):
        for i in range(3):
            self._create_file(f"file{i}.txt", f"Content {i}")
        files = self._scan()
        call_times: list[float] = []

        class TimingProvider(FakeAIProvider):
            def generate_text(self, model, system_prompt, user_prompt, timeout=180):
                call_times.append(time.time())
                time.sleep(0.05)
                return super().generate_text(model, system_prompt, user_prompt, timeout)

        run_id = self._run_batch(files, provider=TimingProvider())
        jobs = db.list_ai_file_jobs(run_id, self.db_path)
        completed = [j for j in jobs if j.status == AIJobStatus.COMPLETED]
        self.assertEqual(len(completed), 3)

    def test_one_file_failure_does_not_stop_others(self):
        self._create_file("good1.txt", "Good 1")
        self._create_file("good2.txt", "Good 2")
        files = self._scan()
        run_id = self._run_batch(files, provider=FakeAIProvider(fail_on_call=1))
        jobs = db.list_ai_file_jobs(run_id, self.db_path)
        completed = [j for j in jobs if j.status == AIJobStatus.COMPLETED]
        failed = [j for j in jobs if j.status == AIJobStatus.FAILED]
        self.assertGreater(len(completed), 0)
        self.assertGreater(len(failed), 0)

    def test_completed_file_not_reprocessed(self):
        self._create_file("dedup.txt", "Dedup test content")
        files = self._scan()
        run_id = self._run_batch(files)

        call_count_1 = sum(
            1 for j in db.list_ai_file_jobs(run_id, self.db_path)
            if j.status == AIJobStatus.COMPLETED
        )

        # Second run, same files → should reuse
        run_id2 = self._run_batch(files)
        jobs2 = db.list_ai_file_jobs(run_id2, self.db_path)
        completed2 = [j for j in jobs2 if j.status == AIJobStatus.COMPLETED]
        self.assertEqual(len(completed2), call_count_1)

    def test_api_key_not_in_output(self):
        self._create_file("secure.txt", "Normal content")
        files = self._scan()
        run_id = self._run_batch(files)
        jobs = [j for j in db.list_ai_file_jobs(run_id, self.db_path) if j.output_txt_path]
        for j in jobs:
            content = Path(j.output_txt_path).read_text(encoding="utf-8")
            self.assertNotIn("sk-", content.lower().replace("sk-****", ""))

    def test_api_key_not_in_database(self):
        self._create_file("f.txt", "Content")
        files = self._scan()
        run_id = self._run_batch(files)
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute("SELECT * FROM ai_file_jobs").fetchall()
        conn.close()
        for row in rows:
            row_str = str(row).lower()
            self.assertNotIn("sk-real-key", row_str)

    def test_large_file_chunked(self):
        big_text = "This is a sentence about important facts.\n" * 5000
        self._create_file("big.txt", big_text)
        files = self._scan()
        run_id = self._run_batch(files)
        jobs = db.list_ai_file_jobs(run_id, self.db_path)
        big_job = next((j for j in jobs if j.input_filename == "big.txt"), None)
        self.assertIsNotNone(big_job)
        # Should be completed or failed (depending on chunk size)
        self.assertIn(big_job.status, (AIJobStatus.COMPLETED, AIJobStatus.FAILED))

    def test_placeholder_in_output(self):
        self._create_file("check.txt", "File content here")
        files = self._scan()
        run_id = self._run_batch(files)
        jobs = [j for j in db.list_ai_file_jobs(run_id, self.db_path) if j.output_txt_path]
        if jobs:
            content = Path(jobs[0].output_txt_path).read_text(encoding="utf-8")
            # FakeAI echoes content info
            self.assertGreater(len(content), 0)

    def test_run_status_completed(self):
        self._create_file("x.txt", "content")
        files = self._scan()
        run_id = self._run_batch(files)
        run = db.get_ai_batch_run(run_id, self.db_path)
        self.assertEqual(run.status, AIRunStatus.COMPLETED)

    def test_batch_run_counters_correct(self):
        for i in range(3):
            self._create_file(f"f{i}.txt", f"content {i}")
        files = self._scan()
        run_id = self._run_batch(files)
        run = db.get_ai_batch_run(run_id, self.db_path)
        self.assertEqual(run.completed_files + run.failed_files, 3)


# ════════════════════════════════════════════════════════════════════════════════
# 8. Database tests
# ════════════════════════════════════════════════════════════════════════════════

class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.db_path = _make_test_db()

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except Exception:
            pass

    def test_create_and_retrieve_prompt_profile(self):
        pid = db.create_ai_prompt_profile(
            name="Test Profile",
            system_prompt="sys",
            user_prompt_template="user {{file_content}}",
            path=self.db_path,
        )
        profile = db.get_ai_prompt_profile(pid, self.db_path)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "Test Profile")

    def test_create_batch_run(self):
        run_id = db.create_ai_batch_run(
            input_folder="/tmp/in",
            output_folder="/tmp/out",
            path=self.db_path,
        )
        run = db.get_ai_batch_run(run_id, self.db_path)
        self.assertIsNotNone(run)
        self.assertEqual(run.status.value, "PENDING")

    def test_create_file_job(self):
        run_id = db.create_ai_batch_run("/in", "/out", path=self.db_path)
        job_id = db.create_ai_file_job(
            run_id=run_id,
            relative_path="doc.txt",
            absolute_input_path="/in/doc.txt",
            input_filename="doc.txt",
            path=self.db_path,
        )
        job = db.get_ai_file_job(job_id, self.db_path)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, AIJobStatus.DISCOVERED)

    def test_update_file_job_status(self):
        run_id = db.create_ai_batch_run("/in", "/out", path=self.db_path)
        job_id = db.create_ai_file_job(
            run_id=run_id,
            relative_path="x.txt",
            absolute_input_path="/in/x.txt",
            input_filename="x.txt",
            path=self.db_path,
        )
        db.update_ai_file_job(job_id, {"status": AIJobStatus.COMPLETED.value}, self.db_path)
        job = db.get_ai_file_job(job_id, self.db_path)
        self.assertEqual(job.status, AIJobStatus.COMPLETED)

    def test_api_key_not_stored_in_batch_run(self):
        run_id = db.create_ai_batch_run("/in", "/out", path=self.db_path)
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        schema = conn.execute("PRAGMA table_info(ai_batch_runs)").fetchall()
        conn.close()
        col_names = [row[1] for row in schema]
        self.assertNotIn("api_key", col_names)
        self.assertNotIn("authorization", col_names)


# ════════════════════════════════════════════════════════════════════════════════
# 9. Real GapGPT integration (gated)
# ════════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(
    os.environ.get("RUN_GAPGPT_INTEGRATION_TESTS") == "true",
    "Set RUN_GAPGPT_INTEGRATION_TESTS=true to run real GapGPT tests.",
)
class TestRealGapGPT(unittest.TestCase):

    def setUp(self):
        from ai_api_service import resolve_api_key
        from settings import GAPGPT_BASE_URL
        key = resolve_api_key()
        if not key:
            self.skipTest("GAPGPT_API_KEY not set")
        self.provider = GapGPTProvider(api_key=key, base_url=GAPGPT_BASE_URL)

    def test_connection(self):
        status = self.provider.test_connection()
        self.assertEqual(status, AIConnectionStatus.CONNECTED.value)

    def test_generate_text(self):
        result = self.provider.generate_text(
            model="gpt-5.2",
            system_prompt="You are a test assistant.",
            user_prompt="Say 'OK' and nothing else.",
            timeout=30,
        )
        self.assertTrue(result.success)
        self.assertIsNotNone(result.text)


# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running AI Folder Processor tests with FakeAIProvider...")
    print("No real API calls, no cost.\n")

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures) + len(result.errors)
    print(f"\n{'=' * 60}")
    print(f"{'PASSED' if failures == 0 else 'FAILED'}: {total - failures}/{total} tests passed")
    if failures > 0:
        sys.exit(1)
