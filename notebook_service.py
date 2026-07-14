"""
notebook_service.py – NotebookLM client abstraction.

Two implementations share the same async interface:
  - FakeNotebookClient  → artificial delays, for local testing
  - RealNotebookClient  → wraps notebooklm-py, for production

Use `get_client(profile, use_fake)` as an async context manager.
"""
from __future__ import annotations

import asyncio
import logging
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


# ─── Custom exceptions ─────────────────────────────────────────────────────────

class NotebookLMAuthError(Exception):
    """Raised when authentication/session has expired."""


class NotebookLMRateLimitError(Exception):
    """Raised when the account is rate-limited."""


class NotebookLMError(Exception):
    """General NotebookLM API error."""


def _status_text(value) -> str:
    """Normalize string and enum-like status values from notebooklm-py."""
    return str(getattr(value, "value", value) or "").lower()


# ══════════════════════════════════════════════════════════════════════════════
# FakeNotebookClient – for testing
# ══════════════════════════════════════════════════════════════════════════════

class FakeNotebookClient:
    """
    Simulates NotebookLM with realistic-feeling artificial delays.
    Designed for parallel-execution testing without real credentials.
    """

    DELAYS = {
        "create_notebook":  (0.2, 0.5),
        "upload_file":      (0.3, 1.0),
        "wait_for_source":  (0.5, 1.5),
        "generate_audio":   (0.3, 0.7),
        "wait_for_audio":   (1.0, 2.5),
        "download_audio":   (0.3, 0.8),
        "delete_notebook":  (0.1, 0.3),
    }

    def __init__(self, profile: str, fail_auth_after: int | None = None):
        self.profile = profile
        self._fail_auth_after = fail_auth_after
        self._call_count = 0

    async def _sleep(self, key: str) -> None:
        lo, hi = self.DELAYS[key]
        await asyncio.sleep(random.uniform(lo, hi))

    def _check_fail(self) -> None:
        self._call_count += 1
        if self._fail_auth_after and self._call_count > self._fail_auth_after:
            raise NotebookLMAuthError(f"Simulated auth failure for {self.profile}")

    async def create_notebook(self, name: str) -> str:
        await self._sleep("create_notebook")
        self._check_fail()
        nb_id = f"nb_{uuid.uuid4().hex[:8]}"
        logger.debug("[%s] create_notebook → %s", self.profile, nb_id)
        return nb_id

    async def upload_file(self, notebook_id: str, file_path: str) -> str:
        """Upload any supported file (PDF, TXT, MD, DOCX)."""
        await self._sleep("upload_file")
        self._check_fail()
        src_id = f"src_{uuid.uuid4().hex[:8]}"
        logger.debug("[%s] upload_file %s → %s", self.profile, Path(file_path).name, src_id)
        return src_id

    async def wait_for_source(self, notebook_id: str, source_id: str) -> None:
        await self._sleep("wait_for_source")
        logger.debug("[%s] source ready: %s", self.profile, source_id)

    async def generate_audio(
        self,
        notebook_id: str,
        source_ids: list[str],
        prompt: str,
        language: str = "fa",
    ) -> str:
        await self._sleep("generate_audio")
        self._check_fail()
        art_id = f"art_{uuid.uuid4().hex[:8]}"
        logger.debug("[%s] generate_audio → %s", self.profile, art_id)
        return art_id

    async def wait_for_audio(self, notebook_id: str, artifact_id: str) -> None:
        await self._sleep("wait_for_audio")
        logger.debug("[%s] audio ready: %s", self.profile, artifact_id)

    async def download_audio(
        self, notebook_id: str, artifact_id: str, output_path: str
    ) -> None:
        await self._sleep("download_audio")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Write a small stub M4A header so the file is non-empty
        out.write_bytes(b"\x00\x00\x00\x20ftypm4a " + b"\x00" * 100)
        logger.debug("[%s] audio saved: %s", self.profile, output_path)

    async def delete_notebook(self, notebook_id: str) -> None:
        await self._sleep("delete_notebook")
        logger.debug("[%s] deleted notebook: %s", self.profile, notebook_id)

    async def close(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# RealNotebookClient – production
# ══════════════════════════════════════════════════════════════════════════════

class RealNotebookClient:
    """
    Wraps notebooklm-py for production use.

    API surface verified against installed package:
      from notebooklm import NotebookLMClient
      async with NotebookLMClient.from_storage(profile="name") as client:
        notebook = await client.notebooks.create(title)   → .id
        source   = await client.sources.add_file(nb_id, Path(path), wait=False)  → .id
        await client.sources.wait_until_ready(nb_id, src_id, timeout=300)
        status   = await client.artifacts.generate_audio(nb_id, source_ids=[...], instructions=prompt)  → .task_id
        await client.artifacts.wait_for_completion(nb_id, task_id, timeout=1200)
        await client.artifacts.download_audio(nb_id, output_path, artifact_id=task_id)
        await client.notebooks.delete(nb_id)
    """

    SOURCE_WAIT_TIMEOUT = 300.0
    AUDIO_WAIT_TIMEOUT  = 1200.0

    def __init__(self, profile: str):
        self.profile = profile
        self._client = None
        self._cm = None

    async def __aenter__(self) -> "RealNotebookClient":
        try:
            from notebooklm import NotebookLMClient  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "notebooklm-py is not installed.\n"
                "Run: pip install \"notebooklm-py[browser]\""
            ) from exc
        self._cm = NotebookLMClient.from_storage(profile=self.profile)
        self._client = await self._cm.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def _wrap_error(self, exc: Exception) -> Exception:
        err = str(exc).lower()
        cls = type(exc).__name__.lower()
        if any(k in err for k in ("auth", "cookie", "401", "403", "login", "expired", "session")):
            return NotebookLMAuthError(str(exc))
        # Also check the exception class name — e.g. notebooklm-py raises RateLimitError
        # whose str() is "rpc_code=USER_DISPLAYABLE_ERROR" (no "rate" in the message).
        if "ratelimit" in cls or any(k in err for k in ("rate", "429", "quota", "too many")):
            return NotebookLMRateLimitError(str(exc))
        return NotebookLMError(str(exc))

    def _wrap_artifact_error(self, exc: Exception) -> Exception:
        """Classify server-side artifact disappearance/refusal as rate limiting."""
        err = str(exc).lower()
        artifact_rejections = (
            "no completed audio",
            "no completed artifact",
            "artifact not found",
            "audio not found",
            "artifact removed",
            "empty artifact",
            "generation failed",
        )
        if any(message in err for message in artifact_rejections):
            return NotebookLMRateLimitError(str(exc))
        return self._wrap_error(exc)

    async def create_notebook(self, name: str) -> str:
        try:
            nb = await self._client.notebooks.create(name)
            return nb.id
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def upload_file(self, notebook_id: str, file_path: str) -> str:
        """Upload any supported file. Does not wait for processing."""
        try:
            source = await self._client.sources.add_file(
                notebook_id, Path(file_path), wait=False
            )
            return source.id
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def wait_for_source(self, notebook_id: str, source_id: str) -> None:
        try:
            await self._client.sources.wait_until_ready(
                notebook_id, source_id, timeout=self.SOURCE_WAIT_TIMEOUT
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

    async def generate_audio(
        self,
        notebook_id: str,
        source_ids: list[str],
        prompt: str,
        language: str = "fa",
    ) -> str:
        try:
            status = await self._client.artifacts.generate_audio(
                notebook_id,
                source_ids=source_ids,
                instructions=prompt,
                language=language,
            )
        except Exception as exc:
            raise self._wrap_error(exc) from exc

        # notebooklm-py swallows USER_DISPLAYABLE_ERROR (rate-limit / daily quota)
        # and returns GenerationStatus(task_id="", status="failed") instead of raising.
        # Detect this and convert it to a proper NotebookLMRateLimitError so the
        # caller stops retrying this account immediately.
        task_id = getattr(status, "task_id", None) if status is not None else None
        state = _status_text(getattr(status, "status", ""))
        if not task_id or state in ("empty", "removed", "failed"):
            err_detail = (
                getattr(status, "error", None)
                if status is not None else None
            ) or state or "rate limit or daily quota exceeded"
            raise NotebookLMRateLimitError(
                f"Audio generation refused by server (status={state or 'empty'}): "
                f"{err_detail}"
            )
        return task_id

    async def wait_for_audio(self, notebook_id: str, artifact_id: str) -> None:
        try:
            result = await self._client.artifacts.wait_for_completion(
                notebook_id, artifact_id, timeout=self.AUDIO_WAIT_TIMEOUT
            )
        except Exception as exc:
            raise self._wrap_artifact_error(exc) from exc

        # wait_for_completion RETURNS (not raises) when the artifact disappears from
        # the listing — typically meaning a daily quota / rate-limit rejection.
        state = _status_text(getattr(result, "status", ""))
        if result is None or state in ("empty", "removed", "failed"):
            err_detail = getattr(result, "error", None) or state or "empty result"
            raise NotebookLMRateLimitError(
                f"Audio artifact {state or 'empty'} during polling: {err_detail}"
            )

    async def download_audio(
        self, notebook_id: str, artifact_id: str, output_path: str
    ) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self._client.artifacts.download_audio(
                notebook_id, str(out), artifact_id=artifact_id
            )
        except Exception as exc:
            raise self._wrap_artifact_error(exc) from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise NotebookLMRateLimitError(
                f"Audio artifact {artifact_id} produced no completed audio."
            )

    async def delete_notebook(self, notebook_id: str) -> None:
        try:
            await self._client.notebooks.delete(notebook_id)
        except Exception as exc:
            # Don't raise on delete failures — log and continue
            logger.warning("Failed to delete notebook %s: %s", notebook_id, exc)

    async def close(self) -> None:
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None
            self._client = None


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def get_client(
    profile: str,
    use_fake: bool = True,
    fail_auth_after: int | None = None,
) -> AsyncGenerator[FakeNotebookClient | RealNotebookClient, None]:
    """
    Async context manager yielding the appropriate client.

    Usage:
        async with get_client("account_01", use_fake=False) as client:
            nb_id = await client.create_notebook("My NB")
    """
    if use_fake:
        client: FakeNotebookClient | RealNotebookClient = FakeNotebookClient(
            profile=profile,
            fail_auth_after=fail_auth_after,
        )
        try:
            yield client
        finally:
            await client.close()
    else:
        client = RealNotebookClient(profile=profile)
        async with client:
            yield client
