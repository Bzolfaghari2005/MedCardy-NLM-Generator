"""
ai_api_service.py – AI provider abstraction and GapGPT implementation.

Design:
  AITextProvider   – abstract interface
  FakeAIProvider   – deterministic fake for testing
  GapGPTProvider   – real OpenAI-compatible API (GapGPT / any OpenAI-compat endpoint)

API key is NEVER stored here. It is passed at construction time from the
caller which resolves it from env / .env / session state.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from models import AIConnectionStatus


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AITextResult:
    text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error_code is None and self.text is not None

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())


# ── Abstract interface ────────────────────────────────────────────────────────

class AITextProvider(ABC):

    @abstractmethod
    def generate_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 180,
    ) -> AITextResult:
        """Send a chat completion request and return the result."""
        ...

    @abstractmethod
    def generate_vision(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        image_data_url: str,
        timeout: int = 180,
    ) -> AITextResult:
        """Send a vision request (text + image) and return the result."""
        ...

    @abstractmethod
    def test_connection(self, model: str = "gpt-5.2") -> str:
        """Return an AIConnectionStatus value string."""
        ...


# ── Fake provider (for testing) ───────────────────────────────────────────────

class FakeAIProvider(AITextProvider):
    """Deterministic fake that echoes back metadata for testing.

    Does NOT make real API calls. Never needs an API key.
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_on_call: int = 0,    # fail on the N-th call (0 = never)
        fail_error_code: str = "API_ERROR",
        simulate_rate_limit_on: int = 0,
        simulate_context_overflow: bool = False,
    ):
        self._delay = delay
        self._fail_on = fail_on_call
        self._fail_code = fail_error_code
        self._rate_limit_on = simulate_rate_limit_on
        self._context_overflow = simulate_context_overflow
        self._call_count = 0

    def generate_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 180,
    ) -> AITextResult:
        self._call_count += 1

        if self._delay:
            time.sleep(self._delay)

        if self._rate_limit_on and self._call_count == self._rate_limit_on:
            return AITextResult(
                error_code="RATE_LIMITED",
                error_message="Fake rate limit triggered.",
            )

        if self._fail_on and self._call_count == self._fail_on:
            return AITextResult(
                error_code=self._fail_code,
                error_message=f"Fake failure on call {self._call_count}.",
            )

        if self._context_overflow:
            return AITextResult(
                error_code="CONTEXT_LENGTH_EXCEEDED",
                error_message="Fake context length exceeded.",
            )

        # Echo: generate a fake response based on input length
        content_preview = user_prompt[:200].replace("\n", " ")
        text = (
            f"[FakeAI] مدل: {model}\n"
            f"طول System Prompt: {len(system_prompt)} کاراکتر\n"
            f"طول User Prompt: {len(user_prompt)} کاراکتر\n"
            f"پیش‌نمایش ورودی: {content_preview}\n\n"
            "این یک پاسخ آزمایشی از Fake AI Provider است."
        )
        in_tok = max(1, len(system_prompt + user_prompt) // 4)
        out_tok = max(1, len(text) // 4)
        return AITextResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        )

    def generate_vision(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        image_data_url: str,
        timeout: int = 180,
    ) -> AITextResult:
        self._call_count += 1
        if self._delay:
            time.sleep(self._delay)
        return AITextResult(
            text=f"[FakeAI Vision] تصویر دریافت شد. مدل: {model}. اندازه Data URL: {len(image_data_url)} کاراکتر.",
            input_tokens=50,
            output_tokens=20,
            total_tokens=70,
        )

    def test_connection(self, model: str = "gpt-5.2") -> str:
        self._call_count += 1
        if self._delay:
            time.sleep(self._delay)
        return AIConnectionStatus.CONNECTED.value


# ── Real GapGPT / OpenAI-compatible provider ─────────────────────────────────

class GapGPTProvider(AITextProvider):
    """OpenAI-compatible provider for GapGPT (and any OpenAI-compatible endpoint).

    api_key is accepted at construction and NEVER logged or written to disk.
    """

    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "کتابخانه openai نصب نشده است. دستور: pip install openai"
            ) from exc
        return OpenAI(base_url=self._base_url + "/", api_key=self._api_key)

    def generate_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 180,
    ) -> AITextResult:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=timeout,
            )

            choice = response.choices[0] if response.choices else None
            if not choice or not choice.message or not choice.message.content:
                return AITextResult(
                    error_code="API_BAD_RESPONSE",
                    error_message="پاسخ API فاقد متن بود.",
                )

            usage = getattr(response, "usage", None)
            return AITextResult(
                text=choice.message.content,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        except ImportError as exc:
            return AITextResult(error_code="API_ERROR", error_message=str(exc))
        except Exception as exc:
            return _classify_exception(exc)

    def generate_vision(
        self,
        model: str,
        system_prompt: str,
        user_text: str,
        image_data_url: str,
        timeout: int = 180,
    ) -> AITextResult:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
                timeout=timeout,
            )

            choice = response.choices[0] if response.choices else None
            if not choice or not choice.message or not choice.message.content:
                return AITextResult(
                    error_code="API_BAD_RESPONSE",
                    error_message="پاسخ API فاقد متن بود.",
                )

            usage = getattr(response, "usage", None)
            return AITextResult(
                text=choice.message.content,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            )

        except ImportError as exc:
            return AITextResult(error_code="API_ERROR", error_message=str(exc))
        except Exception as exc:
            return _classify_exception(exc)

    def test_connection(self, model: str = "gpt-5.2") -> str:
        """Send a minimal request and return AIConnectionStatus value."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=5,
                timeout=30,
            )
            if response.choices:
                return AIConnectionStatus.CONNECTED.value
            return AIConnectionStatus.API_ERROR.value

        except ImportError:
            return AIConnectionStatus.API_ERROR.value
        except Exception as exc:
            status = _classify_exception(exc)
            return status.error_code or AIConnectionStatus.API_ERROR.value


# ── API key resolution ────────────────────────────────────────────────────────

def resolve_api_key(session_key: Optional[str] = None) -> Optional[str]:
    """Resolve API key from: env var → .env file → session argument."""
    import os

    # 1. Environment variable (set directly or via .env loaded at startup)
    key = os.environ.get("GAPGPT_API_KEY", "").strip()
    if key:
        return key

    # 2. Load .env if not already loaded
    try:
        from dotenv import load_dotenv
        from settings import BASE_DIR
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=False)
            key = os.environ.get("GAPGPT_API_KEY", "").strip()
            if key:
                return key
    except ImportError:
        pass

    # 3. Session state value
    if session_key and session_key.strip():
        return session_key.strip()

    return None


def build_provider(
    api_key: str,
    base_url: str,
    *,
    fake: bool = False,
) -> AITextProvider:
    """Build and return the appropriate provider."""
    if fake:
        return FakeAIProvider()
    return GapGPTProvider(api_key=api_key, base_url=base_url)


# ── Exception classifier ──────────────────────────────────────────────────────

def _classify_exception(exc: Exception) -> AITextResult:
    """Map common API exceptions to typed error codes."""
    msg = str(exc)
    lower = msg.lower()

    # Try to read HTTP status code from openai exceptions
    status_code: Optional[int] = None
    if hasattr(exc, "status_code"):
        status_code = exc.status_code
    elif hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        status_code = exc.response.status_code

    if status_code == 401 or "unauthorized" in lower or "invalid api key" in lower or "authentication" in lower:
        return AITextResult(error_code=AIConnectionStatus.INVALID_API_KEY.value, error_message=msg)
    if status_code == 402 or "credit" in lower or "quota" in lower or "billing" in lower:
        return AITextResult(error_code=AIConnectionStatus.INSUFFICIENT_CREDIT.value, error_message=msg)
    if status_code == 429 or "rate limit" in lower or "too many requests" in lower:
        return AITextResult(error_code=AIConnectionStatus.RATE_LIMITED.value, error_message=msg)
    if status_code == 400 and "context" in lower and "length" in lower:
        return AITextResult(error_code="CONTEXT_LENGTH_EXCEEDED", error_message=msg)
    if any(x in lower for x in ("timeout", "timed out", "connection", "network", "unreachable")):
        return AITextResult(error_code=AIConnectionStatus.NETWORK_ERROR.value, error_message=msg)

    return AITextResult(error_code=AIConnectionStatus.API_ERROR.value, error_message=msg)
