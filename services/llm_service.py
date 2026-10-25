"""OpenRouter chat-completion client restricted to free models.

Guarantees:
* Only models listed in LLM_MODELS / LLM_FALLBACK_MODELS can be called.
* Every model ID must carry OpenRouter's ":free" suffix.
* When the public model catalogue is reachable, the model's prompt and completion
  prices must both be 0, otherwise the call is refused. If the catalogue cannot be
  reached, the model is reported as "unverified" (never claimed to be free).
* Requests also cap provider price at 0 (`provider.max_price`), so OpenRouter cannot
  route to a paid endpoint.
* Fallback happens only to the explicitly configured LLM_FALLBACK_MODELS.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Optional

import httpx

from config import Settings, get_settings
from schemas import TRUNCATION_REASONS, LLMResponse, LLMUsage

log = logging.getLogger(__name__)


_KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9\-_]{8,}")


def redact(text: str) -> str:
    """Remove the configured API key and anything shaped like an OpenRouter key."""
    key = get_settings().openrouter_api_key
    if key:
        text = text.replace(key, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", text)


class LLMError(RuntimeError):
    """Base class; `retryable` tells the caller whether another model may help.

    Messages are redacted so a key echoed by a provider never reaches logs, API
    responses or the UI.
    """

    retryable = False

    def __init__(self, *args):
        super().__init__(*(redact(str(a)) for a in args))

    def __str__(self) -> str:
        return redact(super().__str__())


class LLMConfigError(LLMError):
    pass


class ModelNotAllowedError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class LLMProviderError(LLMError):
    retryable = True


class LLMModelUnavailableError(LLMError):
    retryable = True


class LLMInvalidResponseError(LLMError):
    retryable = True


VERIFIED_FREE = "verified_free"
NOT_FREE = "not_free"
NOT_FOUND = "not_found"
UNVERIFIED = "unverified"


class FreeModelRegistry:
    """Caches OpenRouter's public /models catalogue to verify zero pricing."""

    def __init__(self, base_url: str, http: httpx.Client, ttl_s: int = 3600):
        self.base_url = base_url.rstrip("/")
        self.http = http
        self.ttl_s = ttl_s
        self._catalog: Optional[dict[str, dict]] = None
        self._fetched_at = 0.0
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        with self._lock:
            if self._catalog is not None and time.time() - self._fetched_at < self.ttl_s:
                return
            try:
                r = self.http.get(f"{self.base_url}/models", timeout=20)
                r.raise_for_status()
                self._catalog = {m["id"]: m for m in r.json().get("data", [])}
                self._fetched_at = time.time()
                self._error = None
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                log.warning("Could not fetch OpenRouter model catalogue: %s", self._error)

    def status(self, model_id: str) -> tuple[str, str]:
        self._refresh()
        if self._catalog is None:
            return UNVERIFIED, f"catalogue unavailable ({self._error})"
        m = self._catalog.get(model_id)
        if m is None:
            return NOT_FOUND, "model not listed in OpenRouter catalogue"
        p = m.get("pricing") or {}
        prices = [p.get("prompt"), p.get("completion"), p.get("request") or "0"]
        try:
            if all(float(x) == 0.0 for x in prices if x is not None) and p.get("prompt") is not None:
                return VERIFIED_FREE, "prompt/completion price = 0"
        except (TypeError, ValueError):
            return UNVERIFIED, f"unparseable pricing {p}"
        return NOT_FREE, f"pricing {p}"


class OpenRouterLLM:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        http: Optional[httpx.Client] = None,
        registry: Optional[FreeModelRegistry] = None,
        sleep=time.sleep,
    ):
        self.s = settings or get_settings()
        self.http = http or httpx.Client(timeout=self.s.llm_timeout_s)
        self.registry = registry or FreeModelRegistry(self.s.openrouter_base_url, self.http)
        self._sleep = sleep

    # ------------------------------------------------------------------ models
    @property
    def configured_models(self) -> list[str]:
        seen: list[str] = []
        for m in [*self.s.llm_models, *self.s.llm_fallback_models]:
            if m not in seen:
                seen.append(m)
        return seen

    def model_statuses(self) -> list[dict[str, Any]]:
        out = []
        for m in self.configured_models:
            if not m.endswith(":free"):
                st, why = NOT_FREE, "model id lacks ':free' suffix"
            elif self.s.llm_verify_free:
                st, why = self.registry.status(m)
            else:
                st, why = UNVERIFIED, "verification disabled (LLM_VERIFY_FREE=false)"
            out.append(
                {
                    "id": m,
                    "status": st,
                    "detail": why,
                    "default": m == self.s.llm_default_model,
                    "fallback": m in self.s.llm_fallback_models,
                    "usable": st in (VERIFIED_FREE, UNVERIFIED),
                }
            )
        return out

    def check_model_allowed(self, model: str) -> str:
        """Raise ModelNotAllowedError unless `model` is configured and free. Returns status."""
        if model not in self.configured_models:
            raise ModelNotAllowedError(f"Model '{model}' is not in the configured free-model list")
        if not model.endswith(":free"):
            raise ModelNotAllowedError(f"Model '{model}' is not a ':free' OpenRouter model")
        if not self.s.llm_verify_free:
            return UNVERIFIED
        st, why = self.registry.status(model)
        if st == NOT_FREE:
            raise ModelNotAllowedError(f"Model '{model}' is not free on OpenRouter ({why})")
        if st == NOT_FOUND:
            raise LLMModelUnavailableError(f"Model '{model}' is not available on OpenRouter ({why})")
        return st

    # ------------------------------------------------------------------ calls
    def complete(self, messages: list[dict[str, str]], model: Optional[str] = None) -> LLMResponse:
        if not self.s.openrouter_api_key:
            raise LLMConfigError("OPENROUTER_API_KEY is not set (see .env.example)")
        primary = model or self.s.llm_default_model
        chain = [primary] + [m for m in self.s.llm_fallback_models if m != primary]
        attempts: list[dict[str, Any]] = []
        last_exc: Optional[LLMError] = None
        for m in chain:
            try:
                verification = self.check_model_allowed(m)
                resp = self._call_with_retries(m, messages, attempts)
                resp.attempts = attempts
                if verification == UNVERIFIED:
                    resp.attempts.append({"model": m, "note": "free status unverified"})
                return resp
            except ModelNotAllowedError as exc:
                attempts.append({"model": m, "error": str(exc)})
                last_exc = exc
                if m == primary:
                    raise  # never silently swap an explicitly disallowed choice
            except LLMError as exc:
                attempts.append({"model": m, "error": f"{type(exc).__name__}: {exc}"})
                last_exc = exc
                if not exc.retryable:
                    raise
                log.warning("Model %s failed (%s); trying next configured fallback", m, exc)
        assert last_exc is not None
        last_exc.args = (f"{last_exc} (attempts: {attempts})",)
        raise last_exc

    def _call_with_retries(self, model: str, messages, attempts) -> LLMResponse:
        delay = 2.0
        for i in range(self.s.llm_max_retries + 1):
            try:
                return self._call_once(model, messages)
            except (LLMRateLimitError, LLMTimeoutError, LLMProviderError, LLMInvalidResponseError) as exc:
                attempts.append({"model": model, "try": i + 1, "error": f"{type(exc).__name__}: {exc}"})
                if i == self.s.llm_max_retries:
                    raise
                wait = getattr(exc, "retry_after", None) or delay
                self._sleep(min(wait, 20.0))
                delay *= 2
        raise LLMProviderError("unreachable")  # pragma: no cover

    def _call_once(self, model: str, messages) -> LLMResponse:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.s.llm_temperature,
            "max_tokens": self.s.llm_max_tokens,
            # Hard cap: never route to a paid provider endpoint.
            "provider": {"max_price": {"prompt": 0, "completion": 0}},
            # Ask OpenRouter to report token counts and the billed cost (expected 0).
            "usage": {"include": True},
        }
        if self.s.llm_reasoning_effort:
            # Reasoning models otherwise may spend the whole budget thinking; ignored by others.
            payload["reasoning"] = {"effort": self.s.llm_reasoning_effort, "exclude": True}
        headers = {
            "Authorization": f"Bearer {self.s.openrouter_api_key}",
            "HTTP-Referer": self.s.app_referer,
            "X-Title": self.s.app_title,
        }
        t0 = time.time()
        try:
            r = self.http.post(
                f"{self.s.openrouter_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.s.llm_timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"timeout after {self.s.llm_timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"network error: {exc}") from exc
        latency = time.time() - t0

        try:
            body = r.json()
        except ValueError:
            body = None
        err_msg = ""
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            err_msg = str(body["error"].get("message", ""))
            # OpenRouter puts the upstream provider and its raw reason in metadata.
            meta = body["error"].get("metadata") or {}
            if isinstance(meta, dict):
                extra = [str(meta[k])[:200] for k in ("provider_name", "raw") if meta.get(k)]
                if extra:
                    err_msg = f"{err_msg} ({' | '.join(extra)})"
        code = r.status_code
        if isinstance(body, dict) and isinstance(body.get("error"), dict) and code == 200:
            code = int(body["error"].get("code") or 502)

        if code == 401 or (code == 403 and re.search(r"\b(api key|auth|unauthori[sz]ed|credential)", err_msg, re.I)):
            raise LLMConfigError(f"OpenRouter rejected the API key ({code}): {err_msg}")
        if code == 403:
            # Model-level restriction (e.g. "only available on agentic harnesses", moderation)
            raise LLMModelUnavailableError(f"model not usable with this account ({code}): {err_msg}")
        if code == 402:
            raise ModelNotAllowedError(f"OpenRouter requires payment for '{model}' (402): {err_msg}")
        if code == 429:
            exc = LLMRateLimitError(f"rate limited: {err_msg or 'too many requests'}")
            try:
                exc.retry_after = float(r.headers.get("Retry-After", ""))  # type: ignore[attr-defined]
            except ValueError:
                pass
            raise exc
        if code in (404,) or (code == 400 and "model" in err_msg.lower()):
            raise LLMModelUnavailableError(f"model unavailable ({code}): {err_msg}")
        if code == 408:
            raise LLMTimeoutError(f"provider timeout: {err_msg}")
        if code >= 500 or code == 400:
            raise LLMProviderError(f"provider error ({code}): {err_msg}")
        if not isinstance(body, dict):
            raise LLMInvalidResponseError("response is not JSON")

        try:
            choice = body["choices"][0]
            content = (choice.get("message") or {}).get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError(f"unexpected response shape: {str(body)[:200]}") from exc
        if not content or not str(content).strip():
            reason = choice.get("finish_reason")
            if reason == "length" and (choice.get("message") or {}).get("reasoning"):
                raise LLMInvalidResponseError(
                    "model used its whole token budget on reasoning and returned no answer "
                    "(raise LLM_MAX_TOKENS or set LLM_REASONING_EFFORT=low)"
                )
            raise LLMInvalidResponseError(f"empty completion (finish_reason={reason})")

        u = body.get("usage") or {}
        # OpenRouter normalises to OpenAI's "length"; the provider's own value is a fallback.
        # A truncation reported by either field wins.
        reasons = [str(r) for r in (choice.get("finish_reason"), choice.get("native_finish_reason")) if r]
        finish = next((r for r in reasons if r.lower() in TRUNCATION_REASONS), reasons[0] if reasons else None)
        return LLMResponse(
            text=str(content).strip(),
            model=body.get("model") or model,
            latency_s=round(latency, 3),
            finish_reason=finish,
            usage=LLMUsage(
                prompt_tokens=u.get("prompt_tokens"),
                completion_tokens=u.get("completion_tokens"),
                total_tokens=u.get("total_tokens"),
                cost=u.get("cost"),
            ),
        )
