"""Upstream HTTP calls. Two adapters: OpenAI-compatible and Anthropic."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._version import __version__
from .catalog import Model, Provider
from .translate import AnthropicStreamToChat, anthropic_to_chat, chat_to_anthropic

# Parameters forwarded to non-OpenAI OpenAI-compatible providers. Anything else
# is dropped so a request can be rerouted across providers without a 400/422.
_PORTABLE = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stop",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "seed",
    "n",
    "presence_penalty",
    "frequency_penalty",
}
_PROVIDER_EXTRA = {
    "openai": None,  # None = forward everything
    "google": {"reasoning_effort", "stream_options", "user"},
    "groq": {"reasoning_effort", "stream_options", "user", "logprobs", "top_logprobs"},
    "mistral": set(),
    "deepseek": {"stream_options", "logprobs", "top_logprobs"},
    "openrouter": None,
    "ollama": {"stream_options"},
}
_INTERNAL = {"flux_mode", "flux_reroute"}


class ProviderError(Exception):
    """An upstream call failed. ``retryable`` means another model may succeed."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = True, provider: str = ""):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.provider = provider


def _without_extra_content(message: dict[str, Any]) -> dict[str, Any]:
    """Drop Gemini's per-tool-call ``extra_content`` (thought_signature) for other providers,
    so a tool loop that started on Gemini can continue on a model that rejects unknown fields."""
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not any(isinstance(c, dict) and "extra_content" in c for c in calls):
        return message
    return {
        **message,
        "tool_calls": [
            {k: v for k, v in c.items() if k != "extra_content"} if isinstance(c, dict) else c for c in calls
        ],
    }


def _retryable(status: int, text: str) -> bool:
    if status in (400,):
        lowered = text.lower()
        return any(
            s in lowered
            for s in (
                "not supported",
                "unsupported",
                "does not support",
                "context",
                "too long",
                "maximum",
                "thought_signature",
            )
        )
    return status not in (499,)


def _error_message(status: int, text: str) -> str:
    try:
        data = json.loads(text)
        err = data.get("error") if isinstance(data, dict) else None
        msg = err.get("message") if isinstance(err, dict) else err or data.get("message")
        if msg:
            return f"HTTP {status}: {str(msg)[:400]}"
    except (ValueError, AttributeError):
        pass
    return f"HTTP {status}: {text[:400]}"


def _timeout() -> httpx.Timeout:
    total = float(os.environ.get("FLUX_OS_TIMEOUT", "120"))
    return httpx.Timeout(total, connect=float(os.environ.get("FLUX_OS_CONNECT_TIMEOUT", "10")))


class Upstream:
    """Makes calls to providers. One shared ``httpx.AsyncClient`` per instance."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_timeout())
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── request building ────────────────────────────────────────────────────
    def _openai_body(self, req: dict[str, Any], model: Model, provider: Provider) -> dict[str, Any]:
        extra = _PROVIDER_EXTRA.get(provider.name, {"reasoning_effort", "stream_options", "user"})
        body = {k: v for k, v in req.items() if k not in _INTERNAL and v is not None}
        if extra is not None:
            body = {k: v for k, v in body.items() if k in _PORTABLE or k in extra}
            if "max_completion_tokens" in req and "max_tokens" not in body:
                body["max_tokens"] = req["max_completion_tokens"]
            if provider.name == "mistral" and "seed" in body:
                body["random_seed"] = body.pop("seed")
            body["messages"] = [
                {**m, "role": "system"} if m.get("role") == "developer" else m for m in body["messages"]
            ]
        if provider.name == "openai" and "max_tokens" in body:
            body.setdefault("max_completion_tokens", body.pop("max_tokens"))
        if provider.name != "google":
            body["messages"] = [_without_extra_content(m) for m in body["messages"]]
        for p in model.drop_params:
            body.pop(p, None)
        if not body.get("tools"):
            body.pop("tool_choice", None)
            body.pop("parallel_tool_calls", None)
        limit_key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        if limit_key in body and model.max_output_tokens:
            body[limit_key] = min(int(body[limit_key]), model.max_output_tokens)
        body["model"] = model.upstream_id
        return body

    def _headers(self, provider: Provider) -> dict[str, str]:
        headers = {"content-type": "application/json", "user-agent": f"flux-os/{__version__}"}
        key = provider.resolve_key()
        if provider.style == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
            if key:
                headers["x-api-key"] = key
        elif key:
            headers["authorization"] = f"Bearer {key}"
        headers.update(provider.headers)
        return headers

    def _url(self, provider: Provider) -> str:
        base = provider.base_url.rstrip("/")
        if provider.style == "anthropic":
            return base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        return base + "/chat/completions"

    def _prepare(self, req: dict[str, Any], model: Model, provider: Provider) -> dict[str, Any]:
        if provider.style == "anthropic":
            return chat_to_anthropic(req, model.upstream_id, model.max_output_tokens or 8192)
        return self._openai_body(req, model, provider)

    # ── calls ───────────────────────────────────────────────────────────────
    async def complete(self, req: dict[str, Any], model: Model, provider: Provider) -> dict[str, Any]:
        """Non-streaming call. Returns an OpenAI chat completion dict."""
        body = self._prepare({**req, "stream": False}, model, provider)
        body.pop("stream", None)
        body.pop("stream_options", None)
        try:
            r = await self.client.post(self._url(provider), headers=self._headers(provider), json=body)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", provider=provider.name) from exc
        if r.status_code >= 400:
            raise ProviderError(
                _error_message(r.status_code, r.text),
                status=r.status_code,
                retryable=_retryable(r.status_code, r.text),
                provider=provider.name,
            )
        try:
            data = r.json()
        except ValueError as exc:
            raise ProviderError("upstream returned invalid JSON", provider=provider.name) from exc
        if provider.style == "anthropic":
            return anthropic_to_chat(data, model.id)
        if not data.get("choices"):
            raise ProviderError(_error_message(r.status_code, r.text), provider=provider.name)
        data["model"] = model.id
        return data

    async def stream(
        self, req: dict[str, Any], model: Model, provider: Provider
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming call. Yields OpenAI chat.completion.chunk dicts.

        Errors before the first chunk raise ProviderError (so the caller can reroute).
        """
        body = self._prepare({**req, "stream": True}, model, provider)
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        converter = AnthropicStreamToChat(model.id, include_usage) if provider.style == "anthropic" else None
        try:
            cm = self.client.stream("POST", self._url(provider), headers=self._headers(provider), json=body)
            response = await cm.__aenter__()
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", provider=provider.name) from exc
        try:
            if response.status_code >= 400:
                text = (await response.aread()).decode("utf-8", "replace")
                raise ProviderError(
                    _error_message(response.status_code, text),
                    status=response.status_code,
                    retryable=_retryable(response.status_code, text),
                    provider=provider.name,
                )
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("error"):
                    err = event["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise ProviderError(f"stream error: {msg}", provider=provider.name)
                if converter is not None:
                    for chunk in converter.feed(event):
                        yield chunk
                else:
                    event["model"] = model.id
                    yield event
        except httpx.HTTPError as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", provider=provider.name) from exc
        finally:
            await cm.__aexit__(None, None, None)
