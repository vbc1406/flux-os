"""HTTP server: an OpenAI- and Anthropic-compatible routing proxy.

Endpoints
    POST /v1/chat/completions     OpenAI Chat Completions (stream + tools)
    POST /v1/responses            OpenAI Responses API (stream + function tools)
    POST /v1/messages             Anthropic Messages API (stream + tools)
    POST /v1/messages/count_tokens
    POST /v1/route                routing decision only, no model call
    GET  /v1/models               directives + available models
    GET  /health
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ._version import __version__
from .classifier import classify
from .client import FluxOS, RouteInfo, UpstreamError, attempts_header
from .providers import EMPTY_REPLY_MESSAGE, ProviderError
from .router import MODES, RoutingError, parse_directive
from .translate import (
    ChatStreamToAnthropic,
    ChatStreamToResponses,
    anthropic_request_to_chat,
    chat_to_anthropic_response,
    chat_to_responses,
    responses_request_to_chat,
)
from .validate import validate_anthropic_body, validate_chat_request, validate_responses_body

DIRECTIVES = ("auto", "flux-cheap", "flux-fast")


def _route_headers(info: RouteInfo) -> dict[str, str]:
    a = info.decision.analysis
    return {
        "x-flux-model": info.model.id,
        "x-flux-provider": info.model.provider,
        "x-flux-rerouted": "true" if info.rerouted else "false",
        "x-flux-mode": "pinned" if info.decision.pinned else info.decision.mode,
        "x-flux-task": a.task,
        "x-flux-complexity": f"{a.complexity:.2f}",
        "x-flux-attempts": attempts_header(info.attempts),
    }


def _error_headers(exc: UpstreamError) -> dict[str, str]:
    """The x-flux-* headers for an upstream failure (what was tried, and why we got here)."""
    if exc.decision is None or not exc.attempts:
        return {}
    last = exc.attempts[-1]["model"]
    model = next((m for m in exc.decision.candidates if m.id == last), exc.decision.model)
    return _route_headers(RouteInfo(exc.decision, model, exc.attempts))


def _openai_error(
    status: int,
    message: str,
    kind: str = "invalid_request_error",
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": status}}, status_code=status, headers=headers
    )


def _anthropic_error(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    kind = {
        400: "invalid_request_error",
        401: "authentication_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }.get(status, "api_error")
    return JSONResponse(
        {"type": "error", "error": {"type": kind, "message": message}}, status_code=status, headers=headers
    )


def _sse(event: str | None, data: Any) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n" if event else f"data: {payload}\n\n"


# Anything raised by request parsing/translation for a malformed body is the client's fault.
_MALFORMED = (TypeError, ValueError, AttributeError, KeyError, IndexError, RecursionError, OverflowError)


def _malformed(exc: Exception) -> RoutingError:
    print(f"flux-os: malformed request: {type(exc).__name__}", file=sys.stderr)
    return RoutingError("malformed request body")


def _log_upstream_failure(context: str, detail: str) -> None:
    print(f"flux-os: {context} failed: {detail}", file=sys.stderr)


def _upstream_message(debug: bool, detail: str, exc: Exception | None = None) -> str:
    if getattr(exc, "attempts", None) and exc.attempts[-1].get("code") == "empty_reply":  # type: ignore[union-attr]
        return EMPTY_REPLY_MESSAGE  # fixed text, safe to show and actionable
    return detail if debug else "upstream request failed"


def _json_size(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, separators=(",", ":")).encode("utf-8"))


class _History:
    """Bounded memory of Responses API turns, for ``previous_response_id``."""

    def __init__(self, size: int = 2000, max_bytes: int = 50_000_000) -> None:
        self.size = size
        self.max_bytes = max_bytes
        self.data: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._bytes: dict[str, int] = {}
        self.total_bytes = 0
        self.lock = threading.Lock()

    def get(self, key: str | None) -> list[dict[str, Any]] | None:
        if not key:
            return None
        with self.lock:
            return self.data.get(key)

    def put(self, key: str, messages: list[dict[str, Any]]) -> None:
        with self.lock:
            if key in self.data:
                self.total_bytes -= self._bytes.pop(key)
            size = _json_size(messages)
            self.data[key] = messages
            self._bytes[key] = size
            self.total_bytes += size
            while self.data and (len(self.data) > self.size or self.total_bytes > self.max_bytes):
                oldest, _ = self.data.popitem(last=False)
                self.total_bytes -= self._bytes.pop(oldest)


def _assistant_message(resp: dict[str, Any]) -> dict[str, Any]:
    msg = dict(((resp.get("choices") or [{}])[0]).get("message") or {"role": "assistant", "content": ""})
    msg.pop("refusal", None)
    return msg


def create_app(flux: FluxOS | None = None, api_key: str | None = None) -> FastAPI:
    flux = flux or FluxOS()
    api_key = api_key if api_key is not None else os.environ.get("FLUX_OS_API_KEY") or None
    max_body_bytes = int(os.environ.get("FLUX_OS_MAX_BODY_BYTES", "10485760"))
    debug_errors = os.environ.get("FLUX_OS_DEBUG_ERRORS", "").strip().lower() in ("1", "true", "yes", "on")
    history = _History()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await flux.aclose()

    app = FastAPI(title="flux-os", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.flux = flux

    @app.middleware("http")
    async def body_limit(request: Request, call_next: Any) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_big = int(content_length) > max_body_bytes
            except ValueError:
                too_big = False
            if too_big:
                return _openai_error(413, f"request body exceeds {max_body_bytes} bytes")
        else:
            total = 0
            chunks: list[bytes] = []
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_body_bytes:
                    return _openai_error(413, f"request body exceeds {max_body_bytes} bytes")
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.middleware("http")
    async def auth(request: Request, call_next: Any) -> Response:
        if api_key and request.url.path.startswith("/v1/"):
            header = request.headers.get("authorization", "")
            token = (
                header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
            )
            if not secrets.compare_digest(token.encode(), api_key.encode()):
                if request.url.path.startswith("/v1/messages"):
                    return _anthropic_error(401, "invalid flux-os API key")
                return _openai_error(401, "invalid flux-os API key", "authentication_error")
        return await call_next(request)

    async def read_json(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except (ValueError, RecursionError) as exc:
            raise RoutingError("request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise RoutingError("request body must be a JSON object")
        return body

    def apply_headers(request: Request, req: dict[str, Any]) -> bool | None:
        mode = request.headers.get("x-flux-mode", "").strip().lower()
        if mode:
            if mode not in MODES:
                raise RoutingError(f"X-Flux-Mode must be one of {', '.join(MODES)}")
            if parse_directive(req.get("model"))[0] or flux.route_all:
                req["model"] = mode
        reroute = request.headers.get("x-flux-reroute")
        if reroute is None:
            return None
        return reroute.strip().lower() in ("1", "true", "yes", "on")

    # ── meta ────────────────────────────────────────────────────────────────
    @app.get("/")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        data = [{"id": d, "object": "model", "created": 0, "owned_by": "flux-os"} for d in DIRECTIVES]
        data += [
            {"id": m.id, "object": "model", "created": 0, "owned_by": m.provider}
            for m in flux.catalog.available()
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/route")
    async def route(request: Request) -> Response:
        try:
            req = await read_json(request)
            if req.get("messages") is None:
                req["messages"] = []
            validate_chat_request(req)
            apply_headers(request, req)
            decision = flux.route(
                req["messages"],
                model=req.get("model"),
                tools=req.get("tools"),
                response_format=req.get("response_format"),
                max_tokens=req.get("max_completion_tokens") or req.get("max_tokens"),
            )
        except RoutingError as exc:
            return _openai_error(400, str(exc))
        except _MALFORMED as exc:
            return _openai_error(400, str(_malformed(exc)))
        return JSONResponse(decision.as_dict())

    # ── OpenAI Chat Completions ─────────────────────────────────────────────
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            req = await read_json(request)
            if not isinstance(req.get("messages"), list) or not req["messages"]:
                raise RoutingError("'messages' must be a non-empty list")
            validate_chat_request(req)
            reroute = apply_headers(request, req)
            if req.get("stream"):
                stream, info = await flux.dispatch_stream(req, reroute=reroute)
                return StreamingResponse(
                    _chat_sse(stream, debug_errors),
                    media_type="text/event-stream",
                    headers=_route_headers(info),
                )
            resp, info = await flux.dispatch(req, reroute=reroute)
            return JSONResponse(resp, headers=_route_headers(info))
        except RoutingError as exc:
            return _openai_error(400, str(exc))
        except _MALFORMED as exc:
            return _openai_error(400, str(_malformed(exc)))
        except UpstreamError as exc:
            _log_upstream_failure("chat completion", str(exc))
            error: dict[str, Any] = {
                "message": _upstream_message(debug_errors, str(exc), exc),
                "type": "upstream_error",
                "code": exc.status_code,
            }
            if debug_errors:
                error["attempts"] = exc.attempts
            return JSONResponse({"error": error}, status_code=exc.status_code, headers=_error_headers(exc))

    # ── OpenAI Responses API ────────────────────────────────────────────────
    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        try:
            body = await read_json(request)
            validate_responses_body(body)
            prev = body.get("previous_response_id")
            past = history.get(prev)
            if prev and past is None:
                return _openai_error(
                    404, f"previous response '{prev}' not found (flux-os keeps recent ones in memory)"
                )
            req = responses_request_to_chat(body, past)
            if not req["messages"]:
                raise RoutingError("'input' must not be empty")
            validate_chat_request(req)
            reroute = apply_headers(request, req)
            resp_id = f"resp_{uuid.uuid4().hex}"
            if req.get("stream"):
                stream, info = await flux.dispatch_stream(req, reroute=reroute)
                return StreamingResponse(
                    _responses_sse(
                        stream, body, info.model.id, resp_id, req["messages"], history, debug_errors
                    ),
                    media_type="text/event-stream",
                    headers=_route_headers(info),
                )
            resp, info = await flux.dispatch(req, reroute=reroute)
            if body.get("store", True) is not False:
                history.put(resp_id, req["messages"] + [_assistant_message(resp)])
            return JSONResponse(
                chat_to_responses(resp, body, info.model.id, resp_id), headers=_route_headers(info)
            )
        except RoutingError as exc:
            return _openai_error(400, str(exc))
        except _MALFORMED as exc:
            return _openai_error(400, str(_malformed(exc)))
        except UpstreamError as exc:
            _log_upstream_failure("responses", str(exc))
            return _openai_error(
                exc.status_code,
                _upstream_message(debug_errors, str(exc), exc),
                "upstream_error",
                _error_headers(exc),
            )

    # ── Anthropic Messages API ──────────────────────────────────────────────
    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        try:
            body = await read_json(request)
            validate_anthropic_body(body)
            req = anthropic_request_to_chat(body)
            validate_chat_request(req)
            a = classify(req["messages"], tools=req.get("tools"))
        except RoutingError as exc:
            return _anthropic_error(400, str(exc))
        except _MALFORMED as exc:
            return _anthropic_error(400, str(_malformed(exc)))
        return JSONResponse({"input_tokens": a.input_tokens})

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        try:
            body = await read_json(request)
            validate_anthropic_body(body)
            req = anthropic_request_to_chat(body)
            if not req["messages"]:
                raise RoutingError("'messages' must not be empty")
            validate_chat_request(req)
            reroute = apply_headers(request, req)
            if req.get("stream"):
                req["stream_options"] = {"include_usage": True}
                stream, info = await flux.dispatch_stream(req, reroute=reroute)
                return StreamingResponse(
                    _anthropic_sse(stream, info.model.id, debug_errors),
                    media_type="text/event-stream",
                    headers=_route_headers(info),
                )
            resp, info = await flux.dispatch(req, reroute=reroute)
            return JSONResponse(chat_to_anthropic_response(resp, info.model.id), headers=_route_headers(info))
        except RoutingError as exc:
            return _anthropic_error(400, str(exc))
        except _MALFORMED as exc:
            return _anthropic_error(400, str(_malformed(exc)))
        except UpstreamError as exc:
            _log_upstream_failure("messages", str(exc))
            return _anthropic_error(
                exc.status_code, _upstream_message(debug_errors, str(exc), exc), _error_headers(exc)
            )

    return app


async def _chat_sse(stream: AsyncIterator[dict[str, Any]], debug: bool = False) -> AsyncIterator[str]:
    try:
        async for chunk in stream:
            yield _sse(None, chunk)
    except ProviderError as exc:
        _log_upstream_failure("chat completion stream", str(exc))
        yield _sse(None, {"error": {"message": _upstream_message(debug, str(exc)), "type": "upstream_error"}})
    yield "data: [DONE]\n\n"


async def _anthropic_sse(
    stream: AsyncIterator[dict[str, Any]], model: str, debug: bool = False
) -> AsyncIterator[str]:
    conv = ChatStreamToAnthropic(model)
    for ev, data in conv.start():
        yield _sse(ev, data)
    try:
        async for chunk in stream:
            for ev, data in conv.feed(chunk):
                yield _sse(ev, data)
    except ProviderError as exc:
        _log_upstream_failure("messages stream", str(exc))
        message = _upstream_message(debug, str(exc))
        yield _sse("error", {"type": "error", "error": {"type": "api_error", "message": message}})
        return
    for ev, data in conv.end():
        yield _sse(ev, data)


async def _responses_sse(
    stream: AsyncIterator[dict[str, Any]],
    body: dict[str, Any],
    model: str,
    resp_id: str,
    messages: list[dict[str, Any]],
    history: _History,
    debug: bool = False,
) -> AsyncIterator[str]:
    conv = ChatStreamToResponses(body, model, resp_id)
    for ev, data in conv.start():
        yield _sse(ev, data)
    try:
        async for chunk in stream:
            for ev, data in conv.feed(chunk):
                yield _sse(ev, data)
    except ProviderError as exc:
        _log_upstream_failure("responses stream", str(exc))
        conv.seq += 1
        message = _upstream_message(debug, str(exc))
        yield _sse("error", {"type": "error", "sequence_number": conv.seq, "message": message})
        return
    for ev, data in conv.end():
        yield _sse(ev, data)
    if body.get("store", True) is not False:
        text = conv.final.get("output_text") or None
        calls = [
            {
                "id": i["call_id"],
                "type": "function",
                "function": {"name": i["name"], "arguments": i["arguments"]},
            }
            for i in conv.final["output"]
            if i["type"] == "function_call"
        ]
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            msg["tool_calls"] = calls
        history.put(resp_id, messages + [msg])
