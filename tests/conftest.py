"""Shared fixtures: a fake upstream that speaks both OpenAI and Anthropic wire formats."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from flux_os import Catalog, FluxOS
from flux_os.server import create_app

PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in (
        "FLUX_OS_DISABLE",
        "FLUX_OS_ONLY_MODELS",
        "FLUX_OS_DEFAULT_MODE",
        "FLUX_OS_ROUTE_ALL",
        "FLUX_OS_REROUTE_PINNED",
        "FLUX_OS_API_KEY",
        "FLUX_OS_QUALITY_OFFSET",
        "FLUX_OS_MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(__import__("tempfile").mkdtemp())


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(key, "test-key")


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj)}\n\n"


class FakeUpstream:
    """Scriptable fake of every provider. ``fail[upstream_model] = status``."""

    def __init__(self) -> None:
        self.fail: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        self.tool_call = False  # respond with a tool call when tools are offered

    @property
    def models_called(self) -> list[str]:
        return [c["body"]["model"] for c in self.calls]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        style = "anthropic" if request.url.path.endswith("/v1/messages") else "openai"
        self.calls.append(
            {"url": str(request.url), "style": style, "body": body, "headers": dict(request.headers)}
        )
        model = body.get("model")
        if model in self.fail:
            status = self.fail[model]
            return httpx.Response(status, json={"error": {"message": f"simulated {status} for {model}"}})
        wants_tool = self.tool_call and body.get("tools") and not _ends_with_tool_result(body, style)
        text = f"hello from {model}"
        if style == "anthropic":
            return self._anthropic(body, model, text, wants_tool)
        return self._openai(body, model, text, wants_tool)

    def _openai(self, body: dict[str, Any], model: str, text: str, tool: bool) -> httpx.Response:
        call = {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        if body.get("stream"):
            chunks = [
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                    ],
                }
            ]
            if tool:
                chunks.append(
                    {
                        "id": "c1",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": 0, **call}]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                finish = "tool_calls"
            else:
                for word in text.split(" "):
                    chunks.append(
                        {
                            "id": "c1",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                            ],
                        }
                    )
                finish = "stop"
            chunks.append(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }
            )
            payload = "".join(_sse(c) for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})
        message: dict[str, Any] = {"role": "assistant", "content": None if tool else text}
        if tool:
            message["tool_calls"] = [call]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls" if tool else "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    def _anthropic(self, body: dict[str, Any], model: str, text: str, tool: bool) -> httpx.Response:
        if body.get("stream"):
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "usage": {"input_tokens": 12, "output_tokens": 1},
                        },
                    },
                ),
            ]
            if tool:
                events += [
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {
                                "type": "tool_use",
                                "id": "toolu_xyz",
                                "name": "get_weather",
                                "input": {},
                            },
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "input_json_delta", "partial_json": '{"city": '},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ]
                stop = "tool_use"
            else:
                events += [
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ]
                stop = "end_turn"
            events += [
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 7}},
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            payload = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)
            return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})
        content: list[dict[str, Any]] = (
            [{"type": "tool_use", "id": "toolu_xyz", "name": "get_weather", "input": {"city": "Paris"}}]
            if tool
            else [{"type": "text", "text": text}]
        )
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": "tool_use" if tool else "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 7},
            },
        )


def _ends_with_tool_result(body: dict[str, Any], style: str) -> bool:
    msgs = body.get("messages") or []
    if not msgs:
        return False
    last = msgs[-1]
    if style == "openai":
        return last.get("role") == "tool"
    content = last.get("content")
    return isinstance(content, list) and any(b.get("type") == "tool_result" for b in content)


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def flux(keys: None, fake: FakeUpstream) -> FluxOS:
    return FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(fake.handler))


@pytest.fixture
def app(flux: FluxOS):
    return create_app(flux, api_key="")


@pytest.fixture
def live_url(app):
    """Run the app on a real socket (the way users run it)."""
    import socket
    import threading
    import time

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)
