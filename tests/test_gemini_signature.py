"""Gemini 3 tool loops through the Anthropic Messages path must round-trip ``thought_signature``."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from flux_os import Catalog, FluxOS
from flux_os.server import create_app

GEMINI = "gemini-3-flash-preview"
SIG = "c2lnbmF0dXJl+/=="  # base64-ish: contains characters that are not valid in a tool id
CALLS = [
    {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        "extra_content": {"google": {"thought_signature": SIG}},
    },
    # parallel call: Gemini only signs the first one, but still validates "position 2"
    {
        "id": "call_2",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
    },
]
TOOLS_ANTHROPIC = [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}]


class StrictGemini:
    """Fake upstream that rejects a tool-result turn whose first function call lost its signature."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        model = body["model"]
        msgs = body["messages"]
        if msgs[-1].get("role") != "tool":
            if body.get("stream"):
                chunk = {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": [{"index": i, **c} for i, c in enumerate(CALLS)]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
                return httpx.Response(200, content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
            message = {"role": "assistant", "content": None, "tool_calls": CALLS}
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                },
            )
        if model.startswith("gemini"):
            assistant = next(m for m in reversed(msgs) if m.get("role") == "assistant")
            sig = ((assistant["tool_calls"][0].get("extra_content") or {}).get("google") or {}).get(
                "thought_signature"
            )
            if sig != SIG:
                err = "Function call is missing a thought_signature in functionCall parts. position 2"
                return httpx.Response(400, json={"error": {"message": err}})
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )


@pytest.fixture
def gem() -> StrictGemini:
    return StrictGemini()


@pytest.fixture
def client(keys: None, gem: StrictGemini) -> httpx.AsyncClient:
    flux = FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(gem.handler))
    app = create_app(flux, api_key="")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux")


def _first(model: str = GEMINI, **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 100,
        "tools": TOOLS_ANTHROPIC,
        "messages": [{"role": "user", "content": "weather in Paris and Rome?"}],
        **extra,
    }


def _second(first_body: dict[str, Any], content: list[dict[str, Any]]) -> dict[str, Any]:
    results = [{"type": "tool_result", "tool_use_id": b["id"], "content": "18C"} for b in content]
    return {
        **first_body,
        "messages": first_body["messages"]
        + [{"role": "assistant", "content": content}, {"role": "user", "content": results}],
    }


def _signature_sent(body: dict[str, Any]) -> list[Any]:
    assistant = next(m for m in reversed(body["messages"]) if m.get("role") == "assistant")
    return [
        (tc.get("extra_content") or {}).get("google", {}).get("thought_signature")
        for tc in assistant["tool_calls"]
    ]


async def test_signature_round_trips_through_anthropic_blocks(client, gem) -> None:
    first = _first()
    r = await client.post("/v1/messages", json=first)
    blocks = r.json()["content"]
    assert [b["id"] for b in blocks] == ["call_1", "call_2"]
    assert blocks[0]["extra_content"]["google"]["thought_signature"] == SIG
    r2 = await client.post("/v1/messages", json=_second(first, blocks))
    assert r2.status_code == 200, r2.text
    assert _signature_sent(gem.calls[-1]) == [SIG, None]  # restored on the first parallel call


async def test_signature_round_trips_when_streaming(client, gem) -> None:
    first = _first(stream=True)
    r = await client.post("/v1/messages", json=first)
    starts = [
        json.loads(line[5:])
        for line in r.text.splitlines()
        if line.startswith("data:") and "content_block_start" in line
    ]
    blocks = [s["content_block"] for s in starts]
    assert blocks[0]["extra_content"]["google"]["thought_signature"] == SIG
    r2 = await client.post("/v1/messages", json=_second({**first, "stream": False}, blocks))
    assert r2.status_code == 200, r2.text
    assert _signature_sent(gem.calls[-1]) == [SIG, None]


async def test_client_that_strips_unknown_fields_still_works(client, gem) -> None:
    first = _first()
    blocks = (await client.post("/v1/messages", json=first)).json()["content"]
    stripped = [{k: v for k, v in b.items() if k != "extra_content"} for b in blocks]
    r2 = await client.post("/v1/messages", json=_second(first, stripped))
    assert r2.status_code == 200, r2.text
    assert _signature_sent(gem.calls[-1]) == [SIG, None]  # restored from the server-side memory


async def test_lost_signature_400_is_rerouted(client, gem) -> None:
    """After a restart the server has no memory; the 400 must reroute instead of reaching the client."""
    first = _first()
    blocks = (await client.post("/v1/messages", json=first)).json()["content"]
    stripped = [{k: v for k, v in b.items() if k != "extra_content"} for b in blocks]
    client._transport.app.state.flux.router._extra_content.data.clear()  # type: ignore[attr-defined]
    r2 = await client.post("/v1/messages", json=_second(first, stripped), headers={"x-flux-reroute": "true"})
    assert r2.status_code == 200, r2.text
    assert r2.headers["x-flux-rerouted"] == "true"
    assert not gem.calls[-1]["model"].startswith("gemini")
    assert "extra_content" not in json.dumps(gem.calls[-1]["messages"])  # never sent to other providers


async def test_chat_path_unchanged(client, gem) -> None:
    """The Chat Completions path keeps forwarding the signature verbatim to Gemini."""
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": GEMINI,
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    msg = r.json()["choices"][0]["message"]
    assert msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == SIG
    r2 = await client.post(
        "/v1/chat/completions",
        json={
            "model": GEMINI,
            "messages": [
                {"role": "user", "content": "hi"},
                msg,
                {"role": "tool", "tool_call_id": "call_1", "content": "x"},
                {"role": "tool", "tool_call_id": "call_2", "content": "y"},
            ],
        },
    )
    assert r2.status_code == 200
