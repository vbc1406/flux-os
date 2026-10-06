"""Reasoning models that burn the whole max_tokens budget on hidden reasoning return empty replies."""

from __future__ import annotations

import json

import httpx
import pytest

from flux_os import Catalog, FluxOS
from flux_os.server import create_app

BODY = {"model": "auto", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}


def _resp(model: str, content: str, finish: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 48},
            },
        },
    )


def _client(handler) -> httpx.AsyncClient:
    flux = FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(handler))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(flux, api_key="")), base_url="http://f"
    )


@pytest.fixture
def first_model(keys, fake) -> str:
    flux = FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(fake.handler))
    return flux.route(BODY["messages"], model="auto", max_tokens=50).model.upstream_id


async def test_empty_length_reply_reroutes(first_model) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == first_model:
            return _resp(model, "", "length")
        return _resp(model, "hello", "stop")

    async with _client(handler) as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello"
    assert r.headers["x-flux-rerouted"] == "true" and seen[0] == first_model
    assert r.headers["x-flux-attempts"].split(",")[0].endswith(":empty_reply")


async def test_all_empty_gives_clear_error(keys) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _resp(json.loads(request.content)["model"], "", "length")

    async with _client(handler) as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 502
    assert "max_tokens" in r.json()["error"]["message"] and "reasoning" in r.json()["error"]["message"]


async def test_pinned_empty_is_clear_error_and_normal_replies_untouched(keys) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        body = json.loads(request.content)
        return _resp(model, "", "length") if body.get("max_tokens", 0) <= 50 else _resp(model, "ok", "stop")

    async with _client(handler) as c:
        pinned = await c.post("/v1/chat/completions", json={**BODY, "model": "gpt-oss-20b"})
        assert pinned.status_code == 502 and "max_tokens" in pinned.json()["error"]["message"]
        fine = await c.post("/v1/chat/completions", json={**BODY, "model": "gpt-oss-20b", "max_tokens": 2000})
        assert fine.status_code == 200
        # empty content with a *stop* finish is a legitimate reply, not a reasoning failure

    def stop_handler(request: httpx.Request) -> httpx.Response:
        return _resp("m", "", "stop")

    async with _client(stop_handler) as c:
        r = await c.post("/v1/chat/completions", json={**BODY, "model": "gpt-oss-20b"})
        assert r.status_code == 200
