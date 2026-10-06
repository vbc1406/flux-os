"""Type-invalid input is a 400 on every endpoint; nothing may 5xx or hang."""

from __future__ import annotations

import asyncio
import copy
import json
import random
from typing import Any

import httpx
import pytest

ENDPOINTS = (
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/route",
)

VALID: dict[str, dict[str, Any]] = {
    "/v1/chat/completions": {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    "/v1/route": {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
    "/v1/responses": {"model": "auto", "input": "hi"},
    "/v1/messages": {"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    "/v1/messages/count_tokens": {"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
}

NESTED_JSON = (
    "[" * 5000 + '"x"' + "]" * 5000
)  # raw text: building it as a Python object would blow the client's own json.dumps
WEIRD: list[Any] = [
    5,
    -1,
    0,
    1.5,
    1e308,
    True,
    None,
    "abc",
    "",
    [],
    {},
    [1, 2],
    ["a"],
    [None],
    {"a": {"b": 1}},
    [[]],
    [{"type": 5}],
    [{"type": "text", "text": 5}],
    [{"type": "image_url", "image_url": 5}],
    [{"type": "tool_use", "input": 5}],
    [{"type": "tool_result", "content": 5}],
    {"type": "function", "function": 5},
    "\x00",
    "é" * 100,
    10**30,
]
FIELDS = (
    "model",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "stream",
    "n",
    "stop",
    "stop_sequences",
    "messages",
    "input",
    "system",
    "instructions",
    "response_format",
    "parallel_tool_calls",
    "text",
    "user",
    "seed",
    "previous_response_id",
    "stream_options",
)


@pytest.fixture
def http(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux")


@pytest.mark.parametrize("path", ENDPOINTS)
@pytest.mark.parametrize(
    "patch",
    [{"model": 5}, {"max_tokens": "abc"}, {"temperature": "hot"}, {"stream": "yes"}, {"tools": "x"}],
)
async def test_type_invalid_scalars_are_400(http, path, patch) -> None:
    if path == "/v1/responses" and "max_tokens" in patch:
        patch = {"max_output_tokens": "abc"}
    r = await http.post(path, json={**VALID[path], **patch})
    assert r.status_code == 400, (path, patch, r.text)
    assert "error" in r.json()


@pytest.mark.parametrize("path", ENDPOINTS)
async def test_error_bodies_do_not_echo_values(http, path) -> None:
    secret = "sk-SECRET-VALUE-123"
    r = await http.post(path, json={**VALID[path], "tools": secret, "temperature": secret})
    assert r.status_code == 400 and secret not in r.text


@pytest.mark.parametrize("path", ENDPOINTS)
async def test_deeply_nested_and_non_object_bodies(http, path) -> None:
    key = "input" if path == "/v1/responses" else "messages"
    item = (
        f'{{"role":"user","content":{NESTED_JSON}}}' if key == "messages" else f'{{"content":{NESTED_JSON}}}'
    )
    nested = json.dumps({**VALID[path], key: 0}).replace(f'"{key}": 0', f'"{key}": [{item}]')
    for payload in (nested, "[1, 2]", '"str"', "5", "null"):
        r = await asyncio.wait_for(
            http.post(path, content=payload, headers={"content-type": "application/json"}), 20
        )
        assert r.status_code < 500, (path, r.status_code)
    r = await http.post(path, content=b"[" * 200_000, headers={"content-type": "application/json"})
    assert r.status_code == 400


@pytest.mark.parametrize("path", ENDPOINTS)
async def test_fuzz_never_5xx(http, path) -> None:
    rng = random.Random(1234)
    for _ in range(300):
        body = copy.deepcopy(VALID[path])
        for _ in range(rng.randint(1, 3)):
            body[rng.choice(FIELDS)] = copy.deepcopy(rng.choice(WEIRD))
        if rng.random() < 0.3:
            # weird things inside a message / input item
            inner = {"role": rng.choice(["user", "assistant", "tool", 5, None]), "content": rng.choice(WEIRD)}
            inner[rng.choice(["tool_calls", "tool_call_id", "name"])] = rng.choice(WEIRD)
            key = "input" if path == "/v1/responses" else "messages"
            body[key] = [inner]
        r = await asyncio.wait_for(http.post(path, json=body), 20)
        assert r.status_code < 500, (path, r.status_code, body, r.text)
