"""Pinned catalog models with a too-small context window fail loudly; auto picks a big one."""

from __future__ import annotations

import httpx
import pytest

from flux_os import Catalog, FluxOS
from flux_os.server import create_app

BIG = "word " * 120_000  # ~150k tokens: over gpt-4o-mini's 128k window


def _body(model: str) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "Summarize:\n" + BIG}]}


@pytest.fixture
def http(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux")


async def test_pinned_small_context_model_fails_loudly(http, fake) -> None:
    r = await http.post("/v1/chat/completions", json=_body("gpt-4o-mini"))
    assert r.status_code == 400
    msg = r.json()["error"]["message"]
    assert "gpt-4o-mini" in msg and "128000" in msg and "nothing was sent" in msg
    assert fake.calls == []  # nothing forwarded, so nothing truncated


async def test_pinned_small_context_fails_on_every_endpoint(http, fake) -> None:
    text = "Summarize:\n" + BIG
    r = await http.post(
        "/v1/messages",
        json={"model": "gpt-4o-mini", "max_tokens": 10, "messages": [{"role": "user", "content": text}]},
    )
    assert r.status_code == 400
    r = await http.post("/v1/responses", json={"model": "gpt-4o-mini", "input": text})
    assert r.status_code == 400
    assert fake.calls == []


async def test_pinned_fits_is_forwarded(http, fake) -> None:
    r = await http.post(
        "/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200


async def test_auto_picks_large_context_and_forwards_byte_identical(http, fake) -> None:
    body = _body("auto")
    r = await http.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    d = (await http.post("/v1/route", json=body)).json()
    assert d["model"] != "gpt-4o-mini"
    sent = fake.calls[-1]["body"]["messages"]
    assert sent == body["messages"]


async def test_opt_out_env_flag_forwards(keys, fake, monkeypatch) -> None:
    monkeypatch.setenv("FLUX_OS_ALLOW_OVERSIZE", "1")
    app = create_app(FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(fake.handler)), api_key="")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux") as c:
        r = await c.post("/v1/chat/completions", json=_body("gpt-4o-mini"))
    assert r.status_code == 200 and fake.calls[-1]["body"]["messages"] == _body("x")["messages"]
