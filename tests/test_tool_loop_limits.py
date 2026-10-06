"""Tool-loop continuity limits: unknown ids are logged once; a failed holder reroutes with history intact."""

from __future__ import annotations

import httpx
import pytest

from flux_os import Catalog, FluxOS
from flux_os.server import create_app

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


@pytest.fixture
def http(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux")


def _turn2(first_msg: dict, call_id: str) -> dict:
    return {
        "model": "auto",
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            first_msg,
            {"role": "tool", "tool_call_id": call_id, "content": "18C"},
        ],
    }


async def test_failed_holder_reroutes_with_history_intact(http, fake) -> None:
    fake.tool_call = True
    first = await http.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
        },
    )
    holder = first.headers["x-flux-model"]
    msg = first.json()["choices"][0]["message"]
    body = _turn2(msg, msg["tool_calls"][0]["id"])
    fake.calls.clear()
    held = await http.post("/v1/chat/completions", json=body)
    assert held.headers["x-flux-model"] == holder and held.headers["x-flux-rerouted"] == "false"

    upstream = next(m.upstream_id for m in http._transport.app.state.flux.catalog.models if m.id == holder)  # type: ignore[attr-defined]
    fake.fail[upstream] = 503
    fake.calls.clear()
    r = await http.post("/v1/chat/completions", json=body)
    assert r.status_code == 200 and r.headers["x-flux-rerouted"] == "true"
    assert r.headers["x-flux-model"] != holder
    sent = [c for c in fake.calls if c["style"] == "openai"]
    assert len(sent) >= 2
    # the failing attempt and the fallback saw the same conversation: nothing lost or rewritten
    assert sent[0]["body"]["messages"] == sent[-1]["body"]["messages"]
    assert [m["role"] for m in sent[-1]["body"]["messages"]] == ["user", "assistant", "tool"]
    assert sent[-1]["body"]["messages"][1]["tool_calls"][0]["id"] == msg["tool_calls"][0]["id"]


async def test_unknown_tool_call_id_logged_once(http, fake, capsys) -> None:
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_never_seen", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }
    for _ in range(3):
        r = await http.post("/v1/chat/completions", json=_turn2(msg, "call_never_seen"))
        assert r.status_code == 200
    err = capsys.readouterr().err
    assert err.count("unknown tool_call_id") == 1 and "call_never_seen" in err


async def test_restart_loses_continuity_but_works(keys, fake, capsys) -> None:
    """A fresh process (restart) routes the tool result fresh and logs it; it does not error."""
    fake.tool_call = True
    mk = lambda: create_app(  # noqa: E731
        FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(fake.handler)), api_key=""
    )
    a = httpx.AsyncClient(transport=httpx.ASGITransport(app=mk()), base_url="http://flux")
    first = await a.post(
        "/v1/chat/completions",
        json={"model": "auto", "tools": TOOLS, "messages": [{"role": "user", "content": "weather?"}]},
    )
    msg = first.json()["choices"][0]["message"]
    b = httpx.AsyncClient(transport=httpx.ASGITransport(app=mk()), base_url="http://flux")
    r = await b.post("/v1/chat/completions", json=_turn2(msg, msg["tool_calls"][0]["id"]))
    assert r.status_code == 200
    assert "tool_call_id" in capsys.readouterr().err
