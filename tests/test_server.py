"""The HTTP proxy, exercised with the real OpenAI and Anthropic SDKs (wrapper compatibility)."""

from __future__ import annotations

import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from flux_os.server import _History, create_app

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


@pytest.fixture
def http(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux")


@pytest.fixture
def oai(http) -> AsyncOpenAI:
    return AsyncOpenAI(base_url="http://flux/v1", api_key="unused", http_client=http)


@pytest.fixture
def claude(live_url) -> AsyncAnthropic:
    return AsyncAnthropic(base_url=live_url, api_key="unused")


async def test_health_and_models(http) -> None:
    health = (await http.get("/health")).json()
    assert set(health) == {"status", "version"} and health["status"] == "ok"
    root = (await http.get("/")).json()
    assert set(root) == {"status", "version"}
    ids = [m["id"] for m in (await http.get("/v1/models")).json()["data"]]
    assert ids[:3] == ["auto", "flux-cheap", "flux-fast"]


async def test_body_limit(http, monkeypatch: pytest.MonkeyPatch) -> None:
    from flux_os.server import create_app

    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}

    monkeypatch.setenv("FLUX_OS_MAX_BODY_BYTES", "10")
    small_app = create_app(api_key="")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=small_app), base_url="http://flux") as c:
        r = await c.post("/v1/chat/completions", json=body)
        assert r.status_code == 413

        async def gen():
            yield json.dumps(body).encode()

        r = await c.post("/v1/chat/completions", content=gen())
        assert r.status_code == 413


async def test_openai_sdk_chat(oai: AsyncOpenAI) -> None:
    raw = await oai.chat.completions.with_raw_response.create(
        model="auto", messages=[{"role": "user", "content": "hi"}]
    )
    resp = raw.parse()
    assert resp.choices[0].message.content.startswith("hello from")
    assert raw.headers["x-flux-model"] == resp.model
    assert raw.headers["x-flux-task"] == "conversation"


def test_openai_sync_sdk_over_real_socket(live_url) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=live_url + "/v1", api_key="unused")
    resp = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    assert resp.choices[0].message.content.startswith("hello from")
    stream = client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    assert "".join(c.choices[0].delta.content or "" for c in stream if c.choices).startswith("hello")


async def test_openai_sdk_streaming(oai: AsyncOpenAI) -> None:
    stream = await oai.chat.completions.create(
        model="flux-cheap", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    text = "".join([c.choices[0].delta.content or "" async for c in stream if c.choices])
    assert text.startswith("hello from")


async def test_openai_sdk_agent_tool_loop(oai: AsyncOpenAI, fake) -> None:
    fake.tool_call = True
    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    first = await oai.chat.completions.create(model="auto", messages=messages, tools=TOOLS)
    call = first.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    messages += [
        first.choices[0].message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "18C"},
    ]
    second = await oai.chat.completions.create(model="auto", messages=messages, tools=TOOLS)
    assert second.choices[0].message.content
    assert second.model == first.model  # tool result went back to the model that asked for it


async def test_mode_and_reroute_headers(http, fake, flux) -> None:
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    r = await http.post("/v1/chat/completions", json=body, headers={"x-flux-mode": "fast"})
    assert r.headers["x-flux-mode"] == "fast"
    fake.fail["gpt-5"] = 500
    pinned = {**body, "model": "gpt-5"}
    assert (await http.post("/v1/chat/completions", json=pinned)).status_code == 500
    r = await http.post("/v1/chat/completions", json=pinned, headers={"x-flux-reroute": "true"})
    assert r.status_code == 200 and r.headers["x-flux-rerouted"] == "true"


async def test_route_endpoint(http) -> None:
    r = await http.post(
        "/v1/route", json={"messages": [{"role": "user", "content": "Prove Fermat's little theorem"}]}
    )
    d = r.json()
    assert d["analysis"]["task"] == "reasoning" and d["model"] and d["fallbacks"]


async def test_errors_are_openai_shaped(http) -> None:
    r = await http.post(
        "/v1/chat/completions", json={"model": "nope-model", "messages": [{"role": "user", "content": "x"}]}
    )
    assert r.status_code == 400 and "unknown model" in r.json()["error"]["message"]
    r = await http.post("/v1/chat/completions", content=b"not json")
    assert r.status_code == 400


async def test_upstream_error_hygiene(http, fake, flux, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.fail["gpt-5"] = 500
    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    r = await http.post("/v1/chat/completions", json=body)
    assert r.status_code == 500
    err = r.json()["error"]
    assert err["message"] == "upstream request failed"
    assert "attempts" not in err
    assert "simulated 500" not in json.dumps(err)

    monkeypatch.setenv("FLUX_OS_DEBUG_ERRORS", "1")
    debug_app = create_app(flux, api_key="")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=debug_app), base_url="http://flux") as c:
        r = await c.post("/v1/chat/completions", json=body)
        assert r.status_code == 500
        err = r.json()["error"]
        assert "simulated 500" in err["message"]
        assert err["attempts"]


async def test_api_key_auth(flux) -> None:
    app = create_app(flux, api_key="secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux") as c:
        body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
        assert (await c.post("/v1/chat/completions", json=body)).status_code == 401
        ok = await c.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer secret"})
        assert ok.status_code == 200
        ok = await c.post(
            "/v1/messages",
            json={"model": "auto", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "secret"},
        )
        assert ok.status_code == 200
        assert (await c.get("/health")).status_code == 200


# ── Anthropic Messages API (Claude Code, Anthropic SDKs) ──────────────────────


async def test_anthropic_sdk_messages(claude: AsyncAnthropic) -> None:
    msg = await claude.messages.create(
        model="auto",
        max_tokens=100,
        system="be brief",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert msg.content[0].type == "text" and msg.content[0].text.startswith("hello from")
    assert msg.stop_reason == "end_turn"


async def test_anthropic_sdk_streaming(claude: AsyncAnthropic) -> None:
    async with claude.messages.stream(
        model="auto", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        text = "".join([t async for t in stream.text_stream])
        final = await stream.get_final_message()
    assert text.startswith("hello from") and final.stop_reason == "end_turn"


async def test_anthropic_sdk_tools_non_claude_upstream(claude: AsyncAnthropic, fake) -> None:
    fake.tool_call = True
    tools = [
        {
            "name": "get_weather",
            "description": "weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    msg = await claude.messages.create(
        model="gpt-5-mini",
        max_tokens=200,
        tools=tools,
        messages=[{"role": "user", "content": "weather in Paris?"}],
    )
    tool_use = next(b for b in msg.content if b.type == "tool_use")
    assert tool_use.input == {"city": "Paris"} and msg.stop_reason == "tool_use"
    follow = await claude.messages.create(
        model="gpt-5-mini",
        max_tokens=200,
        tools=tools,
        messages=[
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": [b.model_dump() for b in msg.content]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "18C"}],
            },
        ],
    )
    assert follow.content[0].text
    sent = fake.calls[-1]["body"]["messages"]
    assert sent[-1] == {"role": "tool", "tool_call_id": tool_use.id, "content": "18C"}


async def test_anthropic_sdk_streaming_tools(claude: AsyncAnthropic, fake) -> None:
    fake.tool_call = True
    tools = [{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}]
    async with claude.messages.stream(
        model="gpt-5-mini", max_tokens=200, tools=tools, messages=[{"role": "user", "content": "weather?"}]
    ) as stream:
        final = await stream.get_final_message()
    assert final.content[0].type == "tool_use" and final.content[0].input == {"city": "Paris"}


async def test_count_tokens(claude: AsyncAnthropic) -> None:
    r = await claude.messages.count_tokens(
        model="auto", messages=[{"role": "user", "content": "hello world"}]
    )
    assert r.input_tokens > 0


# ── OpenAI Responses API (Agents SDK default, n8n, newer OpenAI code) ─────────


def test_history_bounded_by_bytes() -> None:
    history = _History(size=100, max_bytes=1000)
    big = [{"role": "user", "content": "x" * 400}]
    history.put("a", big)
    history.put("b", big)
    history.put("c", big)
    assert history.total_bytes <= 1000
    assert history.get("a") is None  # evicted to stay under the byte cap
    assert history.get("c") is not None


async def test_responses_api(oai: AsyncOpenAI) -> None:
    r = await oai.responses.create(model="auto", instructions="be brief", input="hi")
    assert r.output_text.startswith("hello from") and r.status == "completed"
    follow = await oai.responses.create(model="auto", input="and again", previous_response_id=r.id)
    assert follow.output_text


async def test_responses_api_streaming(oai: AsyncOpenAI) -> None:
    stream = await oai.responses.create(model="auto", input="hi", stream=True)
    kinds, text = [], ""
    async for ev in stream:
        kinds.append(ev.type)
        if ev.type == "response.output_text.delta":
            text += ev.delta
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert text.startswith("hello from")


async def test_responses_api_function_calls(oai: AsyncOpenAI, fake) -> None:
    fake.tool_call = True
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    r = await oai.responses.create(model="auto", input="weather in Paris?", tools=tools)
    call = next(o for o in r.output if o.type == "function_call")
    assert json.loads(call.arguments) == {"city": "Paris"}
    r2 = await oai.responses.create(
        model="auto",
        tools=tools,
        input=[
            {"role": "user", "content": "weather in Paris?"},
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            },
            {"type": "function_call_output", "call_id": call.call_id, "output": "18C"},
        ],
    )
    assert r2.output_text

    stream = await oai.responses.create(model="auto", input="weather?", tools=tools, stream=True)
    done = [ev async for ev in stream if ev.type == "response.function_call_arguments.done"]
    assert json.loads(done[0].arguments) == {"city": "Paris"}
