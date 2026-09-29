"""In-process client: calling providers, translation, and rerouting on failure."""

from __future__ import annotations

import json

import pytest

from flux_os import FluxOS, UpstreamError

HARD = "Implement a production-grade distributed rate limiter in Go with Redis, ensure it handles edge cases."
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


def req(text: str = "hello", **kw):
    return {"model": "auto", "messages": [{"role": "user", "content": text}], **kw}


async def test_routes_and_calls(flux: FluxOS, fake) -> None:
    resp, info = await flux.dispatch(req())
    assert resp["choices"][0]["message"]["content"].startswith("hello from")
    assert fake.models_called == [info.model.upstream_id]
    assert not info.rerouted


async def test_reroutes_on_failure_to_another_provider(flux: FluxOS, fake) -> None:
    decision = flux.route([{"role": "user", "content": HARD}])
    first = decision.model
    fake.fail[first.upstream_id] = 503
    resp, info = await flux.dispatch(req(HARD))
    assert info.rerouted
    assert info.model.provider != first.provider
    assert info.attempts[0]["status"] == 503 and info.attempts[-1]["status"] == 200
    assert resp["model"] == info.model.id


async def test_rate_limit_and_auth_errors_reroute(flux: FluxOS, fake) -> None:
    first = flux.route([{"role": "user", "content": "hello"}]).model
    for status in (429, 401, 404, 500):
        fake.fail = {first.upstream_id: status}
        _, info = await flux.dispatch(req())
        assert info.rerouted, status


async def test_bad_request_does_not_reroute(flux: FluxOS, fake) -> None:
    first = flux.route([{"role": "user", "content": "hello"}]).model
    fake.fail[first.upstream_id] = 400
    with pytest.raises(UpstreamError) as err:
        await flux.dispatch(req())
    assert len(err.value.attempts) == 1 and err.value.status_code == 400


async def test_all_fail_reports_every_attempt(flux: FluxOS, fake) -> None:
    for m in flux.catalog.models:
        fake.fail[m.upstream_id] = 500
    with pytest.raises(UpstreamError) as err:
        await flux.dispatch(req())
    assert len(err.value.attempts) == flux.max_attempts == 3


async def test_pinned_does_not_reroute_unless_asked(flux: FluxOS, fake) -> None:
    fake.fail["gpt-5"] = 500
    with pytest.raises(UpstreamError):
        await flux.dispatch(req(model="gpt-5"))
    _, info = await flux.dispatch(req(model="gpt-5"), reroute=True)
    assert info.model.id != "gpt-5" and info.attempts[0]["model"] == "gpt-5"


async def test_anthropic_translation(flux: FluxOS, fake) -> None:
    body = req("be brief", model="claude-sonnet-5", max_tokens=50, temperature=0.2, stop="END")
    body["messages"].insert(0, {"role": "system", "content": "You are terse."})
    resp, _ = await flux.dispatch(body)
    sent = fake.calls[-1]
    assert sent["style"] == "anthropic"
    assert sent["headers"]["x-api-key"] == "test-key"
    assert sent["body"]["system"] == "You are terse."
    assert sent["body"]["max_tokens"] == 50 and sent["body"]["stop_sequences"] == ["END"]
    assert resp["choices"][0]["message"]["content"] == "hello from claude-sonnet-5"
    assert resp["usage"]["prompt_tokens"] == 12


async def test_anthropic_tool_round_trip(flux: FluxOS, fake) -> None:
    fake.tool_call = True
    resp, _ = await flux.dispatch(
        req("weather in Paris?", model="claude-sonnet-5", tools=TOOLS, tool_choice="required")
    )
    call = resp["choices"][0]["message"]["tool_calls"][0]
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    sent = fake.calls[-1]["body"]
    assert sent["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    assert sent["tool_choice"] == {"type": "any"}

    followup = req("weather in Paris?", model="claude-sonnet-5", tools=TOOLS)
    followup["messages"] += [
        resp["choices"][0]["message"],
        {"role": "tool", "tool_call_id": call["id"], "content": "18C and sunny"},
    ]
    resp2, _ = await flux.dispatch(followup)
    sent = fake.calls[-1]["body"]["messages"]
    assert sent[1]["content"][0]["type"] == "tool_use"
    assert sent[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": call["id"],
        "content": "18C and sunny",
    }
    assert resp2["choices"][0]["message"]["content"]


async def test_openai_param_quirks(flux: FluxOS, fake) -> None:
    await flux.dispatch(req(model="gpt-5", max_tokens=100, temperature=0.3))
    sent = fake.calls[-1]["body"]
    assert sent["max_completion_tokens"] == 100 and "max_tokens" not in sent
    assert "temperature" not in sent  # gpt-5 rejects non-default temperature
    await flux.dispatch(req(model="mistral-large-3", seed=7, user="u1", reasoning_effort="low"))
    sent = fake.calls[-1]["body"]
    assert sent["random_seed"] == 7 and "user" not in sent and "reasoning_effort" not in sent
    assert sent["model"] == "mistral-large-2512"


async def test_streaming_openai_and_anthropic(flux: FluxOS, fake) -> None:
    for model in ("gpt-5-mini", "claude-haiku-4-5-20251001"):
        stream, info = await flux.dispatch_stream(req(model=model, stream=True))
        text = ""
        async for chunk in stream:
            for c in chunk["choices"]:
                text += c["delta"].get("content") or ""
        assert text.strip() == f"hello from {info.model.upstream_id}"


async def test_streaming_reroutes_before_first_chunk(flux: FluxOS, fake) -> None:
    first = flux.route([{"role": "user", "content": "hello"}]).model
    fake.fail[first.upstream_id] = 503
    stream, info = await flux.dispatch_stream(req(stream=True))
    chunks = [c async for c in stream]
    assert info.rerouted and chunks


async def test_streaming_tool_calls_from_anthropic(flux: FluxOS, fake) -> None:
    fake.tool_call = True
    stream, _ = await flux.dispatch_stream(req("weather?", model="claude-sonnet-5", tools=TOOLS, stream=True))
    args, name = "", None
    async for chunk in stream:
        for c in chunk["choices"]:
            for tc in c["delta"].get("tool_calls") or []:
                name = name or tc.get("function", {}).get("name")
                args += tc.get("function", {}).get("arguments") or ""
    assert name == "get_weather" and json.loads(args) == {"city": "Paris"}


def test_sync_api_and_streaming(flux: FluxOS) -> None:
    resp = flux.chat.completions.create(**req())
    assert resp.choices[0].message.content.startswith("hello from")
    assert resp.flux["model"]
    stream = flux.chat.completions.create(**req(stream=True))
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text.startswith("hello from")


async def test_route_all_overrides_hardcoded_model(keys, fake) -> None:
    import httpx

    flux = FluxOS(route_all=True, transport=httpx.MockTransport(fake.handler))
    _, info = await flux.dispatch(req("hi", model="claude-opus-5"))
    assert not info.decision.pinned
    assert info.model.id != "claude-opus-5"
