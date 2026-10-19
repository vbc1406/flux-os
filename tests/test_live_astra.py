"""Live checks for GPT-6 Astra function calling over Chat Completions.

These call the real OpenAI API through the flux-os server, pinned to ``gpt-6-astra``. They skip
unless OPENAI_API_KEY is set when pytest starts, and they cost real money. Run with:

    OPENAI_API_KEY=sk-... pytest tests/test_live_astra.py
"""

from __future__ import annotations

import os

import httpx
import pytest
from openai import AsyncOpenAI

from flux_os import Catalog, FluxOS
from flux_os.server import create_app

# The autouse clean_env fixture strips provider keys, so read the key before it runs.
LIVE_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-6-astra"

pytestmark = pytest.mark.skipif(not LIVE_KEY, reason="OPENAI_API_KEY not set; live Astra tests skipped")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
PROMPT = [{"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}]


@pytest.fixture
async def oai(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", LIVE_KEY)
    app = create_app(FluxOS(catalog=Catalog.load()), api_key="")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://flux", timeout=120
    ) as http:
        yield AsyncOpenAI(base_url="http://flux/v1", api_key="unused", http_client=http)


async def test_astra_function_call(oai: AsyncOpenAI) -> None:
    r = await oai.chat.completions.create(model=MODEL, messages=PROMPT, tools=TOOLS)
    calls = r.choices[0].message.tool_calls
    assert calls and calls[0].function.name == "get_weather"


async def test_astra_streaming_function_call(oai: AsyncOpenAI) -> None:
    stream = await oai.chat.completions.create(model=MODEL, messages=PROMPT, tools=TOOLS, stream=True)
    names, args = "", ""
    async for chunk in stream:
        for call in (chunk.choices[0].delta.tool_calls or []) if chunk.choices else []:
            names += call.function.name or ""
            args += call.function.arguments or ""
    assert names == "get_weather" and "Paris" in args


async def test_astra_tool_loop(oai: AsyncOpenAI) -> None:
    first = await oai.chat.completions.create(model=MODEL, messages=PROMPT, tools=TOOLS)
    call = first.choices[0].message.tool_calls[0]
    messages = PROMPT + [
        first.choices[0].message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "18C and sunny"},
    ]
    second = await oai.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    assert second.choices[0].message.content
