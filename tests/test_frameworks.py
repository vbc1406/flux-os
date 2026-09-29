"""Real agent frameworks and wrappers pointed at a running flux-os server.

Each test is skipped when its framework isn't installed (``pip install -e ".[frameworks]"``).
The upstream providers are faked; everything between the framework and the
provider (HTTP, SSE, tool calls, format translation, routing) is real.
"""

from __future__ import annotations

import pytest


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and sunny in {city}"


async def test_openai_agents_sdk_chat_completions(live_url, fake) -> None:
    agents = pytest.importorskip("agents")
    from openai import AsyncOpenAI

    agents.set_tracing_disabled(True)
    fake.tool_call = True
    client = AsyncOpenAI(base_url=live_url + "/v1", api_key="unused")
    agent = agents.Agent(
        name="weather",
        instructions="Use tools to answer.",
        tools=[agents.function_tool(get_weather)],
        model=agents.OpenAIChatCompletionsModel(model="auto", openai_client=client),
    )
    result = await agents.Runner.run(agent, "What's the weather in Paris?")
    assert result.final_output.startswith("hello from")
    assert any(c["body"]["messages"][-1].get("role") == "tool" for c in fake.calls if c["style"] == "openai")


async def test_openai_agents_sdk_responses_api(live_url, fake) -> None:
    agents = pytest.importorskip("agents")
    from openai import AsyncOpenAI

    agents.set_tracing_disabled(True)
    fake.tool_call = True
    client = AsyncOpenAI(base_url=live_url + "/v1", api_key="unused")
    agent = agents.Agent(
        name="weather",
        instructions="Use tools to answer.",
        tools=[agents.function_tool(get_weather)],
        model=agents.OpenAIResponsesModel(model="auto", openai_client=client),
    )
    result = await agents.Runner.run(agent, "What's the weather in Paris?")
    assert result.final_output.startswith("hello from")

    streamed = agents.Runner.run_streamed(agent, "What's the weather in Paris?")
    async for _ in streamed.stream_events():
        pass
    assert streamed.final_output.startswith("hello from")


def test_langchain_chat_openai_tools_and_stream(live_url, fake) -> None:
    lc = pytest.importorskip("langchain_openai")
    from langchain_core.tools import tool

    llm = lc.ChatOpenAI(model="auto", base_url=live_url + "/v1", api_key="unused")
    assert llm.invoke("hi").content.startswith("hello from")
    assert "".join(c.content for c in llm.stream("hi")).startswith("hello from")
    fake.tool_call = True
    msg = llm.bind_tools([tool(get_weather)]).invoke("weather in Paris?")
    assert msg.tool_calls[0]["name"] == "get_weather" and msg.tool_calls[0]["args"] == {"city": "Paris"}


def test_langchain_responses_mode(live_url) -> None:
    lc = pytest.importorskip("langchain_openai")
    llm = lc.ChatOpenAI(model="auto", base_url=live_url + "/v1", api_key="unused", use_responses_api=True)
    assert llm.invoke("hi").text.startswith("hello from")


def test_langgraph_react_agent(live_url, fake) -> None:
    pytest.importorskip("langgraph")
    lc = pytest.importorskip("langchain_openai")
    import warnings

    from langchain_core.tools import tool

    fake.tool_call = True
    llm = lc.ChatOpenAI(model="auto", base_url=live_url + "/v1", api_key="unused")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from langgraph.prebuilt import create_react_agent

        graph = create_react_agent(llm, [tool(get_weather)])
    out = graph.invoke({"messages": [("user", "What's the weather in Paris?")]})
    kinds = [m.type for m in out["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]
    assert out["messages"][2].content == "18C and sunny in Paris"


def test_langchain_anthropic_against_messages_endpoint(live_url, fake) -> None:
    la = pytest.importorskip("langchain_anthropic")
    from langchain_core.tools import tool

    llm = la.ChatAnthropic(model="auto", base_url=live_url, api_key="unused", max_tokens=200)
    assert llm.invoke("hi").content.startswith("hello from")
    fake.tool_call = True
    msg = llm.bind_tools([tool(get_weather)]).invoke("weather in Paris?")
    assert msg.tool_calls[0]["args"] == {"city": "Paris"}
