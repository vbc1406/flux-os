"""OpenAI Agents SDK with tools, routed by flux-os (Chat Completions or Responses mode).

pip install openai-agents
flux-os serve
python examples/openai_agents_sdk.py
"""

import asyncio

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

set_tracing_disabled(True)
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="unused")


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and sunny in {city}"


agent = Agent(
    name="assistant",
    instructions="Answer using tools when useful.",
    tools=[get_weather],
    # OpenAIResponsesModel(model="auto", openai_client=client) works as well.
    model=OpenAIChatCompletionsModel(model="auto", openai_client=client),
)

print(asyncio.run(Runner.run(agent, "What's the weather in Paris?")).final_output)
