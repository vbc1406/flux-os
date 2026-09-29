"""LangChain / LangGraph ReAct agent routed by flux-os.

pip install langchain-openai langgraph
flux-os serve
python examples/langgraph_agent.py
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"18C and sunny in {city}"


llm = ChatOpenAI(model="auto", base_url="http://localhost:8000/v1", api_key="unused")
agent = create_react_agent(llm, [get_weather])
result = agent.invoke({"messages": [("user", "What's the weather in Paris?")]})
print(result["messages"][-1].content)
