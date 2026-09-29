"""The Anthropic SDK works too — flux-os serves /v1/messages and can route to any provider.

flux-os serve
python examples/anthropic_sdk.py
"""

from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="unused")
msg = client.messages.create(
    model="auto",
    max_tokens=300,
    messages=[{"role": "user", "content": "Write a haiku about routers."}],
)
print(msg.content[0].text)
