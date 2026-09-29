"""Decision-only mode: flux-os picks the model, you make the call with whatever SDK you like.

Useful for agent frameworks that manage their own provider clients.
"""

from flux_os import FluxOS

flux = FluxOS()
messages = [{"role": "user", "content": "Refactor this function to be thread-safe: ..."}]
d = flux.route(messages, tools=None)

print("provider:", d.model.provider)  # e.g. "anthropic"
print("model:   ", d.model.upstream_id)  # the id to send to that provider
print("if it fails, try:", [m.id for m in d.candidates[1:4]])

# Over HTTP the same thing is:  POST /v1/route  {"messages": [...], "tools": [...]}
