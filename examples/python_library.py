"""Use flux-os in-process — no server. Same API shape as the OpenAI SDK.

export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...   # any subset of providers
python examples/python_library.py
"""

from flux_os import FluxOS, route

# 1) Just ask which model a request should go to (no model call is made).
#    Preview without any keys from the shell: `flux-os route --all "your prompt"`
decision = route("Prove that the square root of 2 is irrational")
print(decision.model.id)

# 2) Route and call. Fails over to the next-best model on 429s, 5xx, timeouts.
flux = FluxOS()
resp = flux.chat.completions.create(
    model="auto",  # or flux-cheap / flux-fast / a concrete model id
    messages=[{"role": "user", "content": "Give me three names for a coffee shop."}],
)
print(resp.choices[0].message.content)
print("served by:", resp.flux["model"], "| rerouted:", resp.flux["rerouted"])

# 3) Streaming
for chunk in flux.chat.completions.create(
    model="auto", messages=[{"role": "user", "content": "Count to five."}], stream=True
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
