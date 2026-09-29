"""Any OpenAI client works: change base_url, set model="auto".

flux-os serve            # in another terminal
python examples/openai_sdk.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")  # or your FLUX_OS_API_KEY

raw = client.chat.completions.with_raw_response.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain backpropagation in two sentences."}],
)
resp = raw.parse()
print(resp.choices[0].message.content)
print("routed to:", raw.headers["x-flux-model"])
