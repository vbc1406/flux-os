<div align="center">

# flux-os

**Send every LLM request to the cheapest model that can actually handle it.**

A drop-in router for the OpenAI and Anthropic APIs. Works with agents and wrappers,
as a proxy or a Python library. No dashboard, no database, no setup.

[![CI](https://github.com/vbc1406/flux-os/actions/workflows/ci.yml/badge.svg)](https://github.com/vbc1406/flux-os/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

```text
"hi"                                          → gpt-oss-20b      (groq)     ~$0.00004
"Translate 'good morning' into Spanish"       → mistral-small-4  (mistral)  ~$0.0002
"Write a Python function that parses dates"   → gpt-oss-20b      (groq)     ~$0.0004
"Implement a lock-free concurrent hash map in
 Rust, thread-safe, no global locks, ..."     → o4-mini          (openai)   ~$0.0053
```

Most traffic doesn't need your most expensive model. flux-os reads each request (task type,
difficulty, tools, images, JSON mode, context size), sets a quality bar, and picks the
**cheapest model that clears it**. If that model fails (rate limit, outage, timeout), the request
is **rerouted** to the next-best model, on a different provider first.

- **Zero code changes.** Point your OpenAI or Anthropic client at flux-os and set `model="auto"`.
- **Agents work.** Tool calls, streaming tool calls, and multi-turn tool loops work across every provider, whichever API format your framework speaks.
- **Three API formats in, any provider out.** Chat Completions, the Responses API and Anthropic Messages, translated to OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, OpenRouter or Ollama. Point another OpenAI-compatible server at flux-os by setting `OPENAI_BASE_URL` (or `OLLAMA_BASE_URL` for an Ollama-compatible one) before starting `flux-os serve`.
- **Fast.** Pure-Python heuristics, no LLM in the decision path, well under 1 ms per decision.

## Quickstart

```bash
pip install "git+https://github.com/vbc1406/flux-os"

# See what it would do. No API keys needed for this:
flux-os route --all "Prove that there are infinitely many primes"
```

```text
model:       gpt-5.6-luna  (openai)
task:        reasoning   complexity 0.55   quality bar 0.84
est. cost:   $0.001203
reroute to:  gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-3.8-flash
```

Then set keys for the providers you use (any subset) and start the proxy:

```bash
export OPENAI_API_KEY=sk-...  ANTHROPIC_API_KEY=sk-ant-...  GROQ_API_KEY=gsk_...
flux-os serve                  # http://127.0.0.1:8000
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="auto",   # or flux-cheap / flux-fast / a concrete model id
    messages=[{"role": "user", "content": "Explain backpropagation in two sentences."}],
)
print(resp.model)   # the model that served it
```

Prefer no server? Use the library in-process. It has the same shape as the OpenAI SDK:

```python
from flux_os import FluxOS

flux = FluxOS()
resp = flux.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
print(resp.choices[0].message.content, resp.flux["model"])
```

## Every way to run it

| How | Command |
|---|---|
| pip / CLI | `pip install "git+https://github.com/vbc1406/flux-os"` then `flux-os serve` |
| uv, no install | `uvx --from git+https://github.com/vbc1406/flux-os flux-os serve` |
| Docker | `docker build -t flux-os . && docker run -p 127.0.0.1:8000:8000 -e OPENAI_API_KEY -e FLUX_OS_API_KEY flux-os` |
| Docker Compose | `docker compose up -d` (reads keys from your environment) |
| Python module | `python -m flux_os serve` |
| Python library | `from flux_os import FluxOS, route`: sync, async and streaming |
| Decision only | `flux_os.route(messages)` or `POST /v1/route`: flux-os picks, you call |

## Works with

Each row below is exercised by the test suite or was run end to end against a live
flux-os server. Provider responses were faked; the HTTP, SSE, tool calling, format
translation and routing are the real code paths.

| Client | Setup | Verified |
|---|---|---|
| OpenAI Python SDK | `OpenAI(base_url="http://localhost:8000/v1")` | chat, streaming, tools, Responses API |
| Anthropic Python SDK | `Anthropic(base_url="http://localhost:8000")` | messages, streaming, tools, count_tokens |
| OpenAI Agents SDK | `OpenAIChatCompletionsModel` **or** `OpenAIResponsesModel` with `AsyncOpenAI(base_url=...)` | multi-step tool loop, streaming |
| LangChain `ChatOpenAI` | `base_url=...` (Responses mode works too) | invoke, stream, `bind_tools` |
| LangGraph | `create_react_agent(ChatOpenAI(base_url=...), tools)` | full ReAct loop |
| LangChain `ChatAnthropic` | `base_url="http://localhost:8000"` | invoke, `bind_tools` |
| Vercel AI SDK | `createOpenAICompatible`, `createOpenAI` (Responses), `.chat()`, `createAnthropic` | `generateText` + `streamText` with multi-step tools |
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8000` with `FLUX_OS_ROUTE_ALL=1` on the server | prompts and tool turns |
| curl / any HTTP client | see [`examples/curl.sh`](examples/curl.sh) | yes |

Anything else that speaks the OpenAI or Anthropic API (n8n, CrewAI, LlamaIndex, Continue, Cline,
Open WebUI, and so on) should work the same way, but hasn't been tested yet. If you try one,
please open an issue with the result.

Runnable examples are in [`examples/`](examples/).

## How routing works

```
request ─► classify ─► filter ─► quality bar ─► rank ─► call ─┬─► response
            task        keys      0.72 + 0.22     cheapest     │
            complexity  tools     × complexity    that clears  └─ failed? reroute to next
            tools/img   vision    (≥0.80 for      the bar         (other providers first)
            JSON, size  context   tool steps)
```

1. **Classify.** Task type (`code_generation`, `code_review`, `reasoning`, `analysis`, `creative_writing`, `summarization`, `extraction`, `translation`, `classification`, `simple_qa`, `conversation`, ...) and a 0–1 complexity score from signals like constraints, math, domain difficulty, multi-part questions, and prompt length.
2. **Filter.** Only models whose provider has a key, and that support what the request needs: tools, images, JSON mode, and a large enough context window.
3. **Set a bar.** Harder requests need higher-rated models. Tool-calling agent steps never go below 0.80.
4. **Rank.** Models that clear the bar are sorted by estimated cost for this request, and the rest follow by quality. That ordered list is also the reroute order.
5. **Call, and reroute on failure.** Rate limits, 5xx errors, timeouts, auth errors, unknown models and capability errors move the request to the next candidate, trying other providers before the one that failed. Plain bad requests (a 400) are returned as they are, and streams are only rerouted before the first token.

**Agent tool loops stay on one model, while that model is healthy.** When a tool result comes back,
flux-os sends it to the model that made the tool call, so an agent doesn't change models in the
middle of a step. Two limits to know about:

- The tool-call-id → model map lives in memory of **one flux-os process**. After a restart, behind
  several replicas without sticky routing, or for tool-call ids flux-os never issued, the tool result
  is routed fresh (flux-os logs this once per id to stderr). Run a single instance for agent traffic.
- If the model holding the loop fails (429, 5xx, ...), flux-os reroutes **mid-loop** to the next
  candidate, possibly another provider, and sends it the full message history unchanged. That is by
  design: an answer from a different model beats an error. Pin a model and leave rerouting off
  (the default for pinned models) if you need a loop to stay put or fail.

### Modes

| `model=` | Behaviour |
|---|---|
| `auto` | Cheapest model that clears the quality bar (default) |
| `flux-cheap` | Lowers the bar slightly and saves more |
| `flux-fast` | Lowest-latency model that clears the bar |
| `gpt-5`, `claude-sonnet-5`, ... | Pinned: that exact model, never swapped (unless you allow rerouting) |
| `ollama/llama3.2`, `groq/<id>`, ... | Pinned to any model on a known provider, even if it isn't in the catalog |

You can also set the mode per request with the `X-Flux-Mode: cheap|fast|auto` header, or
for the whole server with `FLUX_OS_DEFAULT_MODE`.

### Response headers

| Header | Meaning |
|---|---|
| `x-flux-model` / `x-flux-provider` | What actually served the request |
| `x-flux-rerouted` | `true` if a fallback served it |
| `x-flux-task` / `x-flux-complexity` / `x-flux-mode` | Classification |

## Endpoints

| Endpoint | Format |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, JSON mode, images |
| `POST /v1/responses` | OpenAI Responses API: streaming, function tools, `previous_response_id` |
| `POST /v1/messages` | Anthropic Messages: streaming, tools, images |
| `POST /v1/messages/count_tokens` | Anthropic token estimate |
| `POST /v1/route` | The routing decision as JSON, with no model call |
| `GET /v1/models` | Directives plus every model you have a key for |
| `GET /health` | Liveness and available providers |

## Configuration

Everything works with nothing but provider keys. For more control:

| Env var | Default | |
|---|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `GROQ_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` | | Providers to route across. Unset providers are skipped. |
| `FLUX_OS_API_KEY` | none | Clients must send it as `Authorization: Bearer ...` or `x-api-key`. **Set it whenever the server is reachable by others.** |
| `FLUX_OS_HOST` / `FLUX_OS_PORT` | `127.0.0.1` / `8000` | Bind address (`PORT` is honoured too) |
| `FLUX_OS_DEFAULT_MODE` | `auto` | `auto`, `cheap` or `fast` |
| `FLUX_OS_ROUTE_ALL` | off | Treat every model name as `auto`, for tools that hard-code a model (Claude Code) |
| `FLUX_OS_REROUTE_PINNED` | off | Let pinned models fall back too (per request: `X-Flux-Reroute: true`) |
| `FLUX_OS_ALLOW_OVERSIZE` | off | Forward a pinned catalog model's request even when the input exceeds its context window (default: loud 400, nothing sent) |
| `FLUX_OS_MAX_ATTEMPTS` | `3` | Models to try before giving up |
| `FLUX_OS_QUALITY_OFFSET` | `0` | Shift every quality bar, e.g. `0.05` for stricter or `-0.05` for cheaper |
| `FLUX_OS_ONLY_MODELS` / `FLUX_OS_DISABLE` | | Comma-separated model ids or `provider:<name>` |
| `FLUX_OS_TIMEOUT` | `120` | Upstream timeout in seconds |
| `FLUX_OS_MAX_BODY_BYTES` | `10485760` (10 MiB) | Max request body size; larger requests get a 413 |
| `FLUX_OS_ALLOW_UNAUTHENTICATED` | off | Let `flux-os serve` start on a non-loopback host without `FLUX_OS_API_KEY` |
| `FLUX_OS_DEBUG_ERRORS` | off | Return full upstream error detail in HTTP responses instead of a generic message |

### Claude Code

```bash
FLUX_OS_ROUTE_ALL=1 flux-os serve
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=unused claude
```

Claude Code always asks for a specific Claude model, so `FLUX_OS_ROUTE_ALL=1` lets flux-os pick
instead. The catalog decides whether a coding turn needs a frontier model.

## Model catalog

27 current models from OpenAI, Anthropic, Google, Groq and Mistral ship in
[`flux_os/models.json`](flux_os/models.json), with prices (USD per 1M tokens), context windows,
capabilities and per-task quality ratings. Run `flux-os models --all` to list them. The quality
ratings are editorial estimates built from public benchmarks, not measurements of your
workload.

## Security

- **Set `FLUX_OS_API_KEY`** and require it on every request. `flux-os serve` refuses to start on a
  non-loopback host without it (override with `FLUX_OS_ALLOW_UNAUTHENTICATED=1` if you understand
  the risk).
- **Bind to localhost** (`FLUX_OS_HOST=127.0.0.1`, the default) unless you have a reason not to.
- **Put a TLS reverse proxy in front** (nginx, Caddy, your cloud load balancer, ...) before exposing
  flux-os beyond your own machine. flux-os itself speaks plain HTTP.
- flux-os has **no rate limiting, budgets or spend caps**. Set spend limits at each provider
  (OpenAI, Anthropic, Google, ...) rather than relying on flux-os to stop runaway usage.

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Data handling

- Prompts and responses go directly to the providers you configure, under those providers' own
  terms — flux-os does not add its own processing or retention.
- flux-os writes nothing to disk and has no telemetry.
- Recent Responses API turns (for `previous_response_id`) are held in memory only, bounded by
  count and total size, and are lost on restart.
- The operator running flux-os is responsible for the provider terms and privacy law that apply to
  their own users' data.

## Scope

flux-os does routing and rerouting, and nothing else. It has **no** dashboard, spend tracking,
budgets, caching, database or telemetry, and it stores nothing on disk. The only in-memory state
is a small LRU cache of tool-call ids (for agent loop continuity) and recent Responses API turns
(for `previous_response_id`).

Known limits:
- The Responses API supports function tools only. Built-in tools like `web_search` and `file_search` are ignored.
- Anthropic server tools (`web_search_*`, `bash_*`, and similar) aren't forwarded. Client tools work.
- Extended-thinking blocks aren't carried across providers.
- Classification is heuristic. It's fast and predictable, but not an LLM judge.

## Development

```bash
git clone https://github.com/vbc1406/flux-os && cd flux-os
pip install -e ".[dev,frameworks]"
pytest -q          # routing, rerouting, 3 API formats, real SDKs and agent frameworks
ruff check . && ruff format --check .
```

Issues and pull requests are welcome, especially compatibility reports for tools not in the table
above, and quality ratings backed by data. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Disclaimer

flux-os is an independent, unofficial project and is not affiliated with, endorsed by, or
sponsored by OpenAI, Anthropic, Google, Groq, Mistral, DeepSeek, OpenRouter, Ollama, LangChain or
Vercel. Their names are used only to describe API and provider compatibility. Prices and quality
ratings in this repository are estimates as of 2026-09-27 and must be checked against each
provider's current, official pricing before you rely on them.

## License

MIT
