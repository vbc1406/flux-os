# Changelog

## Unreleased

### Changed
- README no longer says tool calls work "across every provider". It now states what the mocked tests cover and that no provider has been checked live.
- Added `tests/test_live_astra.py` (function call, streaming function call, tool loop against the real API). They skip without `OPENAI_API_KEY` and have not been run. OpenAI's model page lists Chat Completions and function calling for Astra, while a third-party relay doc says Chat Completions rejects `tools` for it. Until the live tests pass, `gpt-6-astra` stays `tools: false`. Its `function_calling` rating (0.93) is the model's general ability and has no effect on routing while tools are excluded.

### Fixed
- `gpt-6-astra` is no longer marked `tools: true`. OpenAI requires the Responses API for
  GPT-6 Astra tool calling, and flux-os only speaks Chat Completions to OpenAI, so tool
  requests routed to Astra would have failed upstream. Requests that include tools now skip
  Astra; requests without tools can still use it. Pinning `model="gpt-6-astra"` with tools is
  still honoured as-is and will hit that upstream limit until a Responses adapter exists.

## 0.2.0
Fixes from a routing evaluation (real providers, stub upstreams, 138 tests).

### Fixed
- Anthropic Messages path no longer breaks Gemini 3 tool loops: the `thought_signature` round-trips on `tool_use` blocks (streaming and not, parallel calls too) and is remembered server-side for clients that strip unknown fields. A "missing thought_signature" 400 is now reroutable.
- Type-invalid input (`"model": 5`, `"max_tokens": "abc"`, ...) returns 400, not 500, on every endpoint.
- Empty replies from reasoning models that spend all of `max_tokens` on hidden reasoning (`finish_reason: length`) now reroute, or return a clear error. Non-streaming only.
- Catalog: gpt-oss-120b reasoning rating 0.86, gpt-oss-20b 0.80 (the 120b leads on MMLU, GPQA, HLE and Tau-Bench in OpenAI's model card).

### Added
- Reroute observability: a stderr line per reroute, an `x-flux-attempts` header, `x-flux-*` headers on upstream errors.
- Quality-bar tolerance band, `FLUX_OS_QUALITY_TOLERANCE` (default 0.02; `0` restores the strict bar).
- Pinned catalog models whose context window the input exceeds now fail with a loud 400 (`FLUX_OS_ALLOW_OVERSIZE=1` opts out).
- Unknown tool_call_ids are logged once.

### Behaviour change to know about
- The 0.02 tolerance sends noticeably more traffic to the cheapest adequate model than 0.1.0 did. On a hard 50-prompt math/STEM set, `auto` scored 49/50 versus 40/50 for always-cheapest, with the reroute safety net rescuing many first picks. See the README for limits.

### Docs
- README examples are now checked by tests; notes on single-instance tool loops, mid-loop reroutes, and tier-gated models.
