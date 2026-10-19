# Changelog

## Unreleased

### Fixed
- `gpt-6-astra` is no longer marked `tools: true`. OpenAI requires the Responses API for
  GPT-6 Astra tool calling, and flux-os only speaks Chat Completions to OpenAI, so tool
  requests routed to Astra would have failed upstream. Requests that include tools now skip
  Astra; requests without tools can still use it. Pinning `model="gpt-6-astra"` with tools is
  still honoured as-is and will hit that upstream limit until a Responses adapter exists.
- README Quickstart `flux-os route` sample output updated to match the current catalog.
