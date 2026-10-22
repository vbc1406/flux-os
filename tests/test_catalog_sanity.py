"""Guards against catalog mistakes that silently skew routing."""

from __future__ import annotations

import pytest

from flux_os import Catalog, Router

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY")
EASY = [{"role": "user", "content": "hi"}]


def _router(monkeypatch: pytest.MonkeyPatch, *keys: str) -> Router:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    for key in keys:
        monkeypatch.setenv(key, "test-key")
    return Router(Catalog.load())


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("mode", ["auto", "cheap", "fast"])
def test_easy_prompt_never_lands_on_the_priciest_model(
    monkeypatch: pytest.MonkeyPatch, key: str, mode: str
) -> None:
    # flux-fast once sent "hi" to claude-fable-5-1 ($10/$50) because its latency_ms was listed below Haiku's
    router = _router(monkeypatch, key)
    pool = router.pool()
    d = router.route(EASY, mode=mode)
    priciest = max(pool, key=lambda m: m.output_cost)
    assert d.model.output_cost < priciest.output_cost or len(pool) == 1, (key, mode, d.model.id)


def test_catalog_entries_are_well_formed() -> None:
    models = Catalog.load().models
    ids = [m.id for m in models]
    assert len(ids) == len(set(ids)), "duplicate model ids"
    for m in models:
        assert m.input_cost > 0 and m.output_cost > 0, m.id
        assert m.context_window > 0 and m.max_output_tokens > 0 and m.latency_ms > 0, m.id
        assert all(0 <= v <= 1 for v in m.quality.values()), m.id


def test_retired_models_are_not_in_the_catalog() -> None:
    # OpenAI shuts o4-mini down on 2026-10-23 (https://developers.openai.com/api/docs/deprecations)
    assert Catalog.load().get("o4-mini") is None


@pytest.mark.parametrize(
    ("model", "kept"),
    [
        ("claude-sonnet-5", False),
        ("claude-opus-4-7", False),
        ("claude-haiku-5-5", False),
        ("claude-fable-5-1", False),
        ("claude-sonnet-4-6", True),
        ("claude-haiku-4-5-20251001", True),
    ],
)
async def test_sampling_params_follow_what_each_claude_model_accepts(
    keys, fake, model: str, kept: bool
) -> None:
    # Claude 4.7+ answer 400 to a non-default temperature / top_p, which would fail a request that
    # an OpenAI-style client sent with temperature=0.2
    import httpx

    from flux_os import FluxOS

    flux = FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(fake.handler))
    body = {"model": model, "messages": EASY, "temperature": 0.2, "max_tokens": 20}
    await flux.dispatch(body)
    sent = fake.calls[-1]["body"]
    assert ("temperature" in sent) is kept, model


@pytest.mark.parametrize(
    ("name", "dropped"),
    [
        ("claude-opus-4-8", True),
        ("claude-opus-5-5", True),
        ("claude-sonnet-6-0", True),
        ("claude-sonnet-4-6", False),
        ("claude-opus-4-5-20251101", False),
        ("claude-3-7-sonnet-20250219", False),
    ],
)
def test_passthrough_claude_names_drop_sampling_only_when_required(name: str, dropped: bool) -> None:
    m = Catalog.load().resolve_passthrough(name)
    assert m is not None and bool(m.drop_params) is dropped, name
