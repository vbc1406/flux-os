from __future__ import annotations

import pytest

from flux_os import Catalog, Router, RoutingError, classify, route
from flux_os.router import parse_directive

EASY = "What is the capital of France?"
HARD = (
    "Implement a lock-free concurrent hash map in Rust that is thread-safe, handles resizing "
    "without blocking readers, and must not use global locks. Ensure correctness under all edge cases."
)
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def msgs(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


@pytest.fixture
def router(keys: None) -> Router:
    return Router(Catalog.load())


@pytest.mark.parametrize(
    "text,task",
    [
        ("hi there", "conversation"),
        (EASY, "simple_qa"),
        ("Translate 'good morning' into Japanese", "translation"),
        ("Summarize the key points of this memo: ...", "summarization"),
        ("Write a Python function that merges two sorted lists", "code_generation"),
        ("Why does this code deadlock?\n```go\nmu.Lock()\n```", "code_review"),
        ("Write a short poem about the sea", "creative_writing"),
        ("Prove that the square root of 2 is irrational", "reasoning"),
        ("Compare the trade-offs of REST and gRPC for internal services", "analysis"),
        ("Classify the sentiment of this review: great product!", "classification"),
        ("Extract every email address from the text below", "extraction"),
    ],
)
def test_task_detection(text: str, task: str) -> None:
    assert classify(msgs(text)).task == task


def test_complexity_orders_easy_below_hard() -> None:
    assert classify(msgs(EASY)).complexity < 0.3 < classify(msgs(HARD)).complexity


def test_easy_prompt_goes_cheaper_than_hard(router: Router) -> None:
    easy, hard = router.route(msgs(EASY)), router.route(msgs(HARD))
    assert easy.model.output_cost < hard.model.output_cost
    assert hard.model.quality_for("code_generation") >= hard.quality_bar
    assert easy.model.quality_for("simple_qa") >= easy.quality_bar


def test_chosen_model_is_cheapest_that_clears_bar(router: Router) -> None:
    d = router.route(msgs(HARD))
    a = d.analysis
    clearing = [m for m in router.pool() if m.quality_for(a.task) >= d.quality_bar]
    cheapest = min(m.estimate_cost(a.input_tokens, a.output_tokens) for m in clearing)
    assert d.estimated_cost == pytest.approx(cheapest)


def test_fast_mode_prefers_low_latency(router: Router) -> None:
    auto_ = router.route(msgs(HARD))
    fast = router.route(msgs(HARD), model="flux-fast")
    assert fast.model.latency_ms <= auto_.model.latency_ms


def test_cheap_mode_never_costs_more(router: Router) -> None:
    for text in (EASY, HARD, "Compare Postgres and MySQL for analytics"):
        assert (
            router.route(msgs(text), model="flux-cheap").estimated_cost
            <= router.route(msgs(text)).estimated_cost
        )


def test_tools_filter_to_tool_capable_models(router: Router) -> None:
    d = router.route(msgs("what's the weather in Paris?"), tools=TOOLS)
    assert all(m.tools for m in d.candidates)
    assert d.analysis.agent_step == "tool_call"
    assert d.quality_bar >= 0.8


def test_vision_filter(router: Router) -> None:
    content = [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    d = router.route([{"role": "user", "content": content}])
    assert d.analysis.needs_vision
    assert all(m.vision for m in d.candidates)


def test_json_and_context_filters(router: Router) -> None:
    d = router.route(msgs("list the colors as json"), response_format={"type": "json_object"})
    assert all(m.json_mode for m in d.candidates)
    huge = "word " * 700_000  # ~875k tokens
    d = router.route(msgs("summarize: " + huge))
    assert all(m.context_window >= 800_000 for m in d.candidates)


def test_only_models_with_keys_are_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    d = Router(Catalog.load()).route(msgs(HARD))
    assert {m.provider for m in d.candidates} == {"anthropic"}


def test_no_keys_is_a_clear_error() -> None:
    with pytest.raises(RoutingError, match="no models available"):
        Router(Catalog.load()).route(msgs(EASY))


def test_pinned_model_is_honoured(router: Router) -> None:
    d = router.route(msgs(EASY), model="claude-opus-5")
    assert d.pinned and d.model.id == "claude-opus-5" and len(d.candidates) == 1


def test_passthrough_models(router: Router) -> None:
    assert router.route(msgs(EASY), model="gpt-4.1").model.provider == "openai"
    assert router.route(msgs(EASY), model="ollama/llama3.2").model.upstream_id == "llama3.2"
    with pytest.raises(RoutingError, match="unknown model"):
        router.route(msgs(EASY), model="totally-made-up")


def test_directives() -> None:
    assert parse_directive("auto") == ("auto", None)
    assert parse_directive(None) == ("auto", None)
    assert parse_directive("flux-cheap") == ("cheap", None)
    assert parse_directive("gpt-5") == (None, "gpt-5")


def test_tool_loop_continuity(router: Router) -> None:
    first = router.route(msgs("weather in Paris?"), tools=TOOLS)
    other = first.candidates[3]
    router.remember_tool_calls([{"id": "call_1"}], other.id)
    convo = msgs("weather in Paris?") + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
    ]
    d = router.route(convo, tools=TOOLS)
    assert d.model.id == other.id
    assert "tool_loop_continuity" in d.signals


def test_only_and_disable_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("FLUX_OS_DISABLE", "provider:openai")
    cat = Catalog.load()
    providers = {m.provider for m in cat.available()}
    assert providers == {"anthropic"}


def test_module_level_route(keys: None) -> None:
    import flux_os

    flux_os._default = None
    assert route("hello").model.id
    assert route("hello").as_dict()["analysis"]["task"] == "conversation"


def test_routing_is_fast(router: Router) -> None:
    import time

    start = time.perf_counter()
    for _ in range(200):
        router.route(msgs(HARD))
    assert (time.perf_counter() - start) / 200 < 0.005


def _two_model_catalog(cheap_quality: float) -> Catalog:
    from flux_os.catalog import Model

    base = Catalog.load()
    models = [
        Model(
            id="cheap", provider="openai", input_cost=0.1, output_cost=0.1, quality={"general": cheap_quality}
        ),
        Model(id="pricey", provider="openai", input_cost=5.0, output_cost=5.0, quality={"general": 0.95}),
    ]
    return Catalog(providers=base.providers, models=models)


def test_quality_tolerance_band_stops_razor_thin_bumps() -> None:
    probe = Router(_two_model_catalog(0.5), quality_tolerance=0.0, require_credentials=False)
    bar = probe.route(msgs("hi"), model="auto").quality_bar
    # the cheap model misses the bar by 0.002 (the real-world gpt-oss-20b vs gpt-oss-120b case)
    catalog = _two_model_catalog(round(bar - 0.002, 4))
    strict = Router(catalog, quality_tolerance=0.0, require_credentials=False)
    assert strict.route(msgs("hi"), model="auto").model.id == "pricey"
    banded = Router(catalog, quality_tolerance=0.02, require_credentials=False)
    d = banded.route(msgs("hi"), model="auto")
    assert d.model.id == "cheap" and d.quality_bar == bar  # reported bar itself is unchanged
    # but a real gap (0.05 below the bar) is still not waved through
    far = Router(_two_model_catalog(round(bar - 0.05, 4)), quality_tolerance=0.02, require_credentials=False)
    assert far.route(msgs("hi"), model="auto").model.id == "pricey"
    # deterministic
    assert [banded.route(msgs("hi"), model="auto").model.id for _ in range(5)] == ["cheap"] * 5


def test_quality_tolerance_env_and_default(monkeypatch: pytest.MonkeyPatch, keys: None) -> None:
    assert Router(Catalog.load()).quality_tolerance == 0.02
    monkeypatch.setenv("FLUX_OS_QUALITY_TOLERANCE", "0")
    assert Router(Catalog.load()).quality_tolerance == 0.0
    with pytest.raises(ValueError):
        Router(Catalog.load(), quality_tolerance=-0.1)
