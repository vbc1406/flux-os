"""Regression tests for the pre-launch security review."""

from __future__ import annotations

import time

import httpx
import pytest

from flux_os import Catalog, FluxOS, Router
from flux_os.client import sanitize_reason
from flux_os.server import create_app

CHAT = {"model": "auto", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}


def _client(fake, **app_kwargs) -> httpx.AsyncClient:
    flux = FluxOS(catalog=Catalog.load(), transport=httpx.MockTransport(fake.handler))
    app_kwargs.setdefault("api_key", "")
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(flux, **app_kwargs)), base_url="http://f"
    )


# ── ReDoS: the multi-part-list regex let `\s*` span newlines (quadratic) ─────────────────────────
@pytest.mark.parametrize("filler", ["\n", " \n", "\t\n\n", "1\n"])
def test_classifier_is_linear_on_whitespace_runs(filler: str) -> None:
    from flux_os.classifier import classify

    start = time.perf_counter()
    classify([{"role": "user", "content": filler * 20_000 + "1"}])
    assert time.perf_counter() - start < 0.5


# ── pinning must not bypass FLUX_OS_DISABLE / FLUX_OS_ONLY_MODELS ────────────────────────────────
@pytest.fixture
def restricted(monkeypatch: pytest.MonkeyPatch) -> Router:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("FLUX_OS_ONLY_MODELS", "provider:openai")
    monkeypatch.setenv("FLUX_OS_DISABLE", "gpt-5,provider:anthropic")
    return Router(Catalog.load())


@pytest.mark.parametrize("name", ["gpt-5", "claude-opus-4-7", "claude-brand-new-9-0", "anthropic/claude-x"])
def test_pinned_models_the_operator_switched_off_are_refused(restricted: Router, name: str) -> None:
    from flux_os.router import RoutingError

    with pytest.raises(RoutingError):
        restricted.route([{"role": "user", "content": "hi"}], model=name)


def test_pinning_still_works_for_allowed_models(restricted: Router) -> None:
    for name in ("gpt-5-mini", "gpt-9-ultra"):  # in the catalog / a pass-through name on an allowed provider
        assert restricted.route([{"role": "user", "content": "hi"}], model=name).model.provider == "openai"


def test_only_models_with_a_model_id_blocks_other_passthrough_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from flux_os.router import RoutingError

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("FLUX_OS_ONLY_MODELS", "gpt-5-mini")
    router = Router(Catalog.load())
    assert router.route([{"role": "user", "content": "hi"}], model="gpt-5-mini").model.id == "gpt-5-mini"
    with pytest.raises(RoutingError):
        router.route([{"role": "user", "content": "hi"}], model="gpt-9-ultra")


# ── model names reach response headers and logs ──────────────────────────────────────────────────
async def test_control_characters_in_model_are_rejected(keys, fake) -> None:
    async with _client(fake) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "model": "gpt-4o\r\nInjected: 1"})
    assert r.status_code == 400 and "model" in r.json()["error"]["message"]
    assert not fake.calls  # nothing was sent upstream


async def test_non_ascii_passthrough_model_keeps_headers_valid(keys, fake) -> None:
    async with _client(fake) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "model": "gpt-é€"})
    assert all(ord(ch) < 127 for ch in r.headers.get("x-flux-model", ""))


# ── loopback server without a key: CSRF and DNS rebinding ────────────────────────────────────────
@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({"host": "localhost:8000"}, 200),
        ({"host": "127.0.0.1:8000", "origin": "http://localhost:3000"}, 200),
        ({"host": "[::1]:8000"}, 200),
        ({"host": "rebind.evil.example:8000"}, 403),  # DNS rebinding
        ({"host": "localhost:8000", "origin": "http://evil.example"}, 403),  # cross-site POST
        ({"host": "localhost:8000", "origin": "null"}, 403),  # sandboxed iframe / data: URL
    ],
)
async def test_local_only_server_checks_host_and_origin(keys, fake, headers: dict, status: int) -> None:
    async with _client(fake, local_only=True) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers=headers)
    assert r.status_code == status
    if status == 403:
        assert not fake.calls, "a refused request must never spend credits"


async def test_allowed_hosts_env_extends_the_list(keys, fake, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX_OS_ALLOWED_HOSTS", "flux.internal")
    async with _client(fake, local_only=True) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers={"host": "flux.internal:8000"})
    assert r.status_code == 200


async def test_host_check_is_off_when_not_local_only(keys, fake) -> None:
    async with _client(fake) as c:  # library use / tests / reverse proxies
        r = await c.post("/v1/chat/completions", json=CHAT, headers={"host": "anything.example"})
    assert r.status_code == 200


def test_cli_serve_enables_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    import flux_os.cli as cli
    import flux_os.server as server

    seen: dict = {}
    monkeypatch.delenv("FLUX_OS_API_KEY", raising=False)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(server, "create_app", lambda flux, **kw: seen.update(kw) or object())
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert seen == {"local_only": True}
    seen.clear()
    monkeypatch.setenv("FLUX_OS_API_KEY", "k")  # with a key, auth is the protection
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert seen == {"local_only": False}


# ── cost knobs a client must not set ────────────────────────────────────────────────────────────
async def test_billing_and_retention_knobs_are_not_forwarded(keys, fake) -> None:
    body = {
        **CHAT,
        "model": "gpt-5-mini",
        "service_tier": "priority",
        "store": True,
        "plugins": [{"id": "web"}],
    }
    async with _client(fake) as c:
        r = await c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    sent = fake.calls[-1]["body"]
    assert not {"service_tier", "store", "plugins"} & sent.keys()


async def test_huge_n_is_rejected(keys, fake) -> None:
    async with _client(fake) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "n": 500})
    assert r.status_code == 400 and not fake.calls


# ── logs ────────────────────────────────────────────────────────────────────────────────────────
def test_configured_key_values_are_masked_in_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "abcdefghijklmnopqrstuvwxyz123456")  # no recognisable prefix
    out = sanitize_reason("401 invalid key abcdefghijklmnopqrstuvwxyz123456 for org-1", 400)
    assert "abcdefghij" not in out and "***" in out
