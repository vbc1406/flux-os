"""Reroutes are observable: stderr log line, x-flux-attempts header, x-flux-* headers on errors."""

from __future__ import annotations

import httpx
import pytest

from flux_os.client import attempts_header, sanitize_reason

BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def http(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://flux")


async def _first_pick(http) -> tuple[str, str]:
    d = (await http.post("/v1/route", json=BODY)).json()
    return d["model"], d["upstream_model"]


async def test_attempts_header_and_stderr_log(http, fake, capsys) -> None:
    first, upstream = await _first_pick(http)
    fake.fail[upstream] = 429
    r = await http.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200 and r.headers["x-flux-rerouted"] == "true"
    attempts = r.headers["x-flux-attempts"].split(",")
    assert attempts[0] == f"{first}:429" and attempts[-1].endswith(":200")
    err = capsys.readouterr().err
    lines = [ln for ln in err.splitlines() if "reroute" in ln]
    assert len(lines) == 1
    assert f"from={first}" in lines[0] and "status=429" in lines[0] and "simulated 429" in lines[0]


async def test_no_reroute_no_log(http, capsys) -> None:
    r = await http.post("/v1/chat/completions", json=BODY)
    assert r.headers["x-flux-attempts"].endswith(":200") and "," not in r.headers["x-flux-attempts"]
    assert "reroute" not in capsys.readouterr().err


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", BODY),
        ("/v1/responses", {"model": "auto", "input": "hi"}),
        ("/v1/messages", {**BODY, "max_tokens": 10}),
    ],
)
async def test_error_responses_carry_flux_headers(http, flux, fake, path, body) -> None:
    for m in flux.catalog.models:
        fake.fail[m.upstream_id] = 503
    r = await http.post(path, json=body)
    assert r.status_code == 503
    for h in ("x-flux-model", "x-flux-provider", "x-flux-rerouted", "x-flux-mode", "x-flux-task"):
        assert h in r.headers, h
    assert r.headers["x-flux-attempts"].count(":503") == 3  # FLUX_OS_MAX_ATTEMPTS default


async def test_pinned_failure_has_headers(http, fake) -> None:
    fake.fail["gpt-5"] = 500
    r = await http.post("/v1/chat/completions", json={**BODY, "model": "gpt-5"})
    assert r.status_code == 500
    assert r.headers["x-flux-attempts"] == "gpt-5:500" and r.headers["x-flux-mode"] == "pinned"


def test_sanitize_masks_keys_and_truncates() -> None:
    out = sanitize_reason("HTTP 401: bad key sk-abcdefghijklmnop and Bearer abc.def.ghi\nline2 " + "x" * 500)
    assert "sk-abc" not in out and "abc.def" not in out and "\n" not in out and len(out) <= 120
    assert attempts_header([{"model": "a", "status": None}, {"model": "b", "status": 200}]) == "a:err,b:200"
