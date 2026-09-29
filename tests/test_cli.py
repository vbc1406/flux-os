from __future__ import annotations

import json

import pytest

from flux_os.cli import main


def test_route_json_without_keys(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["route", "--all", "--json", "Write a Python function to reverse a list"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["analysis"]["task"] == "code_generation" and d["model"]


def test_route_text_with_keys(keys: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["route", "--tools", "--mode", "fast", "book a flight"]) == 0
    out = capsys.readouterr().out
    assert "model:" in out and "reroute to:" in out


def test_models_listing(keys: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-5" in out and "gpt-5-mini" in out


def test_models_without_keys(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models"]) == 0
    assert "No models available" in capsys.readouterr().out


def test_serve_refuses_public_bind_without_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("FLUX_OS_API_KEY", raising=False)
    monkeypatch.delenv("FLUX_OS_ALLOW_UNAUTHENTICATED", raising=False)
    assert main(["serve", "--host", "0.0.0.0"]) == 1
    assert "refusing to start" in capsys.readouterr().err


def test_serve_allows_public_bind_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUX_OS_API_KEY", "secret")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    assert main(["serve", "--host", "0.0.0.0"]) == 0


def test_serve_allows_public_bind_with_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLUX_OS_API_KEY", raising=False)
    monkeypatch.setenv("FLUX_OS_ALLOW_UNAUTHENTICATED", "1")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    assert main(["serve", "--host", "0.0.0.0"]) == 0


def test_serve_allows_loopback_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLUX_OS_API_KEY", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    assert main(["serve", "--host", "127.0.0.1"]) == 0
