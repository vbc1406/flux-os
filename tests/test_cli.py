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


def _readme_block(start: str) -> str:
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    i = text.index("```text\n" + start)
    return text[i + len("```text\n") : text.index("```", i + 8)]


def test_readme_hero_examples_match_router(monkeypatch: pytest.MonkeyPatch) -> None:
    import re

    from flux_os import Catalog, Router

    for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    router = Router(Catalog.load())
    entries = re.findall(
        r'"(.+?)"\s+→\s+(\S+)\s+\((\w+)\)\s+~\$([\d.]+)', _readme_block('"hi"'), flags=re.DOTALL
    )
    assert len(entries) == 4
    for prompt, model, provider, cost in entries:
        prompt = " ".join(prompt.split())
        d = router.route([{"role": "user", "content": prompt}], model="auto")
        assert (d.model.id, d.model.provider) == (model, provider), prompt
        assert float(cost) == float(f"{d.estimated_cost:.1g}"), prompt


def test_readme_quickstart_route_output_matches(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["route", "--all", "Prove that there are infinitely many primes"]) == 0
    actual = [ln.split() for ln in capsys.readouterr().out.strip().splitlines()]
    expected = [ln.split() for ln in _readme_block("model:").strip().splitlines()]
    assert actual == expected


def test_readme_model_count_matches_catalog() -> None:
    import re
    from pathlib import Path

    from flux_os import Catalog

    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    claimed = {int(n) for n in re.findall(r"\b(\d+) current models", text)}
    assert claimed == {len(Catalog.load().models)}
