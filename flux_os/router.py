"""The routing decision: pick the cheapest model that clears the quality bar.

    1. classify the request (task type, complexity, tools / vision / JSON needs)
    2. filter to models that can serve it (credentials, capabilities, context window)
    3. quality bar = 0.72 + 0.22 * complexity (raised for tool-calling agent steps)
    4. rank models that clear the bar by estimated cost (or latency / quality,
       depending on the mode); models below the bar follow, best quality first
    5. the ranked list is the reroute order if the first choice fails

Deterministic, explainable, no network calls.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .catalog import Catalog, Model
from .classifier import Analysis, classify

MODES = ("auto", "cheap", "fast")

_DIRECTIVES = {
    "": "auto",
    "auto": "auto",
    "flux-auto": "auto",
    "flux-os": "auto",
    "flux": "auto",
    "router": "auto",
    "cheap": "cheap",
    "flux-cheap": "cheap",
    "auto:cheap": "cheap",
    "fast": "fast",
    "flux-fast": "fast",
    "auto:fast": "fast",
}

BAR_BASE = 0.72
BAR_SLOPE = 0.22
TOOL_BAR_MIN = 0.80
CHEAP_BAR_DISCOUNT = 0.06


class RoutingError(Exception):
    """No model can serve the request (bad pin, no keys, or no capable model)."""

    status_code = 400


def parse_directive(model: str | None) -> tuple[str | None, str | None]:
    """Return (mode, pinned_model). Exactly one of them is set."""
    name = (model or "").strip()
    mode = _DIRECTIVES.get(name.lower())
    if mode:
        return mode, None
    return None, name


@dataclass
class Decision:
    model: Model
    candidates: list[Model]
    analysis: Analysis
    mode: str
    quality_bar: float
    pinned: bool = False
    signals: list[str] = field(default_factory=list)

    @property
    def estimated_cost(self) -> float:
        a = self.analysis
        return self.model.estimate_cost(a.input_tokens, a.output_tokens)

    def as_dict(self) -> dict[str, Any]:
        a = self.analysis
        return {
            "model": self.model.id,
            "provider": self.model.provider,
            "upstream_model": self.model.upstream_id,
            "mode": "pinned" if self.pinned else self.mode,
            "quality_bar": round(self.quality_bar, 3),
            "estimated_cost_usd": round(self.estimated_cost, 6),
            "analysis": a.as_dict(),
            "signals": self.signals,
            "fallbacks": [m.id for m in self.candidates[1:]],
        }


class _LRU:
    def __init__(self, size: int) -> None:
        self.size = size
        self.data: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self.lock:
            value = self.data.get(key)
            if value is not None:
                self.data.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)


class Router:
    """Stateless routing apart from a small tool-call → model memory for agents."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        *,
        default_mode: str | None = None,
        quality_offset: float | None = None,
        require_credentials: bool = True,
    ) -> None:
        self.catalog = catalog or Catalog.load()
        self.default_mode = (default_mode or os.environ.get("FLUX_OS_DEFAULT_MODE") or "auto").lower()
        if self.default_mode not in MODES:
            raise ValueError(f"default_mode must be one of {MODES}")
        if quality_offset is None:
            quality_offset = float(os.environ.get("FLUX_OS_QUALITY_OFFSET", "0") or 0)
        self.quality_offset = quality_offset
        self.require_credentials = require_credentials
        self._tool_calls = _LRU(10_000)
        self._extra_content = _LRU(10_000)

    # ── agent tool-loop continuity ──────────────────────────────────────────
    def remember_tool_calls(self, tool_calls: list[dict[str, Any]] | None, model_id: str) -> None:
        """Remember which model issued each tool call, so the tool result goes back to it."""
        for tc in tool_calls or []:
            if isinstance(tc, dict) and tc.get("id"):
                self._tool_calls.put(str(tc["id"]), model_id)
                if tc.get("extra_content"):
                    self._extra_content.put(str(tc["id"]), tc["extra_content"])

    def restore_extra_content(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-attach provider-opaque ``extra_content`` (Gemini 3 ``thought_signature``) to
        assistant tool calls that a client dropped when echoing history back.

        Returns ``messages`` itself when nothing needs restoring.
        """
        out: list[dict[str, Any]] | None = None
        for i, m in enumerate(messages):
            calls = m.get("tool_calls") if m.get("role") == "assistant" else None
            if not isinstance(calls, list):
                continue
            fixed = []
            for tc in calls:
                saved = None
                if isinstance(tc, dict) and tc.get("id") and not tc.get("extra_content"):
                    saved = self._extra_content.get(str(tc["id"]))
                fixed.append({**tc, "extra_content": saved} if saved else tc)
            if any(a is not b for a, b in zip(fixed, calls, strict=True)):
                if out is None:
                    out = list(messages)
                out[i] = {**m, "tool_calls": fixed}
        return out if out is not None else messages

    def _tool_loop_model(self, messages: list[dict[str, Any]]) -> str | None:
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("tool_call_id"):
                found = self._tool_calls.get(str(m["tool_call_id"]))
                if found:
                    return found
            elif m.get("role") != "tool":
                break
        return None

    # ── routing ─────────────────────────────────────────────────────────────
    def pool(self) -> list[Model]:
        if self.require_credentials:
            return self.catalog.available()
        return [m for m in self.catalog.models if m.enabled]

    def route(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[Any] | None = None,
        response_format: Any = None,
        max_tokens: int | None = None,
        mode: str | None = None,
    ) -> Decision:
        """Pick a model for an OpenAI-style chat request.

        ``model`` accepts a directive (``auto``, ``flux-cheap``, ``flux-fast``)
        or a concrete model id, which is honoured as-is (pinned).
        """
        analysis = classify(messages, tools=tools, response_format=response_format, max_tokens=max_tokens)
        directive_mode, pinned = parse_directive(model)
        if pinned:
            return self._pinned(pinned, analysis)
        mode = (mode or directive_mode or self.default_mode).lower()
        if mode == "auto" and directive_mode == "auto" and self.default_mode != "auto":
            mode = self.default_mode
        if mode not in MODES:
            raise RoutingError(f"unknown routing mode '{mode}' (use one of {', '.join(MODES)})")

        pool = self.pool()
        if not pool:
            raise RoutingError(
                "no models available: set at least one provider key "
                "(OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, ...)"
            )
        eligible, dropped = [], {}
        need_ctx = analysis.input_tokens + analysis.output_tokens
        for m in pool:
            if analysis.needs_tools and not m.tools:
                dropped[m.id] = "no tool support"
            elif analysis.needs_vision and not m.vision:
                dropped[m.id] = "no vision"
            elif analysis.needs_json and not m.json_mode:
                dropped[m.id] = "no JSON mode"
            elif need_ctx > m.context_window:
                dropped[m.id] = "context window too small"
            else:
                eligible.append(m)
        if not eligible:
            needs = [
                n
                for n, on in (
                    ("tools", analysis.needs_tools),
                    ("vision", analysis.needs_vision),
                    ("json", analysis.needs_json),
                )
                if on
            ]
            raise RoutingError(
                f"no configured model supports this request (needs: {', '.join(needs) or 'none'}, "
                f"~{need_ctx} tokens)"
            )

        bar = BAR_BASE + BAR_SLOPE * analysis.complexity + self.quality_offset
        if analysis.needs_tools:
            bar = max(bar, TOOL_BAR_MIN)
        if mode == "cheap":
            bar -= CHEAP_BAR_DISCOUNT
        bar = max(0.0, min(bar, 0.99))

        def quality(m: Model) -> float:
            q = m.quality_for(analysis.task)
            if analysis.needs_tools:
                q = (q + m.quality_for("function_calling")) / 2
            return q

        def cost(m: Model) -> float:
            return m.estimate_cost(analysis.input_tokens, analysis.output_tokens)

        passing = [m for m in eligible if quality(m) >= bar]
        failing = [m for m in eligible if quality(m) < bar]
        if mode == "fast":
            ranked = sorted(passing, key=lambda m: (m.latency_ms, cost(m)))
            ranked += sorted(failing, key=lambda m: (-quality(m), m.latency_ms))
        else:
            ranked = sorted(passing, key=lambda m: (cost(m), -quality(m)))
            ranked += sorted(failing, key=lambda m: (-quality(m), cost(m)))

        signals = list(analysis.signals)
        held = self._tool_loop_model(messages) if analysis.agent_step == "tool_result" else None
        if held:
            match = next((m for m in ranked if m.id == held), None)
            if match is not None:
                ranked.remove(match)
                ranked.insert(0, match)
                signals.append("tool_loop_continuity")

        chosen = ranked[0]
        return Decision(
            model=chosen,
            candidates=ranked,
            analysis=analysis,
            mode=mode,
            quality_bar=bar,
            signals=signals,
        )

    def _pinned(self, name: str, analysis: Analysis) -> Decision:
        model = self.catalog.get(name) or self.catalog.resolve_passthrough(name)
        if model is None:
            raise RoutingError(
                f"unknown model '{name}'. Use 'auto' (or flux-cheap / flux-fast), "
                "a catalog id (see `flux-os models`), or 'provider/model' such as 'ollama/llama3.2'"
            )
        if self.require_credentials and not self.catalog.providers[model.provider].configured:
            raise RoutingError(
                f"model '{name}' needs provider '{model.provider}', which has no API key configured"
            )
        return Decision(
            model=model,
            candidates=[model],
            analysis=analysis,
            mode="pinned",
            quality_bar=0.0,
            pinned=True,
            signals=list(analysis.signals) + ["pinned"],
        )
