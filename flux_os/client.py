"""In-process client: route → call → reroute on failure.

Use it directly from Python (no server needed)::

    from flux_os import FluxOS

    flux = FluxOS()
    resp = flux.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}]
    )
    print(resp.choices[0].message.content, resp.flux["model"])

The HTTP server is a thin layer over the same ``FluxOS.dispatch`` /
``FluxOS.dispatch_stream`` methods, so both paths route identically.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from .catalog import Catalog, Model
from .providers import ProviderError, Upstream
from .router import Decision, Router, RoutingError, parse_directive


class UpstreamError(Exception):
    """Every candidate model failed."""

    def __init__(self, message: str, attempts: list[dict[str, Any]], status: int | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status if status and 400 <= status < 600 else 502


@dataclass
class RouteInfo:
    decision: Decision
    model: Model
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rerouted(self) -> bool:
        return self.model.id != self.decision.model.id

    def as_dict(self) -> dict[str, Any]:
        d = self.decision.as_dict()
        d.update(
            {
                "model": self.model.id,
                "provider": self.model.provider,
                "routed_to": self.decision.model.id,
                "rerouted": self.rerouted,
                "attempts": self.attempts,
            }
        )
        return d


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


class Obj(dict):
    """A dict with attribute access, so results read like the OpenAI SDK's objects."""

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap(self[name])
        except KeyError:
            if name in ("tool_calls", "content", "usage", "refusal", "finish_reason"):
                return None
            raise AttributeError(name) from None

    def __getitem__(self, key: Any) -> Any:
        return _wrap(dict.__getitem__(self, key))

    def model_dump(self) -> dict[str, Any]:
        return dict(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, Obj):
        return Obj(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class _LoopThread:
    """One background event loop so the sync API can share an async HTTP client."""

    _lock = threading.Lock()
    _loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def loop(cls) -> asyncio.AbstractEventLoop:
        with cls._lock:
            if cls._loop is None:
                cls._loop = asyncio.new_event_loop()
                threading.Thread(target=cls._loop.run_forever, name="flux-os-loop", daemon=True).start()
            return cls._loop

    @classmethod
    def run(cls, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, cls.loop()).result()


class FluxOS:
    """Route OpenAI-style chat requests to the right model, rerouting on failure."""

    def __init__(
        self,
        *,
        api_keys: dict[str, str] | None = None,
        default_mode: str | None = None,
        max_attempts: int | None = None,
        reroute_pinned: bool | None = None,
        route_all: bool | None = None,
        catalog: Catalog | None = None,
        transport: Any = None,
    ) -> None:
        self.catalog = catalog or Catalog.load(api_keys=api_keys)
        self.router = Router(self.catalog, default_mode=default_mode)
        self.max_attempts = max_attempts or int(os.environ.get("FLUX_OS_MAX_ATTEMPTS", "3"))
        self.reroute_pinned = (
            _env_bool("FLUX_OS_REROUTE_PINNED") if reroute_pinned is None else reroute_pinned
        )
        self.route_all = _env_bool("FLUX_OS_ROUTE_ALL") if route_all is None else route_all
        self._transport = transport  # custom httpx transport (tests, proxies)
        self._upstreams: dict[int, Upstream] = {}
        self.chat = _Chat(self)

    @property
    def upstream(self) -> Upstream:
        # httpx clients are bound to the loop they were created on.
        try:
            key = id(asyncio.get_running_loop())
        except RuntimeError:
            key = 0
        if key not in self._upstreams:
            client = None
            if self._transport is not None:
                import httpx

                client = httpx.AsyncClient(transport=self._transport)
            self._upstreams[key] = Upstream(client)
        return self._upstreams[key]

    async def aclose(self) -> None:
        for up in self._upstreams.values():
            await up.aclose()
        self._upstreams.clear()

    # ── routing only ────────────────────────────────────────────────────────
    def route(self, messages: list[dict[str, Any]], **kwargs: Any) -> Decision:
        """Decide which model should serve a request, without calling it."""
        return self.router.route(messages, model=self._model_for(kwargs.pop("model", None)), **kwargs)

    def _model_for(self, model: str | None) -> str | None:
        """With ``route_all`` every concrete model name is treated as ``auto``
        (for tools that hard-code a model name, like Claude Code)."""
        if self.route_all and parse_directive(model)[1]:
            return "auto"
        return model

    def _decide(self, req: dict[str, Any], reroute: bool | None) -> tuple[Decision, list[Model]]:
        model = self._model_for(req.get("model"))
        decision = self.router.route(
            req.get("messages") or [],
            model=model,
            tools=req.get("tools"),
            response_format=req.get("response_format"),
            max_tokens=req.get("max_completion_tokens") or req.get("max_tokens"),
        )
        candidates = list(decision.candidates)
        allow = self.reroute_pinned if reroute is None else reroute
        if decision.pinned and allow:
            try:
                routed = self.router.route(
                    req.get("messages") or [],
                    model="auto",
                    tools=req.get("tools"),
                    response_format=req.get("response_format"),
                )
                candidates += [m for m in routed.candidates if m.id != decision.model.id]
            except RoutingError:
                pass
        return decision, candidates

    def _restored(self, req: dict[str, Any]) -> dict[str, Any]:
        messages = req.get("messages")
        if not isinstance(messages, list):
            return req
        fixed = self.router.restore_extra_content(messages)
        return req if fixed is messages else {**req, "messages": fixed}

    def _order(self, remaining: list[Model], failed_provider: str | None) -> list[Model]:
        """After a failure, try other providers first (outages are usually provider-wide)."""
        if not failed_provider:
            return remaining
        return sorted(remaining, key=lambda m: m.provider == failed_provider)

    # ── async API ───────────────────────────────────────────────────────────
    async def dispatch(
        self, req: dict[str, Any], *, reroute: bool | None = None
    ) -> tuple[dict[str, Any], RouteInfo]:
        """Route and call (non-streaming). Returns (openai_chat_completion, RouteInfo)."""
        req = self._restored(req)
        decision, remaining = self._decide(req, reroute)
        attempts: list[dict[str, Any]] = []
        last: ProviderError | None = None
        while remaining and len(attempts) < self.max_attempts:
            model = remaining.pop(0)
            provider = self.catalog.providers[model.provider]
            try:
                resp = await self.upstream.complete(req, model, provider)
            except ProviderError as exc:
                attempts.append({"model": model.id, "error": str(exc), "status": exc.status})
                last = exc
                if not exc.retryable:
                    break
                remaining = self._order(remaining, model.provider)
                continue
            msg = ((resp.get("choices") or [{}])[0]).get("message") or {}
            self.router.remember_tool_calls(msg.get("tool_calls"), model.id)
            attempts.append({"model": model.id, "status": 200})
            return resp, RouteInfo(decision, model, attempts)
        raise UpstreamError(
            f"all {len(attempts)} attempt(s) failed; last error: {last}",
            attempts,
            last.status if last else None,
        )

    async def dispatch_stream(
        self, req: dict[str, Any], *, reroute: bool | None = None
    ) -> tuple[AsyncIterator[dict[str, Any]], RouteInfo]:
        """Route and call (streaming). Reroutes only before the first chunk is received."""
        req = self._restored(req)
        decision, remaining = self._decide(req, reroute)
        attempts: list[dict[str, Any]] = []
        last: ProviderError | None = None
        while remaining and len(attempts) < self.max_attempts:
            model = remaining.pop(0)
            provider = self.catalog.providers[model.provider]
            gen = self.upstream.stream(req, model, provider)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                first = None
            except ProviderError as exc:
                attempts.append({"model": model.id, "error": str(exc), "status": exc.status})
                last = exc
                await gen.aclose()
                if not exc.retryable:
                    break
                remaining = self._order(remaining, model.provider)
                continue
            attempts.append({"model": model.id, "status": 200})
            info = RouteInfo(decision, model, attempts)
            return self._relay(first, gen, model.id), info
        raise UpstreamError(
            f"all {len(attempts)} attempt(s) failed; last error: {last}",
            attempts,
            last.status if last else None,
        )

    async def _relay(
        self, first: dict[str, Any] | None, gen: AsyncIterator[dict[str, Any]], model_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        calls: dict[int, dict[str, Any]] = {}

        def note(chunk: dict[str, Any]) -> None:
            for choice in chunk.get("choices") or []:
                for tc in (choice.get("delta") or {}).get("tool_calls") or []:
                    idx = int(tc.get("index", 0))
                    if tc.get("id"):
                        calls[idx] = {"id": tc["id"]}
                    if tc.get("extra_content") and idx in calls:
                        calls[idx]["extra_content"] = tc["extra_content"]

        try:
            if first is not None:
                note(first)
                yield first
            async for chunk in gen:
                note(chunk)
                yield chunk
        finally:
            await gen.aclose()  # type: ignore[attr-defined]
            if calls:
                self.router.remember_tool_calls(list(calls.values()), model_id)

    async def acreate(self, **req: Any) -> Any:
        """Async OpenAI-style call. ``stream=True`` returns an async iterator of chunks."""
        reroute = req.pop("reroute", None)
        if req.get("stream"):
            stream, info = await self.dispatch_stream(req, reroute=reroute)
            return _AsyncStream(stream, info)
        resp, info = await self.dispatch(req, reroute=reroute)
        out = Obj(resp)
        out["flux"] = info.as_dict()
        return out

    # ── sync API ────────────────────────────────────────────────────────────
    def create(self, **req: Any) -> Any:
        """Sync OpenAI-style call. ``stream=True`` returns an iterator of chunks."""
        if req.get("stream"):
            stream = _LoopThread.run(self.acreate(**req))
            return _SyncStream(stream)
        return _LoopThread.run(self.acreate(**req))


class _AsyncStream:
    def __init__(self, stream: AsyncIterator[dict[str, Any]], info: RouteInfo) -> None:
        self._stream = stream
        self.flux = info.as_dict()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Any]:
        async for chunk in self._stream:
            yield Obj(chunk)


class _SyncStream:
    def __init__(self, stream: _AsyncStream) -> None:
        self._it = stream.__aiter__()
        self.flux = stream.flux

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                yield _LoopThread.run(self._it.__anext__())
            except StopAsyncIteration:
                return


class _Completions:
    def __init__(self, flux: FluxOS) -> None:
        self._flux = flux

    def create(self, **req: Any) -> Any:
        return self._flux.create(**req)

    async def acreate(self, **req: Any) -> Any:
        return await self._flux.acreate(**req)


class _Chat:
    def __init__(self, flux: FluxOS) -> None:
        self.completions = _Completions(flux)
