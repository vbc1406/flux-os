"""``flux-os`` command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from ._version import __version__


def _tools_stub() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "example_tool",
                "description": "placeholder tool so routing treats this as an agent step",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .client import FluxOS
    from .server import create_app

    host = args.host or os.environ.get("FLUX_OS_HOST", "127.0.0.1")
    port = int(args.port or os.environ.get("FLUX_OS_PORT") or os.environ.get("PORT") or 8000)
    flux = FluxOS()
    available = flux.catalog.available()
    print(f"flux-os {__version__}", file=sys.stderr)
    print(
        f"  OpenAI API:    http://{host}:{port}/v1   (model: auto | flux-cheap | flux-fast)",
        file=sys.stderr,
    )
    print(f"  Anthropic API: http://{host}:{port}      (ANTHROPIC_BASE_URL)", file=sys.stderr)
    providers = sorted({m.provider for m in available})
    print(f"  routing over {len(available)} models from: {', '.join(providers) or 'none'}", file=sys.stderr)
    if not available:
        print(
            "  ! no provider keys found - set OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, ...",
            file=sys.stderr,
        )
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("FLUX_OS_API_KEY"):
        print(
            "  ! listening on a public interface without FLUX_OS_API_KEY - anyone who can reach "
            "this port can spend your provider credits",
            file=sys.stderr,
        )
        if not _env_bool("FLUX_OS_ALLOW_UNAUTHENTICATED"):
            print(
                "  refusing to start: set FLUX_OS_API_KEY, bind to 127.0.0.1, or set "
                "FLUX_OS_ALLOW_UNAUTHENTICATED=1 to start anyway",
                file=sys.stderr,
            )
            return 1
    uvicorn.run(create_app(flux), host=host, port=port, log_level=args.log_level)
    return 0


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def cmd_route(args: argparse.Namespace) -> int:
    from .catalog import Catalog
    from .router import Router, RoutingError

    catalog = Catalog.load()
    router = Router(catalog, require_credentials=not args.all)
    if not args.all and not catalog.available():
        print(
            "(no provider keys set - showing the decision over the full catalog; use --all to silence)\n",
            file=sys.stderr,
        )
        router = Router(catalog, require_credentials=False)
    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
    try:
        d = router.route(
            [{"role": "user", "content": prompt}],
            model=args.mode,
            tools=_tools_stub() if args.tools else None,
            response_format={"type": "json_object"} if args.json_output else None,
        )
    except RoutingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(d.as_dict(), indent=2))
        return 0
    a = d.analysis
    print(f"model:       {d.model.id}  ({d.model.provider})")
    print(f"task:        {a.task}   complexity {a.complexity:.2f}   quality bar {d.quality_bar:.2f}")
    print(f"est. cost:   ${d.estimated_cost:.6f}")
    print(f"reroute to:  {', '.join(m.id for m in d.candidates[1:4]) or '-'}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from .catalog import Catalog

    catalog = Catalog.load()
    rows = []
    for m in sorted(catalog.models, key=lambda m: (m.provider, m.input_cost)):
        ok = m.enabled and catalog.providers[m.provider].configured
        if not ok and not args.all:
            continue
        flags = "".join(c for c, on in (("T", m.tools), ("V", m.vision), ("J", m.json_mode)) if on)
        rows.append(
            (
                "yes" if ok else "-",
                m.id,
                m.provider,
                f"${m.input_cost:g}/${m.output_cost:g}",
                f"{m.context_window // 1000}k",
                flags,
                f"{m.quality_for('general'):.2f}",
            )
        )
    if not rows:
        print("No models available. Set a provider key (e.g. OPENAI_API_KEY) or use --all.")
        return 0
    head = ("ready", "model", "provider", "$/1M in/out", "context", "caps", "quality")
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)))
    print("\ncaps: T=tools V=vision J=JSON mode")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from .client import FluxOS

    flux = FluxOS()
    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
    req = {"model": args.model, "messages": [{"role": "user", "content": prompt}]}
    if args.stream:
        stream = flux.create(**req, stream=True)
        print(f"[{stream.flux['model']}] ", end="", file=sys.stderr, flush=True)
        for chunk in stream:
            for choice in chunk.get("choices") or []:
                print((choice.get("delta") or {}).get("content") or "", end="", flush=True)
        print()
    else:
        resp = flux.create(**req)
        print(f"[{resp.flux['model']}]", file=sys.stderr)
        print(resp.choices[0].message.content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flux-os", description="Route every LLM request to the right model.")
    p.add_argument("--version", action="version", version=f"flux-os {__version__}")
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("serve", help="run the OpenAI/Anthropic-compatible routing proxy")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--log-level", default="info")
    s.set_defaults(func=cmd_serve)

    r = sub.add_parser("route", help="show which model a prompt would be routed to (no API call)")
    r.add_argument("prompt", nargs="*")
    r.add_argument("--mode", default="auto", help="auto | cheap | fast")
    r.add_argument("--tools", action="store_true", help="treat as an agent step with tools")
    r.add_argument("--json-output", action="store_true", help="request needs JSON mode")
    r.add_argument("--all", action="store_true", help="route over the full catalog, ignoring keys")
    r.add_argument("--json", action="store_true", help="print the decision as JSON")
    r.set_defaults(func=cmd_route)

    m = sub.add_parser("models", help="list models and whether they are ready to route to")
    m.add_argument("--all", action="store_true", help="include models without a configured key")
    m.set_defaults(func=cmd_models)

    c = sub.add_parser("chat", help="send one prompt through the router")
    c.add_argument("prompt", nargs="*")
    c.add_argument("--model", default="auto")
    c.add_argument("--stream", action="store_true")
    c.set_defaults(func=cmd_chat)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        from .client import UpstreamError
        from .router import RoutingError

        if isinstance(exc, (RoutingError, UpstreamError)):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise
