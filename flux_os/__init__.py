"""flux-os — route every LLM request to the cheapest model that can handle it.

Works as a Python library, an OpenAI/Anthropic-compatible proxy, or a
routing-decision API. See https://github.com/vbc1406/flux-os
"""

from ._version import __version__
from .catalog import Catalog, Model, Provider
from .classifier import Analysis, classify
from .client import FluxOS, RouteInfo, UpstreamError
from .providers import ProviderError
from .router import Decision, Router, RoutingError

_default: Router | None = None


def route(messages, **kwargs) -> Decision:
    """Pick a model for an OpenAI-style request without calling it.

    >>> from flux_os import route
    >>> route([{"role": "user", "content": "hi"}]).model.id  # doctest: +SKIP
    """
    global _default
    if _default is None:
        _default = Router()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return _default.route(messages, **kwargs)


__all__ = [
    "Analysis",
    "Catalog",
    "Decision",
    "FluxOS",
    "Model",
    "Provider",
    "ProviderError",
    "RouteInfo",
    "Router",
    "RoutingError",
    "UpstreamError",
    "__version__",
    "classify",
    "route",
]
