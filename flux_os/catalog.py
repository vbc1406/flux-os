"""Model catalog and provider registry.

The catalog is plain data (``models.json``). Providers are either
OpenAI-compatible (one adapter covers OpenAI, Google, Groq, Mistral, Ollama,
vLLM, LM Studio, OpenRouter, Together, ...) or Anthropic's Messages API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

TASK_TYPES = (
    "simple_qa",
    "conversation",
    "translation",
    "classification",
    "extraction",
    "summarization",
    "code_generation",
    "code_review",
    "creative_writing",
    "analysis",
    "reasoning",
    "long_document",
    "general",
)


@dataclass
class Provider:
    """An upstream API that serves one or more models."""

    name: str
    base_url: str
    style: str = "openai"  # "openai" (chat/completions) or "anthropic" (messages)
    api_key_env: tuple[str, ...] = ()
    api_key: str | None = None  # explicit key wins over env vars
    requires_key: bool = True
    headers: dict[str, str] = field(default_factory=dict)

    def resolve_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        for var in self.api_key_env:
            value = os.environ.get(var)
            if value:
                return value
        return None

    @property
    def configured(self) -> bool:
        return not self.requires_key or bool(self.resolve_key())


@dataclass
class Model:
    """One routable model. Costs are USD per 1M tokens."""

    id: str
    provider: str
    name: str = ""
    provider_model_id: str | None = None
    input_cost: float = 0.0
    output_cost: float = 0.0
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    tools: bool = True
    vision: bool = False
    json_mode: bool = True
    latency_ms: int = 1_000
    quality: dict[str, float] = field(default_factory=dict)
    drop_params: tuple[str, ...] = ()
    enabled: bool = True

    @property
    def upstream_id(self) -> str:
        return self.provider_model_id or self.id

    def quality_for(self, task: str) -> float:
        q = self.quality
        return float(q.get(task, q.get("general", 0.5)))

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_cost + output_tokens * self.output_cost) / 1_000_000

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Model:
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in d.items() if k in known}
        if "drop_params" in data:
            data["drop_params"] = tuple(data["drop_params"])
        if not data.get("name"):
            data["name"] = data["id"]
        return cls(**data)


def builtin_providers() -> dict[str, Provider]:
    return {
        "openai": Provider(
            "openai",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key_env=("OPENAI_API_KEY",),
        ),
        "anthropic": Provider(
            "anthropic",
            os.environ.get("FLUX_OS_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            style="anthropic",
            api_key_env=("ANTHROPIC_API_KEY",),
        ),
        "google": Provider(
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            api_key_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        ),
        "groq": Provider("groq", "https://api.groq.com/openai/v1", api_key_env=("GROQ_API_KEY",)),
        "mistral": Provider("mistral", "https://api.mistral.ai/v1", api_key_env=("MISTRAL_API_KEY",)),
        "deepseek": Provider("deepseek", "https://api.deepseek.com/v1", api_key_env=("DEEPSEEK_API_KEY",)),
        "openrouter": Provider(
            "openrouter", "https://openrouter.ai/api/v1", api_key_env=("OPENROUTER_API_KEY",)
        ),
        "ollama": Provider(
            "ollama",
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            requires_key=False,
        ),
    }


def builtin_models() -> list[Model]:
    text = resources.files("flux_os").joinpath("models.json").read_text(encoding="utf-8")
    return [Model.from_dict(m) for m in json.loads(text)["models"]]


@dataclass
class Catalog:
    providers: dict[str, Provider]
    models: list[Model]

    @classmethod
    def load(cls, api_keys: dict[str, str] | None = None) -> Catalog:
        """Built-in providers and models, plus explicit keys."""
        providers = builtin_providers()
        models = builtin_models()

        disabled = {s.strip() for s in os.environ.get("FLUX_OS_DISABLE", "").split(",") if s.strip()}
        for m in models:
            if m.id in disabled or f"provider:{m.provider}" in disabled:
                m.enabled = False
        allow = {s.strip() for s in os.environ.get("FLUX_OS_ONLY_MODELS", "").split(",") if s.strip()}
        if allow:
            for m in models:
                if m.id not in allow and f"provider:{m.provider}" not in allow:
                    m.enabled = False

        for name, key in (api_keys or {}).items():
            if name in providers:
                providers[name].api_key = key

        return cls(providers=providers, models=models)

    def get(self, model_id: str) -> Model | None:
        for m in self.models:
            if m.id == model_id or m.provider_model_id == model_id:
                return m
        return None

    def available(self) -> list[Model]:
        """Enabled models whose provider has credentials configured."""
        return [m for m in self.models if m.enabled and self.providers[m.provider].configured]

    def resolve_passthrough(self, model: str) -> Model | None:
        """Build an ad-hoc Model for a name not in the catalog.

        ``provider/model`` picks the provider explicitly (``ollama/llama3.2``,
        ``groq/llama-3.3-70b-versatile``); otherwise well-known prefixes are used.
        """
        if "/" in model:
            prov, _, rest = model.partition("/")
            if prov in self.providers and rest:
                return Model(id=model, provider=prov, provider_model_id=rest, name=model)
        lowered = model.lower()
        guesses = (
            (("gpt-", "o1", "o3", "o4", "chatgpt-"), "openai"),
            (("claude-",), "anthropic"),
            (("gemini-", "gemma-"), "google"),
            (("mistral-", "codestral", "magistral", "ministral", "pixtral"), "mistral"),
            (("deepseek-",), "deepseek"),
        )
        for prefixes, prov in guesses:
            if lowered.startswith(prefixes):
                drop = (
                    ("temperature", "top_p")
                    if prov == "openai" and lowered.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))
                    else ()
                )
                return Model(id=model, provider=prov, name=model, drop_params=drop)
        return None
