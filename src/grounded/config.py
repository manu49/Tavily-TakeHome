"""Runtime configuration and model construction.

Everything tunable lives here so the pipeline reads as a clean sequence of
steps. Provider and model are resolved from the environment, defaulting to
Anthropic Claude but degrading gracefully to the starter's Nebius stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Latest Claude model IDs (see Anthropic models catalogue).
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_NEBIUS_MODEL = "openai/gpt-oss-120b"

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """All knobs for one run of the pipeline."""

    provider: str = "anthropic"  # "anthropic" | "nebius"
    model: str = DEFAULT_ANTHROPIC_MODEL
    judge_model: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = 4096

    # Retrieval
    max_subqueries: int = 4
    results_per_query: int = 6
    sources_to_extract: int = 6
    max_per_domain: int = 2
    search_depth: str = "advanced"  # "basic" | "advanced"
    extract_depth: str = "advanced"

    # Verification
    min_groundedness: float = 0.6  # claims below this are flagged / caveated

    tracing: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        provider = (os.getenv("GROUNDED_PROVIDER") or "").strip().lower()
        if not provider:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("NEBIUS_API_KEY"):
                provider = "nebius"
            else:
                provider = "anthropic"  # fail later with a clear message

        default_model = (
            DEFAULT_ANTHROPIC_MODEL if provider == "anthropic" else DEFAULT_NEBIUS_MODEL
        )
        model = os.getenv("GROUNDED_MODEL") or default_model
        return cls(
            provider=provider,
            model=model,
            judge_model=os.getenv("GROUNDED_JUDGE_MODEL") or model,
            tracing=os.getenv("GROUNDED_TRACING", "1").strip().lower() in _TRUTHY,
        )

    def missing_credentials(self) -> list[str]:
        """Return the env vars required to run that are not set."""
        missing: list[str] = []
        if not os.getenv("TAVILY_API_KEY"):
            missing.append("TAVILY_API_KEY")
        if self.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if self.provider == "nebius" and not os.getenv("NEBIUS_API_KEY"):
            missing.append("NEBIUS_API_KEY")
        return missing


def build_chat_model(
    settings: Settings,
    *,
    for_judge: bool = False,
    max_tokens: int | None = None,
):
    """Construct a LangChain chat model for the configured provider.

    Provider SDKs are imported lazily so installing only one provider's
    package is enough to run.
    """
    name = settings.judge_model if for_judge else settings.model
    tokens = max_tokens or settings.max_tokens

    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=name, max_tokens=tokens, timeout=120, max_retries=2
        )
    if settings.provider == "nebius":
        from langchain_nebius import ChatNebius

        return ChatNebius(model=name, max_tokens=tokens)

    raise ValueError(f"Unknown provider: {settings.provider!r}")
