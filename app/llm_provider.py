from typing import Literal

ProviderName = Literal["openrouter", "openai", "groq", "custom"]


def detect_provider(provider: str, api_key: str, base_url: str) -> ProviderName:
    normalized = str(provider or "auto").strip().lower()
    if normalized in {"openrouter", "openai", "groq", "custom"}:
        return normalized  # type: ignore[return-value]

    lowered_key = str(api_key or "").strip().lower()
    if lowered_key.startswith("sk-or-"):
        return "openrouter"
    if lowered_key.startswith("gsk_"):
        return "groq"
    if lowered_key.startswith("sk-"):
        return "openai"

    lowered_url = str(base_url or "").lower()
    if "openrouter.ai" in lowered_url:
        return "openrouter"
    if "api.openai.com" in lowered_url:
        return "openai"
    if "api.groq.com" in lowered_url:
        return "groq"
    return "custom"


def supports_openrouter_discovery(provider: str, base_url: str) -> bool:
    if str(provider).lower() == "openrouter":
        return True
    return "openrouter.ai" in str(base_url or "").lower()


def provider_label(provider: str) -> str:
    normalized = str(provider or "").lower()
    labels = {
        "openrouter": "OpenRouter",
        "openai": "OpenAI-compatible provider",
        "groq": "Groq OpenAI-compatible provider",
        "custom": "Custom OpenAI-compatible provider",
    }
    return labels.get(normalized, "LLM provider")
