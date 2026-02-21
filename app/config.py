import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _parse_model_list(raw: str) -> tuple[str, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(values)


def _clean_env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()


def _normalize_provider(raw: str) -> str:
    normalized = str(raw or "auto").strip().lower()
    if normalized in {"auto", "openrouter", "openai", "groq", "custom"}:
        return normalized
    return "auto"


def _infer_provider(provider: str, api_key: str, base_url: str) -> str:
    normalized = _normalize_provider(provider)
    if normalized != "auto":
        return normalized

    normalized_key = api_key.strip().lower()
    if normalized_key.startswith("sk-or-"):
        return "openrouter"
    if normalized_key.startswith("gsk_"):
        return "groq"
    if normalized_key.startswith("sk-"):
        return "openai"

    lowered_url = base_url.lower()
    if "openrouter.ai" in lowered_url:
        return "openrouter"
    if "api.openai.com" in lowered_url:
        return "openai"
    if "api.groq.com" in lowered_url:
        return "groq"
    return "custom"


def _default_base_url(provider: str) -> str:
    defaults = {
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
    }
    return defaults.get(provider, "https://openrouter.ai/api/v1/chat/completions")


def _default_model(provider: str) -> str:
    defaults = {
        "openrouter": "openrouter/free",
        "openai": "gpt-4o-mini",
        "groq": "llama-3.1-8b-instant",
    }
    return defaults.get(provider, "openrouter/free")


def _normalize_base_url(base_url: str, provider: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        return normalized

    if provider not in {"openrouter", "openai", "groq"}:
        return normalized
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1") or normalized.endswith("/api/v1") or normalized.endswith("/openai/v1"):
        return f"{normalized}/chat/completions"
    return normalized


def _parse_int_env(primary: str, secondary: str, default: int) -> int:
    raw = _clean_env(primary) or _clean_env(secondary, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_llm_settings() -> dict:
    provider_input = _clean_env("LLM_PROVIDER", "auto")
    api_key = _clean_env("LLM_API_KEY") or _clean_env("OPENROUTER_API_KEY")
    llm_model = _clean_env("LLM_MODEL")
    legacy_model = _clean_env("OPENROUTER_MODEL")
    llm_base_url = _clean_env("LLM_BASE_URL")
    legacy_openrouter_url = _clean_env("OPENROUTER_URL")
    provider_probe_url = llm_base_url or legacy_openrouter_url

    provider = _infer_provider(provider_input, api_key, provider_probe_url)
    if llm_model:
        model = llm_model
    elif provider == "openrouter" and legacy_model:
        model = legacy_model
    else:
        model = _default_model(provider)

    if llm_base_url:
        base_url_input = llm_base_url
    elif provider == "openrouter" and legacy_openrouter_url:
        base_url_input = legacy_openrouter_url
    else:
        base_url_input = ""

    base_url = _normalize_base_url(base_url_input, provider)
    if not base_url:
        # Default to OpenRouter when custom provider lacks explicit URL to preserve legacy behavior.
        base_url = _default_base_url(provider if provider != "custom" else "openrouter")

    default_fallbacks = (
        "meta-llama/llama-3.1-8b-instruct:free,qwen/qwen-2.5-7b-instruct:free,deepseek/deepseek-chat:free"
        if provider == "openrouter"
        else ""
    )
    llm_fallback = _clean_env("LLM_FALLBACK_MODELS")
    if llm_fallback:
        fallback_raw = llm_fallback
    elif provider == "openrouter":
        fallback_raw = _clean_env("OPENROUTER_FALLBACK_MODELS", default_fallbacks)
    else:
        fallback_raw = ""

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "fallback_models": _parse_model_list(fallback_raw),
        "timeout_seconds": _parse_int_env("LLM_TIMEOUT_SECONDS", "OPENROUTER_TIMEOUT_SECONDS", 45),
        "max_retries": _parse_int_env("LLM_MAX_RETRIES", "OPENROUTER_MAX_RETRIES", 2),
    }


_LLM_SETTINGS = _resolve_llm_settings()


@dataclass(frozen=True)
class Settings:
    app_name: str = "Recursive Knowledge Engine"
    llm_provider: str = _LLM_SETTINGS["provider"]
    llm_api_key: str = _LLM_SETTINGS["api_key"]
    llm_model: str = _LLM_SETTINGS["model"]
    llm_base_url: str = _LLM_SETTINGS["base_url"]
    llm_fallback_models: tuple[str, ...] = _LLM_SETTINGS["fallback_models"]
    llm_timeout_seconds: int = _LLM_SETTINGS["timeout_seconds"]
    llm_max_retries: int = _LLM_SETTINGS["max_retries"]
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vectorstore_dir: Path = BASE_DIR / "vectorstore"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    faiss_index_path: Path = vectorstore_dir / "faiss.index"
    metadata_path: Path = vectorstore_dir / "chunks.json"
    chunk_size_words: int = 220
    chunk_overlap_words: int = 40
    query_cache_size: int = int(os.getenv("QUERY_CACHE_SIZE", "256"))
    early_stop_confidence: float = float(os.getenv("EARLY_STOP_CONFIDENCE", "0.86"))
    deterministic_seed: int = int(os.getenv("DETERMINISTIC_SEED", "17"))
    coverage_warning_threshold: float = float(os.getenv("COVERAGE_WARNING_THRESHOLD", "0.34"))
    frontend_dist_dir: Path = BASE_DIR / "frontend" / "dist"

    @property
    def openrouter_api_key(self) -> str:
        # Backward-compatible alias for older modules/scripts.
        return self.llm_api_key

    @property
    def openrouter_model(self) -> str:
        return self.llm_model

    @property
    def openrouter_fallback_models(self) -> tuple[str, ...]:
        return self.llm_fallback_models

    @property
    def openrouter_url(self) -> str:
        return self.llm_base_url

    @property
    def openrouter_timeout_seconds(self) -> int:
        return self.llm_timeout_seconds

    @property
    def openrouter_max_retries(self) -> int:
        return self.llm_max_retries


settings = Settings()
settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
