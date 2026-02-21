from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.llm_provider import detect_provider, provider_label, supports_openrouter_discovery
from app.text_utils import estimate_tokens


class Answerer:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        provider: str = "auto",
        timeout_seconds: int = 45,
        max_retries: int = 2,
        fallback_models: Optional[List[str]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider = detect_provider(provider, api_key, base_url)
        self.provider_name = provider_label(self.provider)
        self.timeout_seconds = timeout_seconds
        self.fallback_models = [item.strip() for item in (fallback_models or []) if item.strip()]
        self.discovered_fallback_models: List[str] = []
        self.last_fallback_used = False
        self.last_estimated_tokens = 0
        self.session = requests.Session()

        retries = Retry(
            total=max_retries,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _discover_free_models(self) -> List[str]:
        if self.discovered_fallback_models:
            return self.discovered_fallback_models

        if not self.api_key:
            return []
        if not supports_openrouter_discovery(self.provider, self.base_url):
            return []

        try:
            response = self.session.get(
                "https://openrouter.ai/api/v1/models",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=min(self.timeout_seconds, 20),
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data", [])
            free_ids: List[str] = []
            for model_data in models:
                model_id = str(model_data.get("id", "")).strip()
                if ":free" not in model_id:
                    continue
                if model_id == self.model:
                    continue
                free_ids.append(model_id)
                if len(free_ids) >= 6:
                    break
            self.discovered_fallback_models = free_ids
        except Exception:
            self.discovered_fallback_models = []

        return self.discovered_fallback_models

    def _candidate_models(self) -> List[str]:
        candidate_models = [self.model, *self.fallback_models, *self._discover_free_models()]
        deduped: List[str] = []
        seen = set()
        for model_name in candidate_models:
            model_name = model_name.strip()
            if not model_name or model_name in seen:
                continue
            deduped.append(model_name)
            seen.add(model_name)
        return deduped

    def _build_prompt(self, question: str, contexts: List[Dict], answer_verbosity: str = "normal") -> str:
        if contexts:
            context_block = "\n\n".join(
                [
                    f"[Source: {item['source']} | Score: {item['score']}]\n{item['text']}"
                    for item in contexts
                ]
            )
        else:
            context_block = "No context retrieved."

        verbosity = {
            "short": "Keep the response compact (3-5 bullets max).",
            "normal": "Provide a concise but complete answer.",
            "long": "Provide a detailed, structured answer with sections and thorough explanation.",
        }.get(str(answer_verbosity).strip().lower(), "Provide a concise but complete answer.")

        return (
            "Answer using only retrieved context. "
            "If context is incomplete, explicitly say what is missing.\n"
            "Do not invent facts not present in the context.\n"
            "Cite supporting sources inline as [source filename] for factual claims.\n"
            f"{verbosity}\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context_block}\n\n"
            "Return markdown."
        )

    @staticmethod
    def _fallback_answer(
        question: str,
        contexts: List[Dict],
        reason: Optional[str] = None,
        provider_name: str = "LLM provider",
    ) -> str:
        if reason == "auth":
            return (
                f"{provider_name} authentication failed (401/403). "
                "Update LLM_API_KEY in .env (or OPENROUTER_API_KEY for legacy setup), "
                "then restart the backend."
            )
        if reason == "missing_key":
            return (
                "No LLM API key is configured. "
                "Set LLM_API_KEY in .env (or OPENROUTER_API_KEY for legacy setup). "
                "If using a non-default provider, also set LLM_BASE_URL and LLM_MODEL. "
                "Then restart the backend."
            )
        if not contexts:
            return (
                "I could not reach the LLM provider and no document context was retrieved. "
                "Try re-running in a minute or upload more relevant content."
            )

        top_contexts = contexts[:2]
        bullets = [f"- {item['text'][:240].strip()}" for item in top_contexts]
        return (
            "Using a local fallback summary because the LLM provider is unavailable.\n\n"
            f"Question: {question}\n"
            "Most relevant retrieved evidence:\n"
            + "\n".join(bullets)
        )

    def generate_answer(
        self,
        question: str,
        contexts: List[Dict],
        answer_verbosity: str = "normal",
        deterministic_mode: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
        prompt = self._build_prompt(question, contexts, answer_verbosity=answer_verbosity)
        prompt_tokens = estimate_tokens(prompt)
        verbosity_key = str(answer_verbosity).strip().lower()
        max_tokens = {
            "short": 320,
            "normal": 720,
            "long": 1400,
        }.get(verbosity_key, 720)

        if not self.api_key:
            fallback = self._fallback_answer(
                question,
                contexts,
                reason="missing_key",
                provider_name=self.provider_name,
            )
            total_tokens = prompt_tokens + estimate_tokens(fallback)
            self.last_fallback_used = True
            self.last_estimated_tokens = total_tokens
            return {
                "text": fallback,
                "used_fallback": True,
                "estimated_tokens": total_tokens,
            }

        provider_errors: List[str] = []
        auth_failure = False
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        model_candidates = self._candidate_models()

        for model_name in model_candidates:
            payload_base = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You answer questions over retrieved documents with high precision and grounded citations. "
                            "Never fabricate information."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0 if deterministic_mode else 0.72,
                "max_tokens": max_tokens,
            }
            attempt_payloads = []
            if deterministic_mode and seed is not None:
                with_seed = dict(payload_base)
                with_seed["seed"] = int(seed)
                attempt_payloads.append(("seed", with_seed))
                attempt_payloads.append(("noseed", payload_base))
            else:
                attempt_payloads.append(("default", payload_base))

            for attempt_label, payload in attempt_payloads:
                try:
                    response = self.session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()
                    body = response.json()
                    answer_text = body["choices"][0]["message"]["content"].strip()
                    total_tokens = prompt_tokens + estimate_tokens(answer_text)
                    self.last_fallback_used = False
                    self.last_estimated_tokens = total_tokens
                    return {
                        "text": answer_text,
                        "used_fallback": False,
                        "estimated_tokens": total_tokens,
                        "model_used": model_name,
                        "provider_errors": provider_errors,
                    }
                except Exception as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    if status_code in {401, 403}:
                        auth_failure = True
                    provider_errors.append(f"{model_name}[{attempt_label}]: {exc.__class__.__name__}")

        fallback = self._fallback_answer(
            question,
            contexts,
            reason="auth" if auth_failure else None,
            provider_name=self.provider_name,
        )
        total_tokens = prompt_tokens + estimate_tokens(fallback)
        self.last_fallback_used = True
        self.last_estimated_tokens = total_tokens
        return {
            "text": fallback,
            "used_fallback": True,
            "estimated_tokens": total_tokens,
            "provider_errors": provider_errors,
        }
