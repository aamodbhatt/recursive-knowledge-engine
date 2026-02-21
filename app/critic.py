import json
import re
from collections import Counter
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.llm_provider import detect_provider, supports_openrouter_discovery
from app.text_utils import estimate_tokens

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "your",
    "over",
    "into",
    "have",
    "has",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "about",
    "does",
    "than",
}


class Critic:
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
        self.timeout_seconds = timeout_seconds
        self.fallback_models = [item.strip() for item in (fallback_models or []) if item.strip()]
        self.discovered_fallback_models: List[str] = []
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

    def _heuristic_confidence(self, contexts: List[Dict], answer: str) -> float:
        if not contexts:
            return 0.2
        avg_score = sum(c["score"] for c in contexts) / max(1, len(contexts))
        length_bonus = 0.1 if len(answer.split()) > 20 else 0.0
        return round(max(0.0, min(1.0, avg_score + length_bonus)), 3)

    @staticmethod
    def _parse_json(text: str) -> Dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _extract_keywords(contexts: List[Dict], limit: int = 4) -> List[str]:
        if not contexts:
            return []
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", " ".join(c["text"] for c in contexts[:2]))
        tokens = [w.lower() for w in words if w.lower() not in STOPWORDS]
        most_common = Counter(tokens).most_common(limit)
        return [token for token, _count in most_common]

    def _heuristic_refined_query(self, original_question: str, current_query: str, contexts: List[Dict]) -> str:
        keywords = self._extract_keywords(contexts)
        if not keywords:
            return current_query

        query_tokens = set(re.findall(r"[A-Za-z0-9\-]+", current_query.lower()))
        missing = [kw for kw in keywords if kw not in query_tokens]
        if not missing:
            return current_query
        return f"{original_question} focus on {' '.join(missing[:3])}".strip()

    def _fast_evaluate(
        self,
        original_question: str,
        current_query: str,
        answer: str,
        contexts: List[Dict],
        used_fallback: bool = False,
    ) -> Dict:
        refined_query = self._heuristic_refined_query(original_question, current_query, contexts)
        return {
            "critique": "Fast mode heuristic critique applied to reduce latency.",
            "refined_query": refined_query,
            "confidence": self._heuristic_confidence(contexts, answer),
            "used_fallback": used_fallback,
            "estimated_tokens": 0,
        }

    def evaluate(
        self,
        original_question: str,
        current_query: str,
        answer: str,
        contexts: List[Dict],
        fast_mode: bool = False,
        deterministic_mode: bool = False,
        seed: Optional[int] = None,
    ) -> Dict:
        if fast_mode:
            return self._fast_evaluate(original_question, current_query, answer, contexts)

        if not self.api_key:
            return self._fast_evaluate(
                original_question, current_query, answer, contexts, used_fallback=True
            )

        context_summary = "\n\n".join(
            [f"[{c['source']} | {c['score']}] {c['text'][:320]}" for c in contexts]
        )

        prompt = (
            "Evaluate the answer quality against retrieved context and suggest a better retrieval query.\n\n"
            f"Original user question:\n{original_question}\n\n"
            f"Current retrieval query:\n{current_query}\n\n"
            f"Retrieved context:\n{context_summary or 'No context'}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            "Return ONLY valid JSON:\n"
            '{"critique":"...","refined_query":"...","confidence":0.0}\n'
            "Confidence must be in [0,1]. Keep critique concise."
        )
        prompt_tokens = estimate_tokens(prompt)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for model_name in self._candidate_models():
            payload_base = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a strict retrieval QA critic."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0 if deterministic_mode else 0.22,
                "max_tokens": 220,
                "response_format": {"type": "json_object"},
            }
            attempt_payloads = []
            if deterministic_mode and seed is not None:
                with_seed = dict(payload_base)
                with_seed["seed"] = int(seed)
                attempt_payloads.append(with_seed)
                attempt_payloads.append(payload_base)
            else:
                attempt_payloads.append(payload_base)

            for payload in attempt_payloads:
                try:
                    response = self.session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()
                    body = response.json()
                    text = body["choices"][0]["message"]["content"]
                    parsed = self._parse_json(text)

                    critique = str(parsed.get("critique", "No critique generated.")).strip()
                    refined_query = str(parsed.get("refined_query", current_query)).strip()
                    confidence = float(parsed.get("confidence", self._heuristic_confidence(contexts, answer)))
                    confidence = round(max(0.0, min(1.0, confidence)), 3)

                    if not refined_query:
                        refined_query = current_query

                    estimated_tokens = prompt_tokens + estimate_tokens(text)
                    return {
                        "critique": critique,
                        "refined_query": refined_query,
                        "confidence": confidence,
                        "used_fallback": False,
                        "estimated_tokens": estimated_tokens,
                        "model_used": model_name,
                    }
                except Exception:
                    continue

        return self._fast_evaluate(
            original_question, current_query, answer, contexts, used_fallback=True
        )
