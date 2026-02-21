from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Tuple


@dataclass
class FeedbackEvent:
    response_id: str
    rating: int
    updated_sources: int
    notes: Optional[str] = None


class RewardLearningMemory:
    """
    Lightweight in-memory reward learner.
    - Tracks per-source rewards from human feedback.
    - Applies source-level score bias at retrieval time.
    """

    def __init__(self, learning_rate: float = 0.12, max_responses: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.max_responses = max_responses
        self._lock = Lock()
        self._source_weights: Dict[str, float] = {}
        self._responses: Dict[str, Dict] = {}
        self._response_order: List[str] = []
        self._feedback_events: List[FeedbackEvent] = []

    @staticmethod
    def _clamp_score(value: float) -> float:
        return round(max(0.0, min(1.0, value)), 4)

    @staticmethod
    def _bias_from_weight(weight: float) -> float:
        # Smooth bounded bias so feedback improves ranking without overwhelming cosine score.
        if weight == 0:
            return 0.0
        return 0.06 * (weight / (1.0 + abs(weight)))

    def register_response(
        self,
        response_id: str,
        session_id: str,
        question: str,
        final_answer: str,
        sources: List[str],
    ) -> None:
        with self._lock:
            self._responses[response_id] = {
                "session_id": session_id,
                "question": question,
                "final_answer": final_answer,
                "sources": sorted(set(sources)),
            }
            self._response_order.append(response_id)
            while len(self._response_order) > self.max_responses:
                oldest = self._response_order.pop(0)
                self._responses.pop(oldest, None)

    def apply_source_bias(self, retrieved_chunks: List[Dict]) -> Tuple[List[Dict], bool]:
        if not retrieved_chunks:
            return retrieved_chunks, False

        with self._lock:
            if not self._source_weights:
                return retrieved_chunks, False
            source_weights = dict(self._source_weights)

        biased_chunks: List[Dict] = []
        applied = False
        for chunk in retrieved_chunks:
            source = chunk.get("source", "")
            weight = source_weights.get(source, 0.0)
            bias = self._bias_from_weight(weight)
            if bias != 0:
                applied = True
            updated = dict(chunk)
            updated["score"] = self._clamp_score(float(chunk.get("score", 0.0)) + bias)
            biased_chunks.append(updated)

        biased_chunks.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return biased_chunks, applied

    def record_feedback(self, response_id: str, rating: int, notes: Optional[str] = None) -> int:
        if rating not in {-1, 1}:
            raise ValueError("rating must be -1 or 1.")

        with self._lock:
            response = self._responses.get(response_id)
            if response is None:
                raise KeyError(response_id)

            sources = response.get("sources", [])
            for source in sources:
                current = self._source_weights.get(source, 0.0)
                self._source_weights[source] = current + (self.learning_rate * rating)

            self._feedback_events.append(
                FeedbackEvent(
                    response_id=response_id,
                    rating=rating,
                    updated_sources=len(sources),
                    notes=notes,
                )
            )
            return len(sources)

    def stats(self) -> Dict:
        with self._lock:
            sorted_sources = sorted(
                self._source_weights.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            positive = [source for source, score in sorted_sources if score > 0][:5]
            negative = [source for source, score in reversed(sorted_sources) if score < 0][:5]
            return {
                "total_feedback_count": len(self._feedback_events),
                "tracked_sources": len(self._source_weights),
                "top_positive_sources": positive,
                "top_negative_sources": negative,
            }
