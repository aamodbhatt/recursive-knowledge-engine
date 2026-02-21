from datetime import datetime
import re
from threading import Lock
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from app.rlm import RewardLearningMemory
from app.schemas import (
    AblationResult,
    FailureCase,
    FailureLoopSnapshot,
    HistoryItem,
    LoopResult,
    QueryResponse,
    RetrievedChunk,
)
from app.text_utils import tokenize, unique_tokens


NEGATION_MARKERS = {
    "not",
    "no",
    "never",
    "without",
    "except",
    "however",
    "although",
    "but",
    "contrary",
    "conflict",
    "limitation",
    "limitations",
    "uncertain",
    "risk",
    "risks",
    "fails",
    "failure",
}


class SessionHistoryStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._items: Dict[str, List[HistoryItem]] = {}

    def add(self, session_id: str, item: HistoryItem) -> None:
        with self._lock:
            self._items.setdefault(session_id, []).append(item)

    def get(self, session_id: str) -> List[HistoryItem]:
        with self._lock:
            return list(self._items.get(session_id, []))

    def get_summary(self, session_id: str) -> Dict[str, int]:
        with self._lock:
            items = self._items.get(session_id, [])
            return {
                "total_queries": len(items),
                "estimated_tokens_used": sum(item.estimated_tokens_used for item in items),
            }

    def get_global_summary(self) -> Dict[str, int]:
        with self._lock:
            total_queries = 0
            estimated_tokens_used = 0
            for items in self._items.values():
                total_queries += len(items)
                estimated_tokens_used += sum(item.estimated_tokens_used for item in items)
            return {
                "total_queries": total_queries,
                "estimated_tokens_used": estimated_tokens_used,
            }


class FailureCaseStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._items: List[FailureCase] = []

    def add(self, item: FailureCase) -> None:
        with self._lock:
            self._items.append(item)

    def list(self, session_id: Optional[str] = None) -> List[FailureCase]:
        with self._lock:
            if not session_id:
                return list(self._items)
            return [item for item in self._items if item.session_id == session_id]


class RecursiveController:
    def __init__(
        self,
        retriever,
        answerer,
        critic,
        history_store: SessionHistoryStore,
        failure_store: Optional[FailureCaseStore] = None,
        rlm: Optional[RewardLearningMemory] = None,
        early_stop_confidence: float = 0.86,
        coverage_warning_threshold: float = 0.34,
        deterministic_seed: int = 17,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.critic = critic
        self.history_store = history_store
        self.failure_store = failure_store or FailureCaseStore()
        self.rlm = rlm or RewardLearningMemory()
        self.early_stop_confidence = early_stop_confidence
        self.coverage_warning_threshold = coverage_warning_threshold
        self.deterministic_seed = deterministic_seed
        self._runtime_lock = Lock()
        self._last_model_fallback_used = False

    @staticmethod
    def _retrieval_confidence(retrieved_chunks: List[Dict]) -> float:
        if not retrieved_chunks:
            return 0.0
        score = sum(item["score"] for item in retrieved_chunks) / len(retrieved_chunks)
        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def _source_coverage(answer: str, retrieved_chunks: List[Dict]) -> float:
        answer_tokens = unique_tokens(answer)
        if not answer_tokens:
            return 0.0

        source_tokens = set()
        for chunk in retrieved_chunks:
            source_tokens |= unique_tokens(chunk["text"])
        if not source_tokens:
            return 0.0

        overlap = len(answer_tokens & source_tokens) / len(answer_tokens)
        return round(max(0.0, min(1.0, overlap)), 3)

    @staticmethod
    def _confidence_drop_points(loops: List[LoopResult]) -> List[int]:
        points: List[int] = []
        for idx in range(1, len(loops)):
            if loops[idx].confidence < loops[idx - 1].confidence:
                points.append(loops[idx].iteration)
        return points

    @staticmethod
    def _answer_sentences(answer: str) -> List[str]:
        chunks = re.split(r"[\n\r]+|(?<=[.!?])\s+", answer.strip())
        return [segment.strip(" -\t") for segment in chunks if len(segment.strip()) >= 12]

    @staticmethod
    def _groundedness(answer: str, retrieved_chunks: List[Dict]) -> Tuple[float, float]:
        sentences = RecursiveController._answer_sentences(answer)
        if not sentences:
            return 0.0, 0.0
        if not retrieved_chunks:
            return 0.0, 1.0

        chunk_token_sets = [set(tokenize(chunk["text"])) for chunk in retrieved_chunks if chunk.get("text")]
        if not chunk_token_sets:
            return 0.0, 1.0

        supported_sentences = 0
        overlaps: List[float] = []
        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            if len(sentence_tokens) < 4:
                continue
            best_overlap = 0.0
            for source_tokens in chunk_token_sets:
                if not source_tokens:
                    continue
                overlap = len(sentence_tokens & source_tokens) / max(1, len(sentence_tokens))
                if overlap > best_overlap:
                    best_overlap = overlap
            overlaps.append(best_overlap)
            if best_overlap >= 0.35:
                supported_sentences += 1

        if not overlaps:
            return 0.0, 1.0

        groundedness = round(sum(overlaps) / len(overlaps), 3)
        unsupported_ratio = round(
            max(0.0, min(1.0, 1 - (supported_sentences / max(1, len(overlaps))))),
            3,
        )
        return groundedness, unsupported_ratio

    @staticmethod
    def _build_challenge_query(question: str) -> str:
        return (
            f"{question} contradictions caveats limitations counterexamples disagreements"
        ).strip()

    @staticmethod
    def _support_dependency(
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[float, float]:
        answer_tokens = {token for token in tokenize(answer) if len(token) >= 4}
        if not answer_tokens or not retrieved_chunks:
            return 0.0, 0.0

        source_token_sets: Dict[str, Set[str]] = {}
        for chunk in retrieved_chunks:
            source_token_sets.setdefault(chunk.source, set()).update(tokenize(chunk.text))

        support_counts: List[int] = []
        for token in answer_tokens:
            sources_with_token = 0
            for token_set in source_token_sets.values():
                if token in token_set:
                    sources_with_token += 1
            support_counts.append(sources_with_token)

        if not support_counts:
            return 0.0, 0.0

        redundancy = sum(1 for count in support_counts if count >= 2) / len(support_counts)
        single_source_dependency = sum(1 for count in support_counts if count == 1) / len(support_counts)
        return round(redundancy, 3), round(single_source_dependency, 3)

    @staticmethod
    def _challenge_risk_score(
        answer: str,
        challenge_chunks: List[RetrievedChunk],
    ) -> Tuple[float, str]:
        answer_tokens = set(tokenize(answer))
        if not answer_tokens or not challenge_chunks:
            return 0.0, "No challenge evidence available."

        chunk_risks: List[float] = []
        marker_hits = 0
        for chunk in challenge_chunks:
            chunk_tokens = set(tokenize(chunk.text))
            overlap = len(answer_tokens & chunk_tokens) / max(1, len(answer_tokens))
            marker_count = sum(1 for marker in NEGATION_MARKERS if marker in chunk_tokens)
            if marker_count > 0:
                marker_hits += 1
            marker_score = min(1.0, marker_count / 3)
            chunk_risk = (0.65 * overlap) + (0.35 * marker_score)
            chunk_risks.append(chunk_risk)

        aggregate_risk = round(max(chunk_risks), 3)
        marker_ratio = marker_hits / max(1, len(challenge_chunks))
        summary = (
            f"Challenge retrieval found {marker_hits}/{len(challenge_chunks)} chunks with contradiction markers; "
            f"max overlap-adjusted risk={aggregate_risk:.3f}."
        )
        if marker_ratio > 0.5 and aggregate_risk > 0.35:
            summary = (
                f"Challenge retrieval surfaced multiple potentially conflicting chunks; "
                f"max overlap-adjusted risk={aggregate_risk:.3f}."
            )
        return aggregate_risk, summary

    @staticmethod
    def _reliability_grade(
        challenge_risk: float,
        support_redundancy: float,
        single_source_dependency: float,
        low_coverage_warning: bool,
        model_fallback_used: bool,
    ) -> str:
        if model_fallback_used or low_coverage_warning:
            return "low"
        if challenge_risk >= 0.62 or single_source_dependency >= 0.55:
            return "low"
        if challenge_risk >= 0.38 or support_redundancy < 0.28:
            return "medium"
        return "high"

    @staticmethod
    def _detect_failure(loops: List[LoopResult]) -> Tuple[bool, Optional[str]]:
        if len(loops) < 2:
            return False, None

        first = loops[0]
        final = loops[-1]
        best_confidence = max(loop.confidence for loop in loops)
        confidence_drop = final.confidence < (first.confidence - 0.08)
        coverage_drop = final.source_coverage < (first.source_coverage - 0.1)
        no_net_gain = best_confidence <= (first.confidence + 0.01)

        if confidence_drop and (no_net_gain or coverage_drop):
            return True, "Refinement decreased confidence without improving final quality."
        if coverage_drop and final.confidence < first.confidence:
            return True, "Refinement reduced source coverage and confidence."
        return False, None

    def _record_failure(
        self,
        session_id: str,
        question: str,
        depth: int,
        failure_reason: str,
        loops: List[LoopResult],
        created_at: datetime,
    ) -> None:
        if not loops:
            return

        snapshots = [
            FailureLoopSnapshot(
                iteration=loop.iteration,
                query=loop.query,
                confidence=loop.confidence,
                source_coverage=loop.source_coverage,
            )
            for loop in loops
        ]

        case = FailureCase(
            id=str(uuid4()),
            session_id=session_id,
            question=question,
            depth=depth,
            reason=failure_reason,
            created_at=created_at,
            first_confidence=loops[0].confidence,
            final_confidence=loops[-1].confidence,
            best_confidence=max(loop.confidence for loop in loops),
            first_coverage=loops[0].source_coverage,
            final_coverage=loops[-1].source_coverage,
            loops=snapshots,
        )
        self.failure_store.add(case)

    def _run_single_depth(
        self,
        question: str,
        top_k: int,
        max_iterations: int,
        source_filters: Optional[List[str]],
        answer_verbosity: str,
        fast_mode: bool,
        deterministic_mode: bool,
        seed: Optional[int],
    ) -> Dict:
        run_start = perf_counter()
        current_query = question
        loops: List[LoopResult] = []
        best_answer = ""
        best_confidence = -1.0
        best_iteration = 1
        best_coverage = 0.0
        stop_reason: Optional[str] = None
        estimated_tokens_used = 0
        model_fallback_used = False
        rlm_bias_applied = False
        min_iterations = 2 if max_iterations > 1 else 1

        for iteration in range(1, max_iterations + 1):
            loop_start = perf_counter()
            retrieved = self.retriever.search(
                current_query,
                top_k=top_k,
                source_filters=source_filters,
            )
            retrieved, bias_applied = self.rlm.apply_source_bias(retrieved)
            rlm_bias_applied = rlm_bias_applied or bias_applied

            answer_payload = self.answerer.generate_answer(
                question,
                retrieved,
                answer_verbosity=answer_verbosity,
                deterministic_mode=deterministic_mode,
                seed=seed,
            )
            answer = str(answer_payload["text"]).strip()
            estimated_tokens_used += int(answer_payload.get("estimated_tokens", 0))
            model_fallback_used = model_fallback_used or bool(answer_payload.get("used_fallback", False))

            critique_data = self.critic.evaluate(
                original_question=question,
                current_query=current_query,
                answer=answer,
                contexts=retrieved,
                fast_mode=fast_mode,
                deterministic_mode=deterministic_mode,
                seed=seed,
            )
            estimated_tokens_used += int(critique_data.get("estimated_tokens", 0))
            model_fallback_used = model_fallback_used or bool(critique_data.get("used_fallback", False))

            retrieval_confidence = self._retrieval_confidence(retrieved)
            critic_confidence = float(critique_data["confidence"])
            groundedness, unsupported_claim_ratio = self._groundedness(answer, retrieved)
            combined_confidence = round(
                max(
                    0.0,
                    min(
                        1.0,
                        (0.45 * critic_confidence)
                        + (0.35 * retrieval_confidence)
                        + (0.20 * groundedness),
                    ),
                ),
                3,
            )
            if unsupported_claim_ratio > 0.55:
                combined_confidence = round(max(0.0, combined_confidence - 0.1), 3)
            if answer.startswith("Using a local fallback summary"):
                combined_confidence = round(max(0.0, combined_confidence - 0.12), 3)
            source_coverage = self._source_coverage(answer, retrieved)

            previous_confidence = loops[-1].confidence if loops else None
            confidence_delta = (
                round(combined_confidence - previous_confidence, 3)
                if previous_confidence is not None
                else None
            )

            loop_record = LoopResult(
                iteration=iteration,
                query=current_query,
                answer=answer,
                critique=critique_data["critique"],
                refined_query=critique_data["refined_query"],
                confidence=combined_confidence,
                retrieval_confidence=retrieval_confidence,
                groundedness=groundedness,
                unsupported_claim_ratio=unsupported_claim_ratio,
                source_coverage=source_coverage,
                confidence_delta=confidence_delta,
                duration_ms=int((perf_counter() - loop_start) * 1000),
                retrieved_chunks=[RetrievedChunk(**item) for item in retrieved],
            )
            loops.append(loop_record)

            answer_non_empty = bool(answer.strip())
            best_answer_non_empty = bool(best_answer.strip())
            should_promote_for_non_empty = answer_non_empty and not best_answer_non_empty
            should_promote_for_score = (combined_confidence > best_confidence) or (
                combined_confidence == best_confidence and source_coverage > best_coverage
            )

            if should_promote_for_non_empty or should_promote_for_score:
                best_confidence = combined_confidence
                best_answer = answer
                best_iteration = iteration
                best_coverage = source_coverage

            refined_query = critique_data["refined_query"].strip()
            if refined_query:
                current_query = refined_query

            if fast_mode and iteration >= min_iterations and max_iterations > 1:
                same_query = refined_query.lower() == loop_record.query.lower()
                if same_query and combined_confidence >= self.early_stop_confidence:
                    stop_reason = "high_confidence_stable_query"
                    break

        confidence_trajectory = [loop.confidence for loop in loops]
        confidence_drop_points = self._confidence_drop_points(loops)
        is_failure, failure_reason = self._detect_failure(loops)

        if not best_answer.strip():
            for loop in sorted(loops, key=lambda item: item.confidence, reverse=True):
                if loop.answer.strip():
                    best_answer = loop.answer
                    best_iteration = loop.iteration
                    best_confidence = loop.confidence
                    break
        if not best_answer.strip():
            top_chunks = []
            for loop in loops:
                top_chunks.extend(loop.retrieved_chunks[:2])
                if top_chunks:
                    break
            if top_chunks:
                bullet_lines = [f"- {chunk.text[:220].strip()}" for chunk in top_chunks[:2]]
                best_answer = (
                    "Model returned an empty answer. Using retrieved evidence fallback:\n\n"
                    + "\n".join(bullet_lines)
                )
            else:
                best_answer = "No answer text was generated. Try re-running with different docs or settings."

        answer_coverage = loops[best_iteration - 1].source_coverage if loops else 0.0

        return {
            "final_answer": best_answer,
            "confidence": round(best_confidence, 3),
            "loops": loops,
            "total_duration_ms": int((perf_counter() - run_start) * 1000),
            "stopped_early": stop_reason is not None,
            "stop_reason": stop_reason,
            "best_iteration": best_iteration,
            "answer_coverage": answer_coverage,
            "low_coverage_warning": answer_coverage < self.coverage_warning_threshold,
            "confidence_trajectory": confidence_trajectory,
            "confidence_decreased": len(confidence_drop_points) > 0,
            "confidence_drop_points": confidence_drop_points,
            "max_iterations_used": len(loops),
            "estimated_tokens_used": estimated_tokens_used,
            "model_fallback_used": model_fallback_used,
            "rlm_bias_applied": rlm_bias_applied,
            "is_failure": is_failure,
            "failure_reason": failure_reason,
        }

    def runtime_stats(self) -> Dict[str, int | bool]:
        summary = self.history_store.get_global_summary()
        rlm_stats = self.rlm.stats()
        with self._runtime_lock:
            last_model_fallback_used = self._last_model_fallback_used
        return {
            "total_queries": summary["total_queries"],
            "total_estimated_tokens": summary["estimated_tokens_used"],
            "last_model_fallback_used": last_model_fallback_used,
            "rlm_feedback_count": rlm_stats["total_feedback_count"],
            "rlm_tracked_sources": rlm_stats["tracked_sources"],
        }

    def get_failures(self, session_id: Optional[str] = None) -> List[FailureCase]:
        return self.failure_store.list(session_id=session_id)

    def record_feedback(self, response_id: str, rating: int, notes: Optional[str] = None) -> Dict:
        updated_sources = self.rlm.record_feedback(response_id=response_id, rating=rating, notes=notes)
        stats = self.rlm.stats()
        return {
            "response_id": response_id,
            "rating": rating,
            "updated_sources": updated_sources,
            "total_feedback_count": stats["total_feedback_count"],
        }

    def rlm_stats(self) -> Dict:
        return self.rlm.stats()

    def run(
        self,
        question: str,
        top_k: int = 5,
        max_iterations: int = 3,
        session_id: Optional[str] = None,
        fast_mode: bool = False,
        deterministic_mode: bool = False,
        ablation_mode: bool = False,
        ablation_depths: Optional[List[int]] = None,
        source_filters: Optional[List[str]] = None,
        answer_verbosity: str = "normal",
        challenge_mode: bool = True,
    ) -> QueryResponse:
        sid = session_id or str(uuid4())
        response_id = str(uuid4())
        created_at = datetime.utcnow()
        seed = self.deterministic_seed if deterministic_mode else None

        if ablation_mode:
            depths = sorted(set(ablation_depths or [1, 2, 3]))
        else:
            depths = [max_iterations]
        if source_filters and not self.retriever.has_sources(source_filters):
            raise RuntimeError(
                "Selected attachments are not indexed yet. Wait for indexing to complete, then try again."
            )

        run_results: List[Tuple[int, Dict]] = []
        for depth in depths:
            run_data = self._run_single_depth(
                question=question,
                top_k=top_k,
                max_iterations=depth,
                source_filters=source_filters,
                answer_verbosity=answer_verbosity,
                fast_mode=fast_mode,
                deterministic_mode=deterministic_mode,
                seed=seed,
            )
            run_results.append((depth, run_data))

            if run_data["is_failure"] and run_data["failure_reason"]:
                self._record_failure(
                    session_id=sid,
                    question=question,
                    depth=depth,
                    failure_reason=run_data["failure_reason"],
                    loops=run_data["loops"],
                    created_at=created_at,
                )

        best_depth, best_run = max(
            run_results,
            key=lambda item: (
                item[1]["confidence"],
                item[1]["answer_coverage"],
                -item[1]["total_duration_ms"],
            ),
        )

        ablation_results = None
        if ablation_mode:
            ablation_results = [
                AblationResult(
                    depth=depth,
                    final_answer=data["final_answer"],
                    confidence=data["confidence"],
                    total_duration_ms=data["total_duration_ms"],
                    answer_coverage=data["answer_coverage"],
                    best_iteration=data["best_iteration"],
                    stopped_early=data["stopped_early"],
                    stop_reason=data["stop_reason"],
                    estimated_tokens_used=data["estimated_tokens_used"],
                    model_fallback_used=data["model_fallback_used"],
                    is_failure=data["is_failure"],
                    loops=data["loops"],
                    is_best=depth == best_depth,
                )
                for depth, data in run_results
            ]

        total_estimated_tokens = (
            sum(data["estimated_tokens_used"] for _, data in run_results)
            if ablation_mode
            else best_run["estimated_tokens_used"]
        )
        model_fallback_used = any(data["model_fallback_used"] for _, data in run_results)
        rlm_bias_applied = any(data.get("rlm_bias_applied", False) for _, data in run_results)

        challenge_query = None
        challenge_risk = 0.0
        support_redundancy = 0.0
        single_source_dependency = 0.0
        audit_summary = None
        challenge_chunks: List[RetrievedChunk] = []
        if best_run["loops"]:
            best_loop_index = max(0, min(len(best_run["loops"]) - 1, best_run["best_iteration"] - 1))
            best_loop = best_run["loops"][best_loop_index]
            support_redundancy, single_source_dependency = self._support_dependency(
                answer=best_run["final_answer"],
                retrieved_chunks=best_loop.retrieved_chunks,
            )
        if challenge_mode:
            challenge_query = self._build_challenge_query(question)
            challenge_hit_dicts = self.retriever.search(
                challenge_query,
                top_k=min(6, top_k + 1),
                source_filters=source_filters,
            )
            challenge_hit_dicts, _bias_applied = self.rlm.apply_source_bias(challenge_hit_dicts)
            challenge_chunks = [RetrievedChunk(**item) for item in challenge_hit_dicts]
            challenge_risk, audit_summary = self._challenge_risk_score(
                answer=best_run["final_answer"],
                challenge_chunks=challenge_chunks,
            )

        reliability_grade = self._reliability_grade(
            challenge_risk=challenge_risk,
            support_redundancy=support_redundancy,
            single_source_dependency=single_source_dependency,
            low_coverage_warning=best_run["low_coverage_warning"],
            model_fallback_used=model_fallback_used,
        )

        result = QueryResponse(
            response_id=response_id,
            session_id=sid,
            question=question,
            final_answer=best_run["final_answer"],
            confidence=best_run["confidence"],
            loops=best_run["loops"],
            total_duration_ms=best_run["total_duration_ms"],
            stopped_early=best_run["stopped_early"],
            stop_reason=best_run["stop_reason"],
            best_iteration=best_run["best_iteration"],
            answer_coverage=best_run["answer_coverage"],
            low_coverage_warning=best_run["low_coverage_warning"],
            confidence_trajectory=best_run["confidence_trajectory"],
            confidence_decreased=best_run["confidence_decreased"],
            confidence_drop_points=best_run["confidence_drop_points"],
            deterministic_mode=deterministic_mode,
            answer_verbosity=answer_verbosity,
            fast_mode=fast_mode,
            max_iterations_used=best_run["max_iterations_used"],
            estimated_tokens_used=total_estimated_tokens,
            model_fallback_used=model_fallback_used,
            rlm_bias_applied=rlm_bias_applied,
            challenge_mode=challenge_mode,
            challenge_query=challenge_query,
            challenge_risk=challenge_risk,
            support_redundancy=support_redundancy,
            single_source_dependency=single_source_dependency,
            reliability_grade=reliability_grade,
            audit_summary=audit_summary,
            challenge_chunks=challenge_chunks,
            is_failure=best_run["is_failure"],
            failure_reason=best_run["failure_reason"],
            ablation_results=ablation_results,
            ablation_best_depth=best_depth if ablation_mode else None,
            created_at=created_at,
        )

        self.history_store.add(
            sid,
            HistoryItem(
                response_id=response_id,
                question=question,
                final_answer=result.final_answer,
                confidence=result.confidence,
                estimated_tokens_used=total_estimated_tokens,
                model_fallback_used=model_fallback_used,
                created_at=created_at,
            ),
        )
        if result.loops:
            best_index = max(0, min(len(result.loops) - 1, result.best_iteration - 1))
            best_loop_sources = [chunk.source for chunk in result.loops[best_index].retrieved_chunks]
        else:
            best_loop_sources = []
        self.rlm.register_response(
            response_id=response_id,
            session_id=sid,
            question=question,
            final_answer=result.final_answer,
            sources=best_loop_sources,
        )
        with self._runtime_lock:
            self._last_model_fallback_used = model_fallback_used

        return result
