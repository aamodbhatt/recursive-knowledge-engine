from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class UploadResponse(BaseModel):
    filename: str
    stored_as: str
    chunks_added: int
    total_chunks: int


class BatchUploadItem(BaseModel):
    filename: str
    stored_as: str
    chunks_added: int
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    items: List[BatchUploadItem]
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    session_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=12)
    max_iterations: int = Field(default=2, ge=1, le=3)
    fast_mode: bool = True
    deterministic_mode: bool = False
    ablation_mode: bool = False
    ablation_depths: Optional[List[int]] = None
    source_filters: Optional[List[str]] = None
    answer_verbosity: str = Field(default="normal")
    challenge_mode: bool = True

    @validator("ablation_depths")
    def validate_ablation_depths(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return None
        cleaned = sorted({int(v) for v in value if int(v) in {1, 2, 3}})
        if not cleaned:
            raise ValueError("ablation_depths must include at least one value from [1, 2, 3].")
        return cleaned

    @validator("answer_verbosity")
    def validate_answer_verbosity(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"short", "normal", "long"}:
            raise ValueError("answer_verbosity must be one of: short, normal, long.")
        return normalized

    @validator("source_filters")
    def validate_source_filters(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = []
        seen = set()
        for item in value:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned.append(normalized)
        return cleaned or None


class RetrievedChunk(BaseModel):
    source: str
    text: str
    score: float
    chunk_id: int


class LoopResult(BaseModel):
    iteration: int
    query: str
    answer: str
    critique: str
    refined_query: str
    confidence: float
    retrieval_confidence: float
    groundedness: float = 0.0
    unsupported_claim_ratio: float = 0.0
    source_coverage: float = 0.0
    confidence_delta: Optional[float] = None
    duration_ms: int
    retrieved_chunks: List[RetrievedChunk]


class AblationResult(BaseModel):
    depth: int
    final_answer: str
    confidence: float
    total_duration_ms: int
    answer_coverage: float
    best_iteration: int
    stopped_early: bool
    stop_reason: Optional[str] = None
    estimated_tokens_used: int = 0
    model_fallback_used: bool = False
    is_failure: bool = False
    loops: List[LoopResult] = Field(default_factory=list)
    is_best: bool = False


class QueryResponse(BaseModel):
    response_id: str
    session_id: str
    question: str
    final_answer: str
    confidence: float
    loops: List[LoopResult]
    total_duration_ms: int
    stopped_early: bool
    stop_reason: Optional[str] = None
    best_iteration: int = 1
    answer_coverage: float = 0.0
    low_coverage_warning: bool = False
    confidence_trajectory: List[float] = Field(default_factory=list)
    confidence_decreased: bool = False
    confidence_drop_points: List[int] = Field(default_factory=list)
    deterministic_mode: bool = False
    answer_verbosity: str = "normal"
    fast_mode: bool = False
    max_iterations_used: int = 1
    estimated_tokens_used: int = 0
    model_fallback_used: bool = False
    rlm_bias_applied: bool = False
    challenge_mode: bool = False
    challenge_query: Optional[str] = None
    challenge_risk: float = 0.0
    support_redundancy: float = 0.0
    single_source_dependency: float = 0.0
    reliability_grade: str = "unknown"
    audit_summary: Optional[str] = None
    challenge_chunks: List[RetrievedChunk] = Field(default_factory=list)
    is_failure: bool = False
    failure_reason: Optional[str] = None
    ablation_results: Optional[List[AblationResult]] = None
    ablation_best_depth: Optional[int] = None
    created_at: datetime


class HistoryItem(BaseModel):
    response_id: str
    question: str
    final_answer: str
    confidence: float
    estimated_tokens_used: int = 0
    model_fallback_used: bool = False
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: str
    items: List[HistoryItem]
    total_queries: int = 0
    estimated_tokens_used: int = 0


class FailureLoopSnapshot(BaseModel):
    iteration: int
    query: str
    confidence: float
    source_coverage: float


class FailureCase(BaseModel):
    id: str
    session_id: str
    question: str
    depth: int
    reason: str
    created_at: datetime
    first_confidence: float
    final_confidence: float
    best_confidence: float
    first_coverage: float
    final_coverage: float
    loops: List[FailureLoopSnapshot] = Field(default_factory=list)


class FailureCasesResponse(BaseModel):
    count: int
    items: List[FailureCase]


class FeedbackRequest(BaseModel):
    response_id: str
    session_id: Optional[str] = None
    rating: int = Field(..., ge=-1, le=1)
    notes: Optional[str] = None

    @validator("rating")
    def validate_rating(cls, value: int) -> int:
        if value == 0:
            raise ValueError("rating must be -1 or 1.")
        return value


class FeedbackResponse(BaseModel):
    status: str
    response_id: str
    rating: int
    updated_sources: int
    total_feedback_count: int


class RLMStatsResponse(BaseModel):
    total_feedback_count: int
    tracked_sources: int
    top_positive_sources: List[str] = Field(default_factory=list)
    top_negative_sources: List[str] = Field(default_factory=list)
