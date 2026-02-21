from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.answerer import Answerer
from app.config import settings
from app.controller import FailureCaseStore, RecursiveController, SessionHistoryStore
from app.critic import Critic
from app.rlm import RewardLearningMemory
from app.retriever import Retriever
from app.schemas import (
    BatchUploadItem,
    BatchUploadResponse,
    FeedbackRequest,
    FeedbackResponse,
    FailureCasesResponse,
    HistoryResponse,
    QueryRequest,
    QueryResponse,
    RLMStatsResponse,
    UploadResponse,
)

router = APIRouter(prefix="/api", tags=["recursive-knowledge-engine"])

retriever = Retriever(
    embedding_model=settings.embedding_model,
    index_path=settings.faiss_index_path,
    metadata_path=settings.metadata_path,
    chunk_size_words=settings.chunk_size_words,
    chunk_overlap_words=settings.chunk_overlap_words,
    query_cache_size=settings.query_cache_size,
)
answerer = Answerer(
    api_key=settings.llm_api_key,
    model=settings.llm_model,
    base_url=settings.llm_base_url,
    provider=settings.llm_provider,
    timeout_seconds=settings.llm_timeout_seconds,
    max_retries=settings.llm_max_retries,
    fallback_models=list(settings.llm_fallback_models),
)
critic = Critic(
    api_key=settings.llm_api_key,
    model=settings.llm_model,
    base_url=settings.llm_base_url,
    provider=settings.llm_provider,
    timeout_seconds=settings.llm_timeout_seconds,
    max_retries=settings.llm_max_retries,
    fallback_models=list(settings.llm_fallback_models),
)
history_store = SessionHistoryStore()
failure_store = FailureCaseStore()
rlm_store = RewardLearningMemory()
controller = RecursiveController(
    retriever,
    answerer,
    critic,
    history_store,
    failure_store,
    rlm_store,
    early_stop_confidence=settings.early_stop_confidence,
    coverage_warning_threshold=settings.coverage_warning_threshold,
    deterministic_seed=settings.deterministic_seed,
)


@router.get("/health")
def health() -> dict:
    retriever_stats = retriever.stats()
    runtime_stats = controller.runtime_stats()
    return {
        "status": "ok",
        "app": settings.app_name,
        "chunks_indexed": retriever.total_chunks(),
        "documents_indexed": retriever_stats["documents_indexed"],
        "embedding_model_loaded": retriever_stats["embedding_model_loaded"],
        "query_cache_size": retriever_stats["query_cache_size"],
        "faiss_index_size_bytes": retriever_stats["faiss_index_size_bytes"],
        "last_embedding_at": retriever_stats["last_embedding_at"],
        "last_model_fallback_used": runtime_stats["last_model_fallback_used"],
        "total_queries": runtime_stats["total_queries"],
        "total_estimated_tokens": runtime_stats["total_estimated_tokens"],
        "rlm_feedback_count": runtime_stats["rlm_feedback_count"],
        "rlm_tracked_sources": runtime_stats["rlm_tracked_sources"],
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@router.get("/stats")
def stats() -> dict:
    return {
        **retriever.stats(),
        **controller.runtime_stats(),
    }


async def _save_and_index(file: UploadFile) -> tuple[str, str, int]:
    suffix = Path(file.filename).suffix.lower()
    if not retriever.is_supported_file(file.filename):
        supported = ", ".join(retriever.SUPPORTED_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type ({suffix or 'none'}). Supported formats: {supported}.",
        )

    safe_name = f"{uuid4()}_{Path(file.filename).name}"
    save_path = settings.uploads_dir / safe_name

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_path.write_bytes(data)

    try:
        chunks_added = retriever.add_document(save_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {exc}") from exc

    return file.filename, safe_name, chunks_added


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename, safe_name, chunks_added = await _save_and_index(file)

    return UploadResponse(
        filename=filename,
        stored_as=safe_name,
        chunks_added=chunks_added,
        total_chunks=retriever.total_chunks(),
    )


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_documents_batch(files: List[UploadFile] = File(...)) -> BatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    items: List[BatchUploadItem] = []
    for file in files:
        try:
            filename, stored_as, chunks_added = await _save_and_index(file)
            items.append(
                BatchUploadItem(
                    filename=filename,
                    stored_as=stored_as,
                    chunks_added=chunks_added,
                )
            )
        except HTTPException as exc:
            items.append(
                BatchUploadItem(
                    filename=file.filename,
                    stored_as="",
                    chunks_added=0,
                    error=str(exc.detail),
                )
            )

    return BatchUploadResponse(items=items, total_chunks=retriever.total_chunks())


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    if not retriever.has_data():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload files first.")

    try:
        return controller.run(
            question=request.question,
            top_k=request.top_k,
            max_iterations=request.max_iterations,
            session_id=request.session_id,
            fast_mode=request.fast_mode,
            deterministic_mode=request.deterministic_mode,
            ablation_mode=request.ablation_mode,
            ablation_depths=request.ablation_depths,
            source_filters=request.source_filters,
            answer_verbosity=request.answer_verbosity,
            challenge_mode=request.challenge_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.get("/history/{session_id}", response_model=HistoryResponse)
def get_session_history(session_id: str) -> HistoryResponse:
    items = history_store.get(session_id)
    summary = history_store.get_summary(session_id)
    return HistoryResponse(
        session_id=session_id,
        items=items,
        total_queries=summary["total_queries"],
        estimated_tokens_used=summary["estimated_tokens_used"],
    )


@router.get("/failures", response_model=FailureCasesResponse)
def get_failure_cases(session_id: Optional[str] = None) -> FailureCasesResponse:
    items = controller.get_failures(session_id=session_id)
    return FailureCasesResponse(count=len(items), items=items)


@router.get("/failures/export", response_model=FailureCasesResponse)
def export_failure_cases(session_id: Optional[str] = None) -> FailureCasesResponse:
    items = controller.get_failures(session_id=session_id)
    return FailureCasesResponse(count=len(items), items=items)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        result = controller.record_feedback(
            response_id=request.response_id,
            rating=request.rating,
            notes=request.notes,
        )
        return FeedbackResponse(status="ok", **result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown response_id: {request.response_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/rlm/stats", response_model=RLMStatsResponse)
def get_rlm_stats() -> RLMStatsResponse:
    return RLMStatsResponse(**controller.rlm_stats())


@router.post("/index/clear")
def clear_index() -> dict:
    stats_after_clear = retriever.clear_index()
    return {
        "message": "Index cleared.",
        "stats": stats_after_clear,
    }
