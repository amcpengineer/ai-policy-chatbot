"""
intelligence.py — FastAPI routes for the AI Policy RAG API.

Endpoints:
    POST /api/intelligence/ask      — Ask a question, returns answer + validation + triad
    GET  /api/intelligence/health   — Check Ollama + vector store status
    GET  /api/intelligence/metrics  — Aggregate RAG Triad scores + validation stats
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])

_rag_service = RAGService()


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="The question to ask about AI policies.",
        examples=["What does the EU AI Act say about high-risk systems?"],
    )


class SourceItem(BaseModel):
    source:          str
    page:            int
    relevance_score: float
    excerpt:         str


class ValidationResult(BaseModel):
    verdict:                   str
    confidence_score:          int
    grounded:                  bool
    relevant:                  bool
    harmful_handled_correctly: bool
    critical_fail:             bool | str | None
    critical_fail_reason:      str | None
    question_category:         str
    reason:                    str


class TriadResult(BaseModel):
    faithfulness:              float   # 0.0–1.0 normalized
    answer_relevance:          float
    context_precision:         float
    overall_score:             float
    faithfulness_raw:          int     # 0–5 Likert
    answer_relevance_raw:      int
    context_precision_raw:     int
    faithfulness_reason:       str
    answer_relevance_reason:   str
    context_precision_reason:  str
    scorer_available:          bool


class AskResponse(BaseModel):
    answer:      str
    answered_by: str
    sources:     list[SourceItem]
    model:       str
    validation:  ValidationResult
    triad:       TriadResult


class HealthResponse(BaseModel):
    ollama:               str
    dbu_policy_chunks:    int
    global_policy_chunks: int
    model:                str
    ready:                bool


class TARResult(BaseModel):
    tar_r: float   # raw score agreement rate
    tar_a: float   # answer bucket agreement rate
    n:     int


class MetricsResponse(BaseModel):
    total_questions:           int
    avg_faithfulness:          float
    avg_answer_relevance:      float
    avg_context_precision:     float
    avg_overall_score:         float
    avg_validation_confidence: float
    verdict_distribution:      dict
    category_distribution:     dict
    critical_fail_count:       int
    tar:                       TARResult


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question about AI policies.

    Every answer goes through:
    - Two-tier RAG retrieval (DBU policy first, global policies fallback)
    - ValidationAgent (DBU Q&A Test Bank criteria)
    - RAG Triad scoring (Faithfulness, Answer Relevance, Context Precision)
    - Full interaction logged to the database
    """
    try:
        result = _rag_service.ask(request.question)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if Ollama is reachable and both collections have data."""
    try:
        result = _rag_service.health_check()
        return HealthResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """
    Aggregate RAG quality metrics across all logged interactions.

    Returns average RAG Triad scores, validation verdict distribution,
    question category breakdown, critical fail count, and TAR scores.
    Use this endpoint for your live demo dashboard.
    """
    try:
        result = _rag_service.get_metrics()
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return MetricsResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent")
def get_recent(limit: int = 10):
    from sqlmodel import Session, select
    from app.core.db import engine
    from app.models import RAGEvaluationLog
    with Session(engine) as session:
        logs = session.exec(
            select(RAGEvaluationLog)
            .order_by(RAGEvaluationLog.created_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "created_at":         l.created_at,
                "question":           l.question[:80],
                "question_category":  l.question_category,
                "validation_verdict": l.validation_verdict,
                "overall_score":      l.overall_score,
                "answered_by":        l.answered_by,
            }
            for l in logs
        ]