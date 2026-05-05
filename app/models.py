import uuid
from datetime import datetime, timezone

from pydantic import EmailStr
from sqlalchemy import DateTime
from sqlmodel import Field, Relationship, SQLModel
from typing import Optional
import json


def get_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)

class RAGEvaluationLog(SQLModel, table=True):
    __tablename__ = "rag_evaluation_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
    )
    question:     str = Field(max_length=500)
    answer:       str
    answered_by:  str = Field(max_length=50)
    model:        str = Field(max_length=50)
    sources_used: str = Field(default="[]")

    faithfulness:          float = Field(default=0.0)
    answer_relevance:      float = Field(default=0.0)
    context_precision:     float = Field(default=0.0)
    overall_score:         float = Field(default=0.0)
    faithfulness_raw:      int   = Field(default=3)
    answer_relevance_raw:  int   = Field(default=3)
    context_precision_raw: int   = Field(default=3)
    faithfulness_reason:       str = Field(default="")
    answer_relevance_reason:   str = Field(default="")
    context_precision_reason:  str = Field(default="")

    validation_verdict:    str  = Field(default="", max_length=20)
    validation_confidence: int  = Field(default=0)
    question_category:     str  = Field(default="", max_length=20)
    critical_fail:         str  = Field(default="")
    critical_fail_reason:  str  = Field(default="")
    validation_reason:     str  = Field(default="")

    def set_sources(self, sources: list[dict]) -> None:
        self.sources_used = json.dumps(sources)

    def get_sources(self) -> list[dict]:
        try:
            return json.loads(self.sources_used)
        except Exception:
            return []