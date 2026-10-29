
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Citation(BaseModel):
    document_id: str
    chunk_id: str


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool


class TriageRequest(BaseModel):
    ticket: str = Field(min_length=5, max_length=5000)


class TriageResponse(BaseModel):
    category: str
    priority: str
    product_area: str
    summary: str
