from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeTextRequest(BaseModel):
    requirement: str = Field(..., min_length=3)
    top_k_context: int = Field(default=3, ge=0, le=8)
    include_rewrite: bool = Field(default=False)


class AnalyzeResult(BaseModel):
    requirement: str
    predicted_class: str
    predicted_subtype: str
    predicted_ambiguity: str
    confidence_class: float
    confidence_subtype: Optional[float] = None
    confidence_ambiguity: float
    rewrite: Optional[str] = None
    explain_terms: list[str]
    retrieved_context: list[str]


class RagStatus(BaseModel):
    documents: int
    chunks: int


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    llama_loaded: bool
    roberta_subtype_loaded: bool = False
    roberta_ambiguity_loaded: bool = False
