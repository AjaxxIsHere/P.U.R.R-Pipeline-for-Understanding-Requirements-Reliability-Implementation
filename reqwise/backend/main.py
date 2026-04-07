from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from reqwise.backend.schemas import AnalyzeResult, AnalyzeTextRequest, HealthResponse, RagStatus
from reqwise.backend.services.analysis_service import AnalysisService
from reqwise.backend.services.model_service import ModelService
from reqwise.backend.services.rag_service import RagService
from reqwise.backend.services.rewrite_service import RewriteService
from reqwise.backend.services.transformer_service import TransformerService
from reqwise.backend.settings import MAX_BATCH_SIZE, RUNTIME_DIR


app = FastAPI(title="ReqWise API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
rag_service = RagService()
rewrite_service = RewriteService()
transformer_service = TransformerService()
analysis_service = AnalysisService(
    model_service=model_service,
    rag_service=rag_service,
    rewrite_service=rewrite_service,
    transformer_service=transformer_service,
)


@app.on_event("startup")
def startup_event() -> None:
    model_service.load_and_train()
    transformer_service.load_models()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        dataset_loaded=model_service.dataset_loaded,
        llama_loaded=rewrite_service.llama_loaded,
        roberta_subtype_loaded=transformer_service.subtype_loaded,
        roberta_ambiguity_loaded=transformer_service.ambiguity_loaded,
    )


@app.get("/rag/status", response_model=RagStatus)
def rag_status() -> RagStatus:
    status = rag_service.status()
    return RagStatus(documents=status["documents"], chunks=status["chunks"])


@app.post("/rag/upload-pdf")
async def rag_upload_pdf(file: UploadFile = File(...)) -> dict:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = RUNTIME_DIR / file.filename
    content = await file.read()
    tmp_path.write_bytes(content)
    result = rag_service.ingest_pdf(tmp_path, source_name=file.filename)
    return {"status": "ok", "ingested": result}


@app.post("/analyze/text", response_model=AnalyzeResult)
def analyze_text(payload: AnalyzeTextRequest) -> AnalyzeResult:
    return analysis_service.analyze(
        payload.requirement,
        top_k_context=payload.top_k_context,
        include_rewrite=payload.include_rewrite,
    )


@app.post("/analyze/batch")
async def analyze_batch(file: UploadFile = File(...), top_k_context: int = 3, include_rewrite: bool = False) -> dict:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for batch analysis.")

    content = (await file.read()).decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(content))

    candidate_cols = ["Requirement Text", "requirement", "text", "requirement_text"]
    input_col = next((c for c in candidate_cols if c in df.columns), None)
    if input_col is None:
        raise HTTPException(status_code=400, detail="CSV must include a requirement text column.")

    rows = df[input_col].dropna().astype(str).tolist()[:MAX_BATCH_SIZE]
    results = [
        analysis_service.analyze(req, top_k_context=top_k_context, include_rewrite=include_rewrite).model_dump()
        for req in rows
    ]
    return {"count": len(results), "limit": MAX_BATCH_SIZE, "results": results}
