# PURR

Modern local web app for automated requirement analysis using your dissertation pipeline:
- FR/NFR classification
- NFR subtype prediction (RoBERTa checkpoint if available, else SVM fallback)
- ambiguity detection (RoBERTa checkpoint if available, else SVM fallback)
- LLM rewriting (only when ambiguity is detected)
- PDF-based RAG context for rewrite guidance

## Tech Stack
- Backend: FastAPI
- Frontend: Streamlit
- Models: Linear SVM pipelines + local GGUF Llama model

## Project Layout
- `reqwise/backend/main.py`: FastAPI app and endpoints
- `reqwise/backend/services/model_service.py`: Stage 1/2/3 model training and inference
- `reqwise/backend/services/rag_service.py`: PDF ingestion and retrieval
- `reqwise/backend/services/rewrite_service.py`: local Llama rewrite generation
- `reqwise/frontend/streamlit_app.py`: table-first web UI

## Setup
1. Activate your virtual environment.
2. Install dependencies:
   `pip install -r reqwise/requirements.txt`

## Run
Open two terminals from repository root.

Terminal A:
`uvicorn reqwise.backend.main:app --reload --host 127.0.0.1 --port 8000`

Terminal B:
`streamlit run reqwise/frontend/streamlit_app.py`

## Checkpoint Loading
- Subtype model checkpoints are read from `results_roberta/checkpoint-*`.
- Ambiguity model checkpoints are read from `results_roberta_ambiguity/checkpoint-*`.
- The app automatically loads the highest checkpoint number in each directory.
- Verify status at `GET /health` using `roberta_subtype_loaded` and `roberta_ambiguity_loaded`.

## Python Version Note
- If using Python 3.14, PyTorch wheels may be unavailable and RoBERTa checkpoint loading may fall back to SVM.
- For full RoBERTa runtime inference, use Python 3.11 or 3.12 in the virtual environment.

## Usage
1. Upload a PDF in the sidebar to build RAG context.
2. Enter one requirement and click **Analyze and Add to Table**.
3. Each analysis appends a row with class, subtype, ambiguity, confidence, explain terms, and optional rewrite.
4. For batch mode, upload CSV and run analysis (max 50 rows per run).

## CSV Notes
Batch endpoint expects one text column named one of:
- `Requirement Text`
- `requirement`
- `text`
- `requirement_text`

## Behavior Choices Implemented
- PDF only for document ingestion.
- Same local GGUF model from `models/Meta-Llama-3-8B-Instruct.Q8_0.gguf`.
- Rewrite generated only when ambiguity is detected.
- Batch processing capped at 50 requirements per run.
- Transparent UI table with explainable terms.
