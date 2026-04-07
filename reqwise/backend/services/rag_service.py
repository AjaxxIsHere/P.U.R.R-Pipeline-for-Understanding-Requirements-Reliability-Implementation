from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from reqwise.backend.settings import RAG_STATE_PATH


class RagService:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.chunks: list[str] = []
        self.chunk_meta: list[dict] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=1)
        self.chunk_matrix = None
        self._load_state()

    def _load_state(self) -> None:
        if not RAG_STATE_PATH.exists():
            return
        try:
            payload = json.loads(RAG_STATE_PATH.read_text(encoding="utf-8"))
            self.documents = payload.get("documents", [])
            self.chunks = payload.get("chunks", [])
            self.chunk_meta = payload.get("chunk_meta", [])
            if self.chunks:
                self.chunk_matrix = self.vectorizer.fit_transform(self.chunks)
        except Exception:
            self.documents = []
            self.chunks = []
            self.chunk_meta = []
            self.chunk_matrix = None

    def _save_state(self) -> None:
        RAG_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self.documents,
            "chunks": self.chunks,
            "chunk_meta": self.chunk_meta,
        }
        RAG_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
        text = " ".join(text.split())
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def ingest_pdf(self, pdf_path: Path, source_name: str) -> dict:
        text = self._extract_pdf_text(pdf_path)
        chunks = self._chunk_text(text)
        doc_id = len(self.documents) + 1
        self.documents.append({"doc_id": doc_id, "source_name": source_name, "chunks": len(chunks)})

        for idx, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_meta.append({"doc_id": doc_id, "chunk_idx": idx, "source_name": source_name})

        if self.chunks:
            self.chunk_matrix = self.vectorizer.fit_transform(self.chunks)

        self._save_state()
        return {"doc_id": doc_id, "source_name": source_name, "chunks": len(chunks)}

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if not self.chunks or self.chunk_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = (self.chunk_matrix @ query_vec.T).toarray().ravel()
        if scores.size == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            meta = self.chunk_meta[idx]
            results.append(f"[{meta['source_name']} | chunk {meta['chunk_idx']}] {self.chunks[idx]}")
        return results

    def status(self) -> dict:
        return {"documents": len(self.documents), "chunks": len(self.chunks)}
