from __future__ import annotations

from typing import Optional

from llama_cpp import Llama

from reqwise.backend.settings import LLAMA_MODEL_PATH


SUBTYPE_GUIDANCE = {
    "Security": "Replace vague security wording with concrete controls, constraints, or testable requirements.",
    "Performance": "Use measurable response time, throughput, latency, or capacity language.",
    "Availability": "Clarify uptime, failover, recovery, and continuity requirements in measurable terms.",
    "Usability": "Make interaction outcomes observable and testable with clear acceptance language.",
    "Maintainability": "Specify modularity, diagnosability, and changeability constraints without adding new features.",
}


class RewriteService:
    def __init__(self) -> None:
        self.llm: Optional[Llama] = None
        self.llama_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        if not LLAMA_MODEL_PATH.exists():
            self.llama_loaded = False
            return
        self.llm = Llama(model_path=str(LLAMA_MODEL_PATH), n_ctx=2048, verbose=False)
        self.llama_loaded = True

    @staticmethod
    def should_rewrite(predicted_ambiguity: str) -> bool:
        label = (predicted_ambiguity or "").strip().lower()
        return "ambig" in label and "non" not in label and "unamb" not in label

    def _build_prompt(
        self,
        requirement_text: str,
        predicted_class: str,
        predicted_subtype: str,
        predicted_ambiguity: str,
        context_snippets: list[str],
    ) -> str:
        context_block = "\n".join([f"- {c}" for c in context_snippets]) if context_snippets else "- No external document context provided."
        subtype_rule = SUBTYPE_GUIDANCE.get(
            predicted_subtype,
            "Make the requirement specific, testable, and implementation-ready while preserving original intent.",
        )
        return (
            "You are an expert software requirements engineer.\n"
            "Rewrite the requirement so it is clear, unambiguous, concise, and testable.\n"
            "Preserve original meaning. Do not invent new features.\n\n"
            f"Predicted metadata: class={predicted_class}, subtype={predicted_subtype}, ambiguity={predicted_ambiguity}.\n"
            f"Subtype guidance: {subtype_rule}\n\n"
            "Retrieved context from uploaded requirements document:\n"
            f"{context_block}\n\n"
            f"Original requirement:\n\"\"\"{requirement_text}\"\"\"\n\n"
            "Return one rewritten requirement only."
        )

    def rewrite(
        self,
        requirement_text: str,
        predicted_class: str,
        predicted_subtype: str,
        predicted_ambiguity: str,
        context_snippets: list[str],
    ) -> Optional[str]:
        if not self.llama_loaded or self.llm is None:
            return None
        if not self.should_rewrite(predicted_ambiguity):
            return None

        prompt = self._build_prompt(
            requirement_text=requirement_text,
            predicted_class=predicted_class,
            predicted_subtype=predicted_subtype,
            predicted_ambiguity=predicted_ambiguity,
            context_snippets=context_snippets,
        )
        response = self.llm(prompt, max_tokens=220, temperature=0.0)
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("text", "").strip()
        return response.get("text", "").strip()
