from reqwise.backend.schemas import AnalyzeResult
from reqwise.backend.services.model_service import ModelService
from reqwise.backend.services.rag_service import RagService
from reqwise.backend.services.rewrite_service import RewriteService
from reqwise.backend.services.transformer_service import TransformerService


class AnalysisService:
    def __init__(
        self,
        model_service: ModelService,
        rag_service: RagService,
        rewrite_service: RewriteService,
        transformer_service: TransformerService,
    ) -> None:
        self.model_service = model_service
        self.rag_service = rag_service
        self.rewrite_service = rewrite_service
        self.transformer_service = transformer_service

    def analyze(self, requirement: str, top_k_context: int = 3, include_rewrite: bool = False) -> AnalyzeResult:
        prediction = self.model_service.predict(requirement)

        if prediction.predicted_class == "NFR" and self.transformer_service.subtype_loaded:
            subtype_pred = self.transformer_service.predict_subtype(requirement)
            if subtype_pred is not None:
                prediction.predicted_subtype = subtype_pred.label
                prediction.confidence_subtype = subtype_pred.confidence

        if self.transformer_service.ambiguity_loaded:
            amb_pred = self.transformer_service.predict_ambiguity(requirement)
            if amb_pred is not None:
                prediction.predicted_ambiguity = amb_pred.label
                prediction.confidence_ambiguity = amb_pred.confidence

        context = self.rag_service.retrieve(requirement, top_k=top_k_context) if top_k_context > 0 else []

        rewrite = None
        if include_rewrite:
            rewrite = self.rewrite_service.rewrite(
                requirement_text=requirement,
                predicted_class=prediction.predicted_class,
                predicted_subtype=prediction.predicted_subtype,
                predicted_ambiguity=prediction.predicted_ambiguity,
                context_snippets=context,
            )

        return AnalyzeResult(
            requirement=requirement,
            predicted_class=prediction.predicted_class,
            predicted_subtype=prediction.predicted_subtype,
            predicted_ambiguity=prediction.predicted_ambiguity,
            confidence_class=prediction.confidence_class,
            confidence_subtype=prediction.confidence_subtype,
            confidence_ambiguity=prediction.confidence_ambiguity,
            rewrite=rewrite,
            explain_terms=prediction.explain_terms,
            retrieved_context=context,
        )
