from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from reqwise.backend.services.preprocessing import clean_text_ambiguity, clean_text_base
from reqwise.backend.settings import DATASET_PATH, TARGET_SUBTYPES


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_subtype: str
    predicted_ambiguity: str
    confidence_class: float
    confidence_subtype: Optional[float]
    confidence_ambiguity: float
    explain_terms: list[str]


class ModelService:
    def __init__(self) -> None:
        self.dataset_loaded = False
        self.df: Optional[pd.DataFrame] = None

        self.class_vectorizer: Optional[TfidfVectorizer] = None
        self.class_model: Optional[LinearSVC] = None

        self.subtype_vectorizer: Optional[TfidfVectorizer] = None
        self.subtype_model: Optional[LinearSVC] = None
        self.subtype_id2label: dict[int, str] = {}

        self.amb_vectorizer: Optional[TfidfVectorizer] = None
        self.amb_model: Optional[LinearSVC] = None
        self.amb_id2label: dict[int, str] = {}

    def load_and_train(self) -> None:
        self.df = pd.read_csv(DATASET_PATH, nrows=4000)
        self.dataset_loaded = True
        self._train_stage1_class_model()
        self._train_stage2_subtype_model()
        self._train_stage3_ambiguity_model()

    @staticmethod
    def _normalize_binary_margin(margin: float) -> float:
        return float(1.0 / (1.0 + np.exp(-margin)))

    @staticmethod
    def _binary_predicted_confidence(margin: float, predicted_id: int) -> float:
        p_class_1 = float(1.0 / (1.0 + np.exp(-margin)))
        return p_class_1 if predicted_id == 1 else float(1.0 - p_class_1)

    @staticmethod
    def _normalize_multiclass_scores(scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        exp = np.exp(shifted)
        return exp / np.sum(exp)

    def _train_stage1_class_model(self) -> None:
        assert self.df is not None
        data = self.df.dropna(subset=["Requirement Text", "Class"]).copy()
        data["cleaned_text"] = data["Requirement Text"].apply(clean_text_base)
        data = data[data["cleaned_text"].str.split().str.len() >= 3]
        data = data.drop_duplicates(subset=["cleaned_text"])
        data["label"] = data["Class"].astype(str).str.strip().str.upper().map({"FR": 0, "NFR": 1})
        data = data.dropna(subset=["label"])

        class_counts = data["label"].value_counts()
        if len(class_counts) >= 2:
            minority_class = class_counts.idxmin()
            majority_class = class_counts.idxmax()
            minority_count = int(class_counts.min())
            df_majority = data[data["label"] == majority_class]
            df_minority = data[data["label"] == minority_class]
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=minority_count,
                random_state=42,
            )
            data = pd.concat([df_minority, df_majority_downsampled]).sample(frac=1.0, random_state=42)

        self.class_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2)
        X = self.class_vectorizer.fit_transform(data["cleaned_text"])
        y = data["label"].astype(int)
        self.class_model = LinearSVC(random_state=42, dual=False)
        self.class_model.fit(X, y)

    def _train_stage2_subtype_model(self) -> None:
        assert self.df is not None
        data = self.df.dropna(subset=["Requirement Text", "Sub-Type"]).copy()
        data = data[data["Sub-Type"].isin(TARGET_SUBTYPES)]
        data["cleaned_text"] = data["Requirement Text"].apply(clean_text_base)
        data = data[data["cleaned_text"].str.split().str.len() >= 3]
        data = data.drop_duplicates(subset=["cleaned_text"])

        label_mapping = {label: idx for idx, label in enumerate(TARGET_SUBTYPES)}
        data["label"] = data["Sub-Type"].map(label_mapping)
        data = data.dropna(subset=["label"])
        self.subtype_id2label = {idx: label for label, idx in label_mapping.items()}

        self.subtype_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.85, min_df=2)
        X = self.subtype_vectorizer.fit_transform(data["cleaned_text"])
        y = data["label"].astype(int)

        # Keep class balancing behavior from notebook pipeline.
        min_count = y.value_counts().min()
        if min_count >= 2:
            k_neighbors = max(1, min(5, min_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X, y = smote.fit_resample(X, y)

        self.subtype_model = LinearSVC(random_state=42, dual=False)
        self.subtype_model.fit(X, y)

    def _train_stage3_ambiguity_model(self) -> None:
        assert self.df is not None
        data = self.df.dropna(subset=["Requirement Text", "Ambiguity"]).copy()
        data["cleaned_text"] = data["Requirement Text"].apply(clean_text_ambiguity)
        data = data[data["cleaned_text"].str.split().str.len() >= 3]
        data = data.drop_duplicates(subset=["cleaned_text"])

        labels = data["Ambiguity"].astype(str).str.strip()
        label_names = sorted([name for name in labels.unique().tolist() if name and name.lower() != "nan"])
        label_mapping = {label: idx for idx, label in enumerate(label_names)}
        data["label"] = labels.map(label_mapping)
        data = data.dropna(subset=["label"])

        counts = data["label"].value_counts()
        if len(counts) >= 2:
            minority_class = counts.idxmin()
            majority_class = counts.idxmax()
            minority_count = int(counts.min())
            df_majority = data[data["label"] == majority_class]
            df_minority = data[data["label"] == minority_class]
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=minority_count,
                random_state=42,
            )
            data = pd.concat([df_minority, df_majority_downsampled]).sample(frac=1.0, random_state=42)

        self.amb_id2label = {idx: label for label, idx in label_mapping.items()}
        self.amb_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.9, min_df=2)
        X = self.amb_vectorizer.fit_transform(data["cleaned_text"])
        y = data["label"].astype(int)
        self.amb_model = LinearSVC(random_state=42, dual=False)
        self.amb_model.fit(X, y)

    def _get_top_terms_for_class(self, text: str, top_n: int = 8) -> list[str]:
        assert self.class_vectorizer is not None
        assert self.class_model is not None

        cleaned = clean_text_base(text)
        if not cleaned:
            return []

        x = self.class_vectorizer.transform([cleaned])
        feature_names = np.array(self.class_vectorizer.get_feature_names_out())
        coef = self.class_model.coef_[0]
        contrib = x.toarray()[0] * coef
        idx = np.argsort(np.abs(contrib))[::-1]
        terms = [feature_names[i] for i in idx if contrib[i] != 0.0]
        return terms[:top_n]

    def predict(self, text: str) -> PredictionResult:
        if not self.dataset_loaded:
            self.load_and_train()

        assert self.class_vectorizer is not None and self.class_model is not None
        assert self.subtype_vectorizer is not None and self.subtype_model is not None
        assert self.amb_vectorizer is not None and self.amb_model is not None

        cleaned_base = clean_text_base(text)
        cleaned_amb = clean_text_ambiguity(text)

        x_class = self.class_vectorizer.transform([cleaned_base])
        class_id = int(self.class_model.predict(x_class)[0])
        class_margin = float(self.class_model.decision_function(x_class)[0])
        class_conf = self._binary_predicted_confidence(class_margin, class_id)
        predicted_class = "NFR" if class_id == 1 else "FR"

        predicted_subtype = "N/A"
        subtype_conf = None
        if predicted_class == "NFR":
            x_sub = self.subtype_vectorizer.transform([cleaned_base])
            subtype_id = int(self.subtype_model.predict(x_sub)[0])
            subtype_scores = np.ravel(self.subtype_model.decision_function(x_sub))
            subtype_probs = self._normalize_multiclass_scores(subtype_scores)
            subtype_conf = float(subtype_probs[subtype_id])
            predicted_subtype = self.subtype_id2label.get(subtype_id, "N/A")

        x_amb = self.amb_vectorizer.transform([cleaned_amb])
        amb_id = int(self.amb_model.predict(x_amb)[0])
        amb_scores = np.ravel(self.amb_model.decision_function(x_amb))
        if amb_scores.size == 1:
            amb_conf = self._binary_predicted_confidence(float(amb_scores[0]), amb_id)
        else:
            amb_probs = self._normalize_multiclass_scores(amb_scores)
            amb_conf = float(amb_probs[amb_id])
        predicted_ambiguity = self.amb_id2label.get(amb_id, str(amb_id))

        explain_terms = self._get_top_terms_for_class(text)

        return PredictionResult(
            predicted_class=predicted_class,
            predicted_subtype=predicted_subtype,
            predicted_ambiguity=predicted_ambiguity,
            confidence_class=class_conf,
            confidence_subtype=subtype_conf,
            confidence_ambiguity=amb_conf,
            explain_terms=explain_terms,
        )
