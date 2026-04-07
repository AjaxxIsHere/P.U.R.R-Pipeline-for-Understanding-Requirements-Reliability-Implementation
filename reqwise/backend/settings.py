from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "final_corpus_verified - Sheet1.csv"
LLAMA_MODEL_PATH = ROOT_DIR / "models" / "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
RUNTIME_DIR = ROOT_DIR / "reqwise" / "runtime"
RAG_STATE_PATH = RUNTIME_DIR / "rag_state.json"
ROBERTA_SUBTYPE_DIR = ROOT_DIR / "results_roberta"
ROBERTA_AMBIGUITY_DIR = ROOT_DIR / "results_roberta_ambiguity"
MAX_BATCH_SIZE = 50
TARGET_SUBTYPES = [
    "Usability",
    "Security",
    "Performance",
    "Maintainability",
    "Availability",
]
