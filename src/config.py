from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "reports"
CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

def ensure_dirs():
    for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, DOCS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
