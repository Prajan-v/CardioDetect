import requests
from pathlib import Path
from src.config import RAW_DIR, CLEVELAND_URL, ensure_dirs

def download_cleveland(overwrite=False, timeout=30):
    ensure_dirs()
    path = RAW_DIR / "processed.cleveland.data"
    if path.exists() and not overwrite:
        return path
    try:
        r = requests.get(CLEVELAND_URL, timeout=timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
        return path
    except Exception:
        alt = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        r = requests.get(alt, timeout=timeout)
        r.raise_for_status()
        path.write_bytes(r.content)
        return path
