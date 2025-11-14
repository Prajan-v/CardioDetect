from pathlib import Path
from src.config import RAW_DIR, ensure_dirs
import kagglehub
import shutil

def main():
    ensure_dirs()
    path = kagglehub.dataset_download("aasheesh200/framingham-heart-study-dataset")
    p = Path(path)
    cands = list(p.glob("*.csv")) + list(p.rglob("*.csv"))
    if not cands:
        return
    cand = None
    for c in cands:
        if c.name.lower().startswith("framingham"):
            cand = c
            break
    if cand is None:
        cand = cands[0]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cand, RAW_DIR / "framingham.csv")

if __name__ == "__main__":
    main()
