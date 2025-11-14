import argparse
import json
from pathlib import Path
from src.config import ensure_dirs, INTERIM_DIR
from src.ocr.pipeline import extract_structured


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str)
    p.add_argument("--out", type=str, default=str(INTERIM_DIR / "ocr_output.json"))
    args = p.parse_args()
    ensure_dirs()
    d = extract_structured(Path(args.input))
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
