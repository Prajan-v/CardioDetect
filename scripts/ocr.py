import argparse
from pathlib import Path
import json
from src.ocr.pipeline import extract_structured


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="Path to medical report (PDF or image)")
    p.add_argument("--out", type=str, default="output/ocr_result.json")
    args = p.parse_args()
    res = extract_structured(Path(args.input))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved OCR result to {args.out}")


if __name__ == "__main__":
    main()
