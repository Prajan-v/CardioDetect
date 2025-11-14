from pathlib import Path
import pytesseract
from .ingest import pdf_to_images, load_image, preprocess_image
from .parse import parse_text
import os


def extract_structured(path: Path):
    p = Path(path)
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    texts = []
    if p.suffix.lower() == ".pdf":
        images = pdf_to_images(p)
        for img in images:
            proc = preprocess_image(img)
            txt = pytesseract.image_to_string(proc)
            texts.append(txt)
    else:
        img = load_image(p)
        proc = preprocess_image(img)
        texts.append(pytesseract.image_to_string(proc))
    merged = "\n".join(texts)
    return parse_text(merged)
