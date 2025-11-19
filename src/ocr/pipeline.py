from pathlib import Path
import pytesseract
from .ingest import pdf_to_images, load_image, preprocess_image
from .parse import parse_text
import os
import numpy as np
from pytesseract import Output


def extract_structured(path: Path):
    p = Path(path)
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    texts = []
    confs = []
    if p.suffix.lower() == ".pdf":
        images = pdf_to_images(p)
        for img in images:
            proc = preprocess_image(img)
            try:
                data = pytesseract.image_to_data(proc, output_type=Output.DICT)
                page_confs = [int(c) for c in data.get("conf", []) if c != "-1"]
                confs.extend(page_confs)
                words = data.get("text", [])
                txt = " ".join([w for w, c in zip(words, data.get("conf", [])) if c != "-1"]) 
                texts.append(txt)
            except Exception:
                try:
                    texts.append(pytesseract.image_to_string(proc))
                except Exception:
                    continue
    else:
        img = load_image(p)
        proc = preprocess_image(img)
        try:
            data = pytesseract.image_to_data(proc, output_type=Output.DICT)
            page_confs = [int(c) for c in data.get("conf", []) if c != "-1"]
            confs.extend(page_confs)
            words = data.get("text", [])
            txt = " ".join([w for w, c in zip(words, data.get("conf", [])) if c != "-1"]) 
            texts.append(txt)
        except Exception:
            try:
                texts.append(pytesseract.image_to_string(proc))
            except Exception:
                texts.append("")
    merged = "\n".join(texts)
    out = parse_text(merged)
    if confs:
        out["ocr_mean_conf"] = float(np.mean(confs))
    return out
