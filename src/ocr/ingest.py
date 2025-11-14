from pathlib import Path
import fitz
from PIL import Image
import numpy as np
import cv2


def pdf_to_images(path: Path, dpi=300):
    doc = fitz.open(str(path))
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        images.append(np.array(img)[:, :, ::-1])
    return images


def load_image(path: Path):
    img = cv2.imread(str(path))
    return img


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return thr
