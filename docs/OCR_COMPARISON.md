# OCR Engine Comparison Report

Ground truth is the manually transcribed CBC report used in the
`complete_accuracy_test.py` script.

## Tesseract

| Metric | Value |
|--------|-------|
| Processing Time | 3.01 s |
| Character Accuracy | 1.48% |
| Word Accuracy | 52.56% |
| Field Extraction | 3/6 (50.00%) |

## DeepSeek-OCR

Status: **FAILED** — Failed to invoke DeepSeek-OCR binary: [Errno 8] Exec format error: '/Users/prajanv/CardioDetect/ocr_models/deepseek-ocr'

---

**Recommendation:** Use **Tesseract** as the primary OCR engine.
Tesseract remains a fallback when the DeepSeek binary is missing or fails.