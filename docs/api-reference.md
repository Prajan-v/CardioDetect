# CardioDetect — API Reference

> Django REST backend running at `http://localhost:8000`. See the [main README](../README.md) for setup instructions.

---

## Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/predict/` | ✅ Required | Upload a document and receive risk prediction |
| `GET` | `/api/predictions/` | ✅ Required | List all predictions for the authenticated user |
| `GET` | `/api/predictions/<id>/` | ✅ Required | Retrieve a specific prediction result |
| `POST` | `/api/accounts/register/` | ❌ None | User registration |
| `POST` | `/api/accounts/login/` | ❌ None | Login — returns auth token |
| `GET` | `/api/accounts/profile/` | ✅ Required | Get authenticated user profile |

---

## Request — POST `/api/predict/`

```
Content-Type: multipart/form-data
Authorization: Token <your_token>

Body:
  document: <file>   (PDF, PNG, JPG, or TXT)
```

---

## Response — POST `/api/predict/`

```json
{
  "success": true,
  "risk_score": 0.327,
  "risk_category": "MODERATE",
  "recommendation": "Moderate cardiovascular risk (32.7%). Consult healthcare provider...",
  "ocr_confidence": {
    "average": 0.89,
    "per_field": { "age": 0.95, "systolic_bp": 0.91 }
  },
  "fields": {
    "age": 58,
    "systolic_bp": 148,
    "total_cholesterol": 225
  },
  "fields_used": ["age", "systolic_bp", "total_cholesterol", "smoking"],
  "explanations": {
    "top_reasons": [
      "Systolic blood pressure 148 mmHg is elevated and increases risk.",
      "Total cholesterol 225 mg/dL is borderline high.",
      "Age 58 years contributes to elevated risk."
    ]
  },
  "audit": {
    "engine": "tesseract_ocr",
    "model_version": "risk_regressor_v2",
    "timestamp": "2026-02-20T14:18:23",
    "document_path": "lab_report.pdf"
  },
  "warnings": [],
  "errors": []
}
```

---

## Risk Category Thresholds

| Category | 10-Year CHD Risk | Description |
|----------|-----------------|-------------|
| 🟢 **LOW** | < 10% | Continue healthy lifestyle, regular check-ups |
| 🟡 **MODERATE** | 10% – 25% | Consult provider, consider lifestyle modifications |
| 🔴 **HIGH** | > 25% | Immediate medical consultation recommended |

*Thresholds follow ACC/AHA Pooled Cohort Equations guidelines.*

---

## Python SDK (CLI)

```python
from src.cardiodetect_v3_pipeline import CardioDetectV3

# Single document
pipeline = CardioDetectV3()
result = pipeline.run("lab_report.pdf")

# Batch processing
results = pipeline.run_batch(["patient_001.pdf", "patient_002.png"])
for r in results:
    print(f"{r['audit']['document_path']}: {r['risk_category']}")
```
