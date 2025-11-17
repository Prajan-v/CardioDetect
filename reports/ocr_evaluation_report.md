# OCR Evaluation Report

## System Architecture
Pipeline uses PyMuPDF for PDF rasterization, OpenCV for preprocessing, and Tesseract for OCR.

## Testing Methodology
Evaluated on programmatically generated documents (PNG and PDF) with known ground truth values.
This provides a deterministic baseline for field-level accuracy.

## Ground Truth
- bp_sys: 120.0
- bp_dia: 80.0
- chol: 200.0
- hdl: 60.0
- ldl: 120.0
- triglycerides: 150.0
- glucose: 100.0

## Results (PNG)
Overall accuracy: 100.0%
| field         |   pred |   gt |   correct |
|:--------------|-------:|-----:|----------:|
| bp_sys        |    120 |  120 |         1 |
| bp_dia        |     80 |   80 |         1 |
| chol          |    200 |  200 |         1 |
| hdl           |     60 |   60 |         1 |
| ldl           |    120 |  120 |         1 |
| triglycerides |    150 |  150 |         1 |
| glucose       |    100 |  100 |         1 |

## Results (PDF)
Overall accuracy: 100.0%
| field         |   pred |   gt |   correct |
|:--------------|-------:|-----:|----------:|
| bp_sys        |    120 |  120 |         1 |
| bp_dia        |     80 |   80 |         1 |
| chol          |    200 |  200 |         1 |
| hdl           |     60 |   60 |         1 |
| ldl           |    120 |  120 |         1 |
| triglycerides |    150 |  150 |         1 |
| glucose       |    100 |  100 |         1 |

## Error Analysis and Confidence
Errors typically arise from font rendering and thresholding in synthetic samples; real-world scans may introduce skew and noise.
Confidence can be approximated via Tesseract confidences or ensemble OCR; not implemented in this milestone.

## Limitations and Mitigation
- Synthetic tests underrepresent real-world variability; add more diverse scanned samples.
- Layout-aware parsing (tables) and spell-check normalization improve robustness.