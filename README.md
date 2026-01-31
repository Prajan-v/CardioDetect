# AI-Powered Application for Early Detection of Heart Disease Risk


**CardioDetect** is an AI-powered Clinical Decision Support System (CDSS) designed to assist medical professionals in the early detection and risk stratification of cardiovascular diseases. 

> **Note:** This project was conceptualized and developed as part of the **Infosys Springboard Internship Program (2025-2026)**.

## 🚀 Key Features

### 1. Dual-Engine Architecture
- **Disease Detection Module (Screening):** A Voting Ensemble model (Random Forest + Gradient Boosting) for immediate diagnosis.
  - *Accuracy:* **91.30%**
- **Risk Stratification Module (Forecasting):** An XGBoost-based engine to predict 10-year cardiovascular risk.
  - *Risk Category Agreement:* **91.63%**

### 2. Hybrid OCR Pipeline
- Digitizes legacy medical records (scanned PDFs/images) using **Tesseract OCR** and OpenCV preprocessing.
- Extracts clinical values (Cholesterol, BP, Age) automatically to populate the analysis form.

### 3. Explainable AI (XAI)
- Integrated **SHAP (SHapley Additive exPlanations)** to visualize feature importance.
- Provides "Glass Box" transparency, showing clinicians *why* a specific risk score was assigned.

## 🛠️ Tech Stack
- **Frontend:** React.js, Tailwind CSS
- **Backend:** Django REST Framework (Python)
- **ML Engine:** Scikit-learn, XGBoost, SHAP
- **OCR:** Tesseract 4.0, PyMuPDF, OpenCV
- **Database:** PostgreSQL , Redis


## ⚠️ Disclaimer
This system is a **research prototype** developed for educational and experimental purposes. It is **not** a certified medical device and should not be used for direct patient diagnosis or treatment without professional medical supervision.

## 📜 License
This project is open-source and available under the MIT License.
