# CardioDetect - Complete System Documentation

---

## 1. AUTHENTICATION & SECURITY

### Login System
| Feature | Implementation |
|---------|----------------|
| **Authentication** | JWT (JSON Web Token) |
| **Access Token Expiry** | 24 hours |
| **Refresh Token Expiry** | 7 days |
| **Password Hashing** | PBKDF2 with SHA256 (600,000 iterations) |
| **Max Failed Logins** | 5 attempts |
| **Lockout Duration** | 30 minutes |
| **Rate Limiting** | 1000 requests/hour (authenticated) |

### User Roles
| Role | Access |
|------|--------|
| **Patient** | Personal predictions, upload reports |
| **Doctor** | + Dashboard, view assigned patients |
| **Admin** | + Django admin, system settings, user approvals |

---

## 2. FRONTEND (Next.js 16 + React 19)

### Pages
| Route | Purpose | Access |
|-------|---------|--------|
| `/` | Landing page | Public |
| `/login` | User login | Public |
| `/register` | User registration | Public |
| `/forgot-password` | Password reset request | Public |
| `/dashboard` | Manual input form | Auth required |
| `/dashboard/upload` | OCR document upload | Auth required |
| `/dashboard/history` | Prediction history | Auth required |
| `/profile` | User profile & settings | Auth required |
| `/settings` | Account settings | Auth required |
| `/admin-dashboard` | System stats | Doctor/Admin only |
| `/doctor` | Doctor patient management | Doctor only |

### Components (19)
| Component | Description |
|-----------|-------------|
| `AnimatedHeart` | Pulsing heart SVG animation |
| `ECGLine` | Moving ECG waveform |
| `DragDropZone` | File upload with drag-and-drop |
| `RiskGauge` | Circular progress indicator |
| `FactorChart` | Risk factor bar chart |
| `PredictionHistory` | Recent predictions list |
| `FloatingParticles` | Background particle animation |
| `Shimmer` | Loading skeleton |
| `AdminPanel` | System stats component |
| `NotificationBell` | Notification dropdown |
| `ThemeToggle` | Dark/light mode switch |
| `AnatomicalHeart` | SVG heart illustration |
| `AnimatedCounter` | Number animation |
| `FeatureCard` | Feature showcase card |
| `HeartWithStethoscope` | Logo component |
| `NotificationPopup` | Toast notifications |
| `StethoscopeHeartLogo` | Branding logo |

### UI Features
| Feature | Status |
|---------|--------|
| Glassmorphism design | ✅ |
| Gradient animations | ✅ |
| Dark mode (default) | ✅ |
| Toast notifications | ✅ |
| SHAP visualization | ✅ |
| Inline field editing | ✅ |
| Responsive layout | ✅ |
| Framer Motion animations | ✅ |

---

## 3. BACKEND (Django 6)

### API Endpoints (32+)
| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/auth/login/` | ❌ | Get JWT tokens |
| POST | `/api/auth/register/` | ❌ | Create account |
| POST | `/api/auth/refresh/` | ❌ | Refresh token |
| GET | `/api/auth/profile/` | ✅ | Get user profile |
| PATCH | `/api/auth/profile/` | ✅ | Update profile |
| POST | `/api/auth/password-change/` | ✅ | Change password |
| POST | `/api/auth/password-reset/` | ❌ | Request reset |
| POST | `/api/predict/manual/` | ✅ | Manual prediction |
| POST | `/api/predict/ocr/` | ✅ | OCR prediction |
| GET | `/api/predict/history/` | ✅ | Prediction history |
| GET | `/api/predict/{id}/` | ✅ | Prediction detail |
| DELETE | `/api/predict/{id}/` | ✅ | Delete prediction |
| GET | `/api/predict/statistics/` | ✅ | User statistics |
| GET | `/api/predict/dashboard/` | ✅ | Dashboard data |
| GET | `/api/notifications/` | ✅ | User notifications |
| POST | `/api/auth/data-export/` | ✅ | GDPR data export |
| DELETE | `/api/auth/delete-account/` | ✅ | GDPR deletion |
| GET | `/api/health/` | ❌ | Health check |

### Database Models
| Model | Fields |
|-------|--------|
| **User** | email, password, role, phone, email_verified, created_at |
| **Prediction** | user, risk_category, risk_percentage, input_data, created_at |
| **MedicalDocument** | file, ocr_status, confidence, extracted_text |
| **AuditLog** | user, action, timestamp, ip_address |
| **LoginHistory** | user, success, ip, timestamp |
| **Notification** | user, message, read, created_at |
| **DoctorPatient** | doctor, patient, assigned_at |

### Database & Caching
| Component | Technology | Purpose |
|-----------|------------|---------|
| Primary DB | SQLite3 | User accounts, predictions, audit logs |
| Cache Layer | **Redis** | Session data, rate limiting, caching |
| Location | `Milestone_3/db.sqlite3` | Development database |

---

## 4. MACHINE LEARNING MODELS

### Detection Model (Heart Disease)
| Attribute | Value |
|-----------|-------|
| Algorithm | XGBoost Classifier |
| Dataset | UCI Heart Disease (303 samples) |
| Features | 13 clinical parameters |
| Accuracy | **91.45%** |
| Output | Disease / No Disease |

### Prediction Model (10-Year Risk)
| Attribute | Value |
|-----------|-------|
| Algorithm | XGBoost Classifier |
| Dataset | Framingham (5000 samples) |
| Features | 8 risk factors |
| Accuracy | **91.63%** |
| Output | 10-year CHD Risk % |

### Risk Categories
| Category | 10-Year Risk | Color |
|----------|--------------|-------|
| LOW | < 10% | 🟢 Green |
| MODERATE | 10-20% | 🟡 Yellow |
| HIGH | > 20% | 🔴 Red |

### Explainability (SHAP)
| Feature | Description |
|---------|-------------|
| **Library** | SHAP (SHapley Additive exPlanations) |
| **Purpose** | Shows which features contributed to prediction |
| **Output** | Bar chart of feature importance |
| **Clinical Use** | Helps doctors understand AI decisions |

---

## 5. OCR PIPELINE

### Processing Stages
1. **Preprocessing** - Noise removal, deskew, contrast enhancement
2. **Text Extraction** - Tesseract OCR, multi-mode processing
3. **Field Parsing** - Regex + fuzzy matching for medical terms
4. **Post-Processing** - Unit conversion, validation, confidence scoring

### Extracted Fields
| Field | Unit | Example |
|-------|------|---------|
| Age | years | 55 |
| Systolic BP | mmHg | 140 |
| Diastolic BP | mmHg | 90 |
| Total Cholesterol | mg/dL | 220 |
| HDL Cholesterol | mg/dL | 45 |
| Fasting Glucose | mg/dL | 100 |
| Heart Rate | bpm | 72 |
| BMI | kg/m² | 27.5 |
| Smoking Status | yes/no | no |
| Diabetes | yes/no | no |

---

## 6. TESTING (101 AUTOMATED TESTS)

### Backend Tests (41)
| Category | Tool | Count |
|----------|------|-------|
| Auth API Tests | Django APITestCase | 9 |
| Prediction API Tests | Django APITestCase | 13 |
| Email Service Tests | Django TestCase | 19 |

### Frontend Tests (60)
| Category | Tool | Count |
|----------|------|-------|
| Component Tests | Jest + RTL | 22 |
| Auth Page Tests | Jest + RTL | 17 |
| App Page Tests | Jest + RTL | 21 |

### Run Commands
```bash
# Backend
cd Milestone_3 && python manage.py test

# Frontend
cd Milestone_3/frontend && npm test
```

---

## 7. FILE LIMITS

| Setting | Value |
|---------|-------|
| Max upload size | 10 MB |
| Supported formats | PDF, PNG, JPG, JPEG |

---

## 8. ACCESS POINTS

| Resource | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Django Admin | http://localhost:8000/admin |
| Health Check | http://localhost:8000/api/health/ |

---

## 9. STARTUP

```bash
# Start Backend
cd CardioDetect/Milestone_3
source venv/bin/activate
python manage.py runserver

# Start Frontend (new terminal)
cd CardioDetect/Milestone_3/frontend
npm run dev

# Run Tests
python manage.py test    # Backend
npm test                  # Frontend
```

### Login Credentials
| Email | Password | Role |
|-------|----------|------|
| prajan@cardiodetect.com | CardioDetect@ | Admin |

---

## 10. PROJECT STRUCTURE

```
CardioDetect/
├── PROJECT_INFO.md             # This file
├── Milestone_2/                # ML Pipeline
│   ├── models/                 # Trained models (.pkl)
│   ├── pipeline/               # integrated_pipeline.py
│   └── ultra_ocr.py            # OCR processor
└── Milestone_3/                # Web Application
    ├── cardiodetect/           # Django settings
    │   ├── settings.py         # Config (Redis, JWT, etc.)
    │   └── middleware.py       # Rate limiting, security
    ├── accounts/               # User auth app
    │   ├── views.py            # Auth API endpoints
    │   └── tests.py            # 9 auth tests
    ├── predictions/            # Predictions app
    │   ├── views.py            # Prediction API endpoints
    │   └── tests.py            # 13 prediction tests
    ├── services/               # Business logic
    │   └── ml_service.py       # MLService singleton
    ├── reports/                # PDF generation
    │   └── MILESTONE_3_REPORT.pdf
    └── frontend/               # Next.js 16 app
        ├── src/app/            # Pages (file-based routing)
        ├── src/components/     # 19 React components
        ├── jest.config.js      # Test config
        └── src/__tests__/      # 60 Jest tests
```

---

*Last updated: December 2024*
