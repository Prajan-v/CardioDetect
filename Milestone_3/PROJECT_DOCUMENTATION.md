# 🫀 CardioDetect - Complete Technical Documentation

---

# PART 1: PROJECT OVERVIEW

## What is CardioDetect?

CardioDetect is an **AI-powered cardiovascular disease risk prediction system** that:
- Predicts 10-year heart disease risk using machine learning
- Processes medical documents using OCR
- Provides personalized health recommendations
- Supports patient-doctor collaboration

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend | Django 4.2, Django REST Framework |
| Database | SQLite (dev) / PostgreSQL (prod) |
| ML | scikit-learn, XGBoost, LightGBM |
| OCR | Tesseract, OpenCV |

---

# PART 2: COMPLETE DIRECTORY STRUCTURE

```
CardioDetect/
│
├── Milestone_2/                          # ML Pipeline
│   ├── models/                           # Trained ML Models
│   │   ├── detection/                    # Heart disease detection
│   │   │   ├── detection_xgb.pkl         # XGBoost model (91.45%)
│   │   │   ├── detection_lgbm.pkl        # LightGBM model
│   │   │   ├── detection_rf.pkl          # Random Forest
│   │   │   └── detection_scaler.pkl      # Feature scaler
│   │   │
│   │   └── prediction/                   # 10-year risk prediction
│   │       ├── prediction_xgb.pkl        # XGBoost (94.01%)
│   │       ├── prediction_lgbm.pkl       # LightGBM
│   │       ├── prediction_rf.pkl         # Random Forest
│   │       └── prediction_scaler.pkl     # Feature scaler
│   │
│   ├── pipeline/                         # Inference Pipelines
│   │   ├── integrated_pipeline.py        # Full pipeline
│   │   ├── detection_pipeline.py         # Detection only
│   │   └── prediction_pipeline.py        # Prediction only
│   │
│   ├── Source_Code/src/                  # Core ML Code
│   │   ├── preprocessing.py              # Data preprocessing
│   │   ├── ensembles.py                  # Ensemble methods
│   │   ├── evaluation.py                 # Metrics
│   │   └── risk_scoring.py               # Risk calculation
│   │
│   ├── ocr/                              # OCR Components
│   │   └── Final_ocr/
│   │       └── production_ocr.py         # Production OCR
│   │
│   └── experiments/                      # Training Scripts
│       ├── train_cv_ensemble.py          # Ensemble training
│       └── tune_ensemble.py              # Hyperparameter tuning
│
│
├── Milestone_3/                          # Web Application
│   │
│   ├── accounts/                         # User Management App
│   │   ├── models.py                     # User, LoginHistory models
│   │   ├── views.py                      # Auth API views
│   │   ├── serializers.py                # DRF serializers
│   │   └── urls.py                       # Auth routes
│   │
│   ├── predictions/                      # Predictions App
│   │   ├── models.py                     # Prediction, MedicalDocument
│   │   ├── views.py                      # Prediction API views
│   │   ├── serializers.py                # Prediction serializers
│   │   └── ml_service.py                 # ML integration
│   │
│   ├── cardiodetect/                     # Django Project
│   │   ├── settings.py                   # Configuration
│   │   └── urls.py                       # Main URL routing
│   │
│   ├── frontend/                         # Next.js Frontend
│   │   ├── src/
│   │   │   ├── app/                      # Pages (App Router)
│   │   │   │   ├── page.tsx              # Landing page
│   │   │   │   ├── login/page.tsx        # Login
│   │   │   │   ├── register/page.tsx     # Registration
│   │   │   │   ├── dashboard/page.tsx    # Patient dashboard
│   │   │   │   └── doctor/
│   │   │   │       └── dashboard/page.tsx# Doctor portal
│   │   │   │
│   │   │   ├── components/               # Reusable components
│   │   │   │   ├── AnimatedHeart.tsx     # Heart animation
│   │   │   │   ├── PredictionHistory.tsx # History display
│   │   │   │   └── NotificationBell.tsx  # Notifications
│   │   │   │
│   │   │   └── services/                 # API clients
│   │   │       └── auth.ts               # Auth service
│   │   │
│   │   └── .env.local                    # Frontend config
│   │
│   ├── db.sqlite3                        # Database
│   ├── requirements.txt                  # Python deps
│   └── manage.py                         # Django CLI
│
└── start.sh                              # Start script
```

---

# PART 3: DATA PIPELINE (ML)

## 3.1 Data Sources

| Dataset | Location | Records | Purpose |
|---------|----------|---------|---------|
| UCI Heart Disease | External | 303 | Detection training |
| Framingham Heart Study | External | 11,500 | Prediction training |
| Kaggle Heart Disease | External | 1,190 | Validation |

## 3.2 Data Preprocessing Code

**File:** `Milestone_2/Source_Code/src/preprocessing.py`

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define feature columns
NUMERIC_FEATURES = [
    'age', 'systolic_bp', 'diastolic_bp', 
    'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol',
    'triglycerides', 'fasting_glucose', 'bmi', 'heart_rate'
]

CATEGORICAL_FEATURES = ['sex', 'smoking', 'diabetes', 'hypertension']

# Create preprocessing pipeline
def create_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])
```

## 3.3 Feature Engineering

```python
def engineer_features(df):
    # Pulse Pressure (arterial stiffness indicator)
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # Mean Arterial Pressure
    df['map'] = df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3
    
    # Cholesterol Ratio (atherogenic index)
    df['cholesterol_ratio'] = df['total_cholesterol'] / df['hdl_cholesterol']
    
    return df
```

---

# PART 4: MACHINE LEARNING MODELS

## 4.1 Detection Model (Heart Disease Detection)

**Files:**
- Training: `Milestone_2/experiments/train_classification.py`
- Model: `Milestone_2/models/detection/detection_xgb.pkl`
- Pipeline: `Milestone_2/pipeline/detection_pipeline.py`

**Accuracy: 91.45%**

```python
# Milestone_2/pipeline/detection_pipeline.py

class DetectionPipeline:
    def __init__(self):
        self.model = joblib.load('models/detection/detection_xgb.pkl')
        self.scaler = joblib.load('models/detection/detection_scaler.pkl')
    
    def predict(self, patient_data: dict) -> dict:
        features = self._prepare_features(patient_data)
        scaled = self.scaler.transform(features)
        
        probability = self.model.predict_proba(scaled)[0][1]
        prediction = probability >= 0.5
        
        return {
            'disease_detected': bool(prediction),
            'probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 2)
        }
```

## 4.2 Prediction Model (10-Year Risk)

**Files:**
- Training: `Milestone_2/experiments/train_cv_ensemble.py`
- Model: `Milestone_2/models/prediction/prediction_xgb.pkl`

**Accuracy: 94.01%**

```python
# Milestone_2/pipeline/prediction_pipeline.py

class PredictionPipeline:
    def __init__(self):
        self.models = {
            'xgb': joblib.load('models/prediction/prediction_xgb.pkl'),
            'lgbm': joblib.load('models/prediction/prediction_lgbm.pkl'),
            'rf': joblib.load('models/prediction/prediction_rf.pkl')
        }
    
    def predict(self, patient_data: dict) -> dict:
        # Ensemble prediction (average of models)
        predictions = [m.predict_proba(scaled)[0][1] for m in self.models.values()]
        risk_probability = np.mean(predictions)
        risk_percentage = risk_probability * 100
        
        # Risk categorization
        if risk_percentage < 10:
            category = 'LOW'
        elif risk_percentage < 25:
            category = 'MODERATE'
        else:
            category = 'HIGH'
        
        return {
            'risk_probability': float(risk_probability),
            'risk_percentage': float(risk_percentage),
            'risk_category': category
        }
```

## 4.3 Risk Thresholds

| Category | Risk Range | Clinical Action |
|----------|------------|-----------------|
| 🟢 LOW | <10% | Lifestyle advice, annual checkup |
| 🟡 MODERATE | 10-25% | Enhanced monitoring |
| 🔴 HIGH | ≥25% | Specialist referral |

---

# PART 5: DATABASE MODELS

## 5.1 User Model

**File:** `Milestone_3/accounts/models.py`

```python
class User(AbstractUser):
    class Role(models.TextChoices):
        PATIENT = 'patient', 'Patient'
        DOCTOR = 'doctor', 'Doctor'
        ADMIN = 'admin', 'Administrator'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    email = models.EmailField(unique=True)
    USERNAME_FIELD = 'email'
    
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    phone = models.CharField(max_length=20, blank=True)
    date_of_birth = models.DateField(null=True)
    gender = models.CharField(max_length=1)
    
    role = models.CharField(max_length=10, choices=Role.choices)
    
    # Doctor fields
    license_number = models.CharField(max_length=50, blank=True)
    specialization = models.CharField(max_length=100, blank=True)
    hospital = models.CharField(max_length=200, blank=True)
    
    # Security
    email_verified = models.BooleanField(default=False)
    failed_login_attempts = models.IntegerField(default=0)
```

## 5.2 Prediction Model

**File:** `Milestone_3/predictions/models.py`

```python
class Prediction(models.Model):
    class RiskCategory(models.TextChoices):
        LOW = 'LOW', 'Low Risk'
        MODERATE = 'MODERATE', 'Moderate Risk'
        HIGH = 'HIGH', 'High Risk'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    model_type = models.CharField(max_length=15)  # detection, prediction, both
    input_method = models.CharField(max_length=10)  # manual, ocr
    
    risk_probability = models.FloatField()
    risk_percentage = models.FloatField()
    risk_category = models.CharField(max_length=10, choices=RiskCategory.choices)
    
    disease_detected = models.BooleanField(null=True)
    confidence_score = models.FloatField(default=0.85)
    
    input_data = models.JSONField()
    lifestyle_recommendations = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
```

## 5.3 Medical Document Model

```python
class MedicalDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    file = models.FileField(upload_to='documents/')
    file_type = models.CharField(max_length=10)  # pdf, png, jpg
    
    ocr_status = models.CharField(max_length=15)  # pending, success, failed
    extracted_data = models.JSONField(null=True)
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
```

## 5.4 Doctor-Patient Relationship

```python
class DoctorPatient(models.Model):
    doctor = models.ForeignKey(User, related_name='patients')
    patient = models.ForeignKey(User, related_name='doctors')
    
    status = models.CharField(max_length=10)  # pending, active, revoked
    can_view_history = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ['doctor', 'patient']
```

---

# PART 6: BACKEND API

## 6.1 URL Configuration

**File:** `Milestone_3/cardiodetect/urls.py`

```python
urlpatterns = [
    path('api/auth/', include('accounts.urls')),
    path('api/', include('predictions.urls')),
    path('admin/', admin.site.urls),
]
```

## 6.2 Authentication Views

**File:** `Milestone_3/accounts/views.py`

```python
class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        user = authenticate(email=email, password=password)
        
        if user is None:
            return Response({'error': 'Invalid credentials'}, status=401)
        
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': UserSerializer(user).data
        })
```

## 6.3 Prediction Views

**File:** `Milestone_3/predictions/views.py`

```python
class PredictView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        input_data = request.data
        mode = input_data.get('mode', 'prediction')
        
        from .ml_service import get_prediction
        result = get_prediction(input_data, mode)
        
        prediction = Prediction.objects.create(
            user=request.user,
            model_type=mode,
            input_data=input_data,
            risk_probability=result['risk_probability'],
            risk_percentage=result['risk_percentage'],
            risk_category=result['risk_category']
        )
        
        return Response({'prediction': PredictionSerializer(prediction).data})
```

---

# PART 7: FRONTEND UI

## 7.1 Dashboard Page

**File:** `Milestone_3/frontend/src/app/dashboard/page.tsx`

```tsx
'use client';

export default function DashboardPage() {
    const [mode, setMode] = useState<'detection' | 'prediction'>('prediction');
    const [formData, setFormData] = useState({});
    const [result, setResult] = useState(null);
    
    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        const token = localStorage.getItem('auth_token');
        const response = await fetch('/api/predict/', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ...formData, mode })
        });
        
        const data = await response.json();
        setResult(data.prediction);
    };
    
    return (
        <div className="min-h-screen bg-[#0a0a1a]">
            {/* Navigation */}
            {/* Form */}
            {/* Results Display */}
        </div>
    );
}
```

## 7.2 Auth Service

**File:** `Milestone_3/frontend/src/services/auth.ts`

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL;

export async function login(credentials: { email: string; password: string }) {
    const response = await fetch(`${API_BASE}/auth/login/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(credentials)
    });
    
    const data = await response.json();
    
    if (response.ok) {
        localStorage.setItem('auth_token', data.access);
        localStorage.setItem('user', JSON.stringify(data.user));
        return { success: true, user: data.user };
    }
    
    return { success: false, error: data.error };
}
```

---

# PART 8: END-TO-END DATA FLOW

```
USER REGISTRATION
┌─────────────┐    POST /auth/register/    ┌─────────────────┐
│  Register   │ ────────────────────────────▶│  Create User    │
│    Form     │                              │  in Database    │
└─────────────┘                              └─────────────────┘

USER LOGIN
┌─────────────┐    POST /auth/login/        ┌─────────────────┐
│   Login     │ ────────────────────────────▶│  Return JWT     │
│    Form     │                              │  Tokens         │
└─────────────┘                              └─────────────────┘

PREDICTION FLOW
┌─────────────┐    POST /api/predict/       ┌─────────────────┐
│  Dashboard  │ ────────────────────────────▶│  ML Pipeline    │
│   Form      │                              │  Predict Risk   │
└─────────────┘                              └────────┬────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │  Save to DB     │
                                             │  predictions    │
                                             └────────┬────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │  Display Result │
                                             │  🟢 18.5% LOW   │
                                             └─────────────────┘
```

---

# PART 9: HOW TO RUN

```bash
# Start both backend and frontend
./start.sh

# Or manually:
cd Milestone_3 && python manage.py runserver  # Backend: localhost:8000
cd Milestone_3/frontend && npm run dev         # Frontend: localhost:3000
```

## Test Credentials

| Role | Email | Password |
|------|-------|----------|
| Patient | patient@cardiodetect.com | Patient@123 |
| Doctor | doctor@cardiodetect.com | Doctor@123 |

---

# PART 10: PROJECT METRICS

| Metric | Value |
|--------|-------|
| Total Files | ~105 |
| Lines of Code | ~25,000 |
| Database Tables | 7 |
| API Endpoints | 24 |
| ML Accuracy | 91.45% / 94.01% |
| ESLint Errors | 0 ✅ |

---

**Document Version:** 2.0  
**Created:** December 2024
