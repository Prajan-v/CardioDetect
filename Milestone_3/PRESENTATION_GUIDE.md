# 🎤 CardioDetect: Individual Project Presentation Guide
## Infosys Expert Panel Defense Preparation

---

# SECTION 1: MY OWNERSHIP NARRATIVE

> **Key Principle:** Frame everything as YOUR deliberate decision. Use "I chose...", "I implemented...", "I optimized..."

## Opening Statement (30 seconds)

*"Good morning. I'm here to present CardioDetect, an AI-powered cardiovascular risk assessment platform that I designed and built from the ground up. This project represents not just code I wrote, but a series of deliberate architectural decisions I made to solve real-world medical ML challenges. I'll walk you through my reasoning for each major decision and demonstrate why I chose specific technologies over alternatives."*

---

# SECTION 2: DEFENDING MY ARCHITECTURAL DECISIONS

## 2.1 Algorithm Choice: XGBoost over Neural Networks

### The Question They'll Ask:
*"Why didn't you use Deep Learning? It's more advanced."*

### My Defense Script:

*"I consciously chose XGBoost over neural networks for three specific reasons:*

**First, dataset size.** I was working with approximately 16,000 samples. Neural networks require millions of samples to generalize well and avoid overfitting. With my dataset size, a neural network would memorize the training data rather than learn patterns. XGBoost, on the other hand, is specifically designed to perform optimally on datasets of this scale.

**Second, interpretability.** In medical applications, I can't just tell a doctor 'the neural network says this patient has 75% risk.' They'll ask 'why?' With XGBoost, I integrated SHAP analysis, which provides a breakdown: 'The patient's age contributed +15% to the risk, their blood pressure +12%, their smoking status +8%.' This level of explainability is non-negotiable in clinical settings.

**Third, the nature of the data.** My data is tabular—structured rows and columns of clinical measurements. Neural networks excel at unstructured data like images, audio, and text. For tabular data, gradient boosting algorithms consistently outperform deep learning in benchmarks. I made my decision based on evidence, not hype."

### Supporting Data:
| Factor | Neural Network | XGBoost (My Choice) |
|--------|----------------|---------------------|
| Best for | Images, text, audio | Tabular data ✅ |
| Samples needed | Millions | Thousands ✅ |
| Training time | Hours/Days | Minutes ✅ |
| Hardware | GPU required | CPU sufficient ✅ |
| Interpretability | Black box | SHAP built-in ✅ |

---

## 2.2 Authentication: JWT over Sessions

### The Question They'll Ask:
*"Why JWT? Wouldn't sessions be simpler?"*

### My Defense Script:

*"I chose JWT over session-based authentication for three architectural reasons:*

**First, statelessness.** With sessions, every request requires a database lookup to verify the session ID. This creates a bottleneck. With JWT, the token itself contains all verification information. Any server can validate it using just the secret key, without touching a database.

**Second, horizontal scalability.** If I need to scale to 10 servers behind a load balancer, sessions require a shared session store—Redis or a database—that all servers can access. This is an additional point of failure and complexity. JWT tokens are self-contained; any server can verify them independently. I designed this system to scale from day one.

**Third, API performance.** Each database round-trip adds 5-10ms of latency. For a session-based system handling 1,000 concurrent users, that's 1,000 extra database queries per second just for authentication. My JWT approach eliminates this overhead entirely."

### My Configuration:
```python
# cardiodetect/settings.py, Lines 178-184
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),   # Short-lived access
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),    # Week-long sessions
    'ROTATE_REFRESH_TOKENS': True,                  # Prevents token reuse attacks
}
```

*"I set access tokens to 24 hours to balance security with user experience. I enabled refresh token rotation so that every time a refresh token is used, it's invalidated and a new one is issued—this prevents stolen tokens from being used indefinitely."*

---

## 2.3 Security: PBKDF2 with 600,000 Iterations

### The Question They'll Ask:
*"Why so many iterations? Isn't that overkill?"*

### My Defense Script:

*"I didn't arbitrarily choose 600,000 iterations—this is Django's current default based on NIST recommendations and industry security standards.*

**The math works like this:** Each iteration takes approximately 0.001 seconds. For a legitimate user logging in, that's 0.6 seconds total—imperceptible. But for an attacker trying to brute-force passwords, they need to compute 600,000 iterations for every guess.

A modern GPU can try billions of simple MD5 hashes per second. With PBKDF2 at 600,000 iterations, that drops to roughly 1,000 guesses per second. At that rate, cracking an 8-character password with letters and numbers would take approximately **19 years**.

I also implemented rate limiting on authentication endpoints—20 attempts per 5 minutes. This means even if an attacker finds a fast way to compute hashes, they can't submit more than 240 guesses per hour to my server. Combined, these measures make brute-force attacks computationally infeasible."

---

## 2.4 OCR: Tesseract Pipeline vs Cloud APIs

### The Question They'll Ask:
*"Why not use AWS Textract or Google Cloud Vision? They'd be more accurate."*

### My Defense Script:

*"I made a deliberate decision to use Tesseract for three critical reasons:*

**First, data privacy.** Medical documents contain Protected Health Information under HIPAA regulations. Sending this data to a third-party cloud API creates compliance concerns. With Tesseract, all processing happens on my own servers. The data never leaves my infrastructure.

**Second, cost efficiency.** Cloud OCR services charge per page or per request. AWS Textract costs approximately $1.50 per 1,000 pages. For a healthcare application processing thousands of documents daily, this adds up quickly. Tesseract is open-source and free—my only cost is compute resources I already have.

**Third, latency.** Cloud APIs add network round-trip time. My on-premise Tesseract pipeline processes documents in 200-500ms. Cloud APIs can take 2-5 seconds including network latency. For user experience, faster is always better.

I compensated for Tesseract's lower baseline accuracy by implementing a robust preprocessing pipeline: adaptive thresholding for varying lighting conditions, deskewing for rotated scans, and multi-mode extraction to find the best result."

---

# SECTION 3: LINE-BY-LINE CODE WALKTHROUGH SCRIPTS

## 3.1 The MLService Singleton

**File:** `services/ml_service.py`, Lines 20-36

```python
class MLService:
    _instance = None  # Line 1
    
    def __new__(cls):  # Line 3
        if cls._instance is None:  # Line 4
            cls._instance = super().__new__(cls)  # Line 5
            cls._instance._initialized = False  # Line 6
        return cls._instance  # Line 7
    
    def __init__(self):  # Line 9
        if self._initialized:  # Line 10
            return  # Line 11
        self._initialized = True  # Line 12
        self._load_pipeline()  # Line 13
```

### My Walkthrough Script:

*"Let me walk you through this code line by line, explaining why each decision matters:*

**Line 1:** I declared `_instance` as a class variable, not an instance variable. This is critical—class variables are shared across all instances, which is exactly what I need for a singleton.

**Line 3:** I override `__new__` instead of just `__init__`. In Python, `__new__` is called BEFORE an object is created—it controls WHETHER an object is created. `__init__` only initializes an already-created object. By overriding `__new__`, I intercept the creation process itself.

**Lines 4-6:** This is the core singleton logic. If no instance exists, I create one and store it in the class variable. I also set an `_initialized` flag to False—this is important because `__init__` will still be called, and I need to track whether first-time initialization has happened.

**Line 7:** Regardless of whether I just created the instance or found an existing one, I return it. Every call to `MLService()` returns the same object.

**Lines 10-13:** In `__init__`, I check the flag. If I've already initialized (loaded the model), I return immediately. This prevents reloading the model on every access. Only the first time do I set the flag and call `_load_pipeline()`.

**Why this matters for performance:** Loading my XGBoost model takes 2-3 seconds. Without this singleton, every user request would wait 3 seconds. With it, only the first request incurs the loading time. Subsequent requests access the already-loaded model in milliseconds."

---

## 3.2 The 0.25 Threshold Optimization

**File:** `Milestone_2/pipeline/integrated_pipeline.py`

```python
def predict_with_optimized_threshold(self, probability, threshold=0.25):
    """
    I use 0.25 instead of the default 0.50 threshold.
    This optimizes for medical sensitivity over accuracy.
    """
    return 1 if probability >= threshold else 0
```

### My Walkthrough Script:

*"This single line—`threshold=0.25`—represents one of the most important decisions I made in this project. Let me explain my reasoning:*

**The default is 0.50.** Most machine learning frameworks predict 'positive' when the probability exceeds 50%. This makes intuitive sense for general classification.

**But medicine is different.** I asked myself: what happens when I'm wrong? 

If I tell a sick patient they're healthy—a **false negative**—they go home without treatment. Their condition worsens. They might die.

If I tell a healthy patient they might be at risk—a **false positive**—they get additional testing. The tests come back negative. They're relieved. Minor inconvenience.

**These errors are NOT equal.** Missing a sick patient is catastrophic. Flagging a healthy patient for extra testing is merely cautious.

So I ran experiments at different thresholds:

| Threshold | Accuracy | Recall (Sick Patients Caught) |
|-----------|----------|-------------------------------|
| 0.50 | 85.00% | 42.10% |
| 0.25 | 76.84% | 62.77% |

**I traded 8% accuracy for 20% more true disease detection.** That's not a bug—that's a design decision. I intentionally made my system more sensitive, accepting more false alarms in exchange for catching more truly sick patients.

**The 0.25 value wasn't arbitrary.** I analyzed the ROC curve and found this was the inflection point where additional sensitivity gains became marginal while false positive rates increased significantly. It's the optimal operating point for a medical screening tool."

---

## 3.3 JWT Authentication Flow

**File:** `accounts/views.py`, Lines 60-76

```python
from rest_framework_simplejwt.tokens import RefreshToken

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        
        user = authenticate(email=email, password=password)  # Line 1
        
        if user is None:
            return Response({'error': 'Invalid credentials'}, status=401)  # Line 2
        
        refresh = RefreshToken.for_user(user)  # Line 3
        
        return Response({
            'tokens': {
                'access': str(refresh.access_token),  # Line 4
                'refresh': str(refresh),  # Line 5
            },
            'user': UserSerializer(user).data
        })
```

### My Walkthrough Script:

*"Let me walk through my authentication flow:*

**Line 1:** I call `authenticate()` which internally uses PBKDF2 hashing. It doesn't compare plain passwords—it hashes the input with the stored salt and compares the result. Even if an attacker stole my database, they'd have hashes, not passwords.

**Line 2:** Security best practice—I give the same error message for 'wrong password' and 'user doesn't exist'. If I said 'user not found' vs 'wrong password', an attacker could enumerate which emails are registered.

**Line 3:** `RefreshToken.for_user()` creates a JWT containing the user ID, expiration time, and other claims. The token is signed with my secret key. Anyone with the token can read the claims, but only my server can verify the signature.

**Lines 4-5:** I return two tokens. The access token (24-hour lifetime) is used for every API request. The refresh token (7-day lifetime) is only used to get new access tokens when the current one expires. This dual-token system means even if an access token is stolen, it's only valid for 24 hours. The refresh token is stored more securely and used less frequently."

---

# SECTION 4: ADVANCED CLINICAL REASONING

## The Precision-Recall Trade-off

### My Explanation Script:

*"As the architect of this system, I faced a fundamental trade-off that every medical ML engineer must understand: Precision versus Recall.*

**Precision** answers: 'Of all patients I flagged as having disease, how many actually have it?'
**Recall** answers: 'Of all patients who actually have disease, how many did I catch?'

These metrics are inversely related. If I'm very cautious and only flag patients I'm 100% sure about, my precision is high but my recall is low—I miss many sick patients. If I flag everyone with even slight risk, my recall is high but my precision is low—I create many false alarms.

**My decision as the architect:** I prioritized Recall.

Here's my reasoning: CardioDetect is a **screening tool**, not a diagnostic tool. Its job is to identify patients who MIGHT need further evaluation. The follow-up care—echocardiograms, stress tests, specialist consultations—provides the diagnosis.

In screening, the cost of missing a sick patient (cardiovascular event, potential death) vastly exceeds the cost of over-referring a healthy patient (unnecessary but harmless testing). This isn't just my opinion—it's the clinical standard. The American Heart Association's risk assessment guidelines follow the same philosophy.

I documented this decision explicitly in my code comments, so future developers understand why I chose the thresholds I did. This is what separates engineering from just coding—I made a judgment call based on domain knowledge and documented my reasoning."

---

# SECTION 5: EXPERT Q&A SIMULATION

## The "Brutal" Questions and My Professional Answers

### Question 1: JWT Secret Key Leak

**Skeptical Architect:** *"What happens if your JWT secret key is leaked? Couldn't an attacker forge tokens for any user?"*

**My Answer:**

*"You've identified a critical risk, and I have mitigations in place at multiple levels.*

**First, prevention.** My secret key is stored as an environment variable, never committed to version control. My `.gitignore` explicitly excludes `.env` files. In production, I would use a secrets management service like AWS Secrets Manager or HashiCorp Vault.

**Second, if a leak is detected.** I designed my system to support key rotation. I would generate a new secret key, deploy it, and issue a forced re-authentication for all users. The leaked key would immediately become useless.

**Third, damage limitation.** My tokens have a 24-hour lifetime. Even with a leaked key, an attacker could only forge tokens that work for 24 hours before needing to forge new ones, giving me a window to detect and respond.

**For a production deployment,** I would implement additional measures: asymmetric keys (RS256 instead of HS256), token blacklisting for revocation, and anomaly detection to catch suspicious token usage patterns."

---

### Question 2: Biased Medical Data

**Skeptical Architect:** *"How does your model handle biased medical data? The Framingham study was predominantly white Americans from the 1950s."*

**My Answer:**

*"This is an excellent and important point. Medical AI bias is a known issue, and I want to be transparent about my approach:*

**First, acknowledging the limitation.** You're correct—the Framingham Heart Study population was primarily white Americans. My model may perform differently on other populations. I documented this limitation explicitly in my technical documentation.

**Second, mitigation strategies I implemented.** I supplemented Framingham data with additional datasets—the UCI Heart Disease dataset from four international hospitals, and Kaggle datasets with more diverse populations. This provides some cross-population validation, though it's not perfect.

**Third, what I would do for production.** Before deploying to a specific population, I would validate the model on a representative sample from that population. If performance differs significantly, I would collect additional local data and either retrain or calibrate the model.

**Fourth, ongoing monitoring.** I designed my prediction storage to include demographics. In production, I could analyze model performance stratified by age, sex, and ethnicity, alerting us to any disparities that emerge.

I don't claim to have solved the bias problem—no one has. But I've acknowledged it, documented it, and designed systems to detect and address it over time."

---

### Question 3: Model Interpretability Challenge

**Skeptical Architect:** *"You mentioned SHAP for interpretability. But what if SHAP tells you a feature is important for a prediction that doesn't make clinical sense? Does your model override clinical judgment?"*

**My Answer:**

*"Absolutely not. My model is designed to augment clinical judgment, never replace it.*

**The Clinical Advisor module exemplifies this philosophy.** It doesn't just output risk scores—it provides specific, evidence-based recommendations tied to established guidelines like ACC/AHA. A clinician can review both the ML prediction AND the guideline-based advice, then make their own judgment.

**If SHAP identifies a clinically nonsensical factor,** that's actually valuable information. It might indicate a data quality issue, a spurious correlation in the training data, or a feature that should be removed. The interpretability isn't just for patients—it's for model debugging.

**My deployment philosophy:** The model provides a recommendation. The clinical decision-maker sees the recommendation, the confidence level, and the SHAP explanation. They can accept it, modify it, or reject it entirely. The final decision always rests with the human.

This is critical for legal and ethical reasons. AI can't be held liable for medical decisions—physicians can. I designed my system to support physicians, not supplant them."

---

### Question 4: Handling Concurrent Requests

**Skeptical Architect:** *"Your Singleton pattern loads the model once. What happens under high concurrency? Is XGBoost thread-safe? Could two requests corrupt each other's predictions?"*

**My Answer:**

*"Great question about thread safety.*

**XGBoost prediction is thread-safe by default.** The `predict()` method only reads from the model's internal data structures—it doesn't modify them. Multiple threads can call predict simultaneously without issues.

**My Django deployment uses Gunicorn with multiple workers.** Each worker is a separate Python process with its own memory space. Within a worker, multiple requests share the same singleton model, but XGBoost handles this correctly.

**Where I was careful:** My MLService class has mutable state for the loaded model. I ensured that after initialization, this state is only read, never written. The initialization itself uses Python's GIL (Global Interpreter Lock), which prevents race conditions during the first load.

**For extreme scale,** I would consider model-as-a-service architectures like TensorFlow Serving or Triton Inference Server, which are specifically designed for high-throughput, low-latency inference. But for my current scale—hundreds of requests per minute—the Singleton pattern is efficient and correct."

---

### Question 5: Recovery from Model Failures

**Skeptical Architect:** *"What happens if your ML model fails during a prediction? Does the entire request crash? How do you handle graceful degradation?"*

**My Answer:**

*"I implemented defensive error handling throughout the prediction pipeline:*

**At the model level:** My prediction code is wrapped in try-except blocks. If the model throws an exception (which should never happen with clean input, but edge cases exist), I catch it, log detailed diagnostics, and return a user-friendly error rather than a stack trace.

**At the validation level:** Most failures happen because of bad input—missing fields, wrong types, out-of-range values. My serializers validate every input before it reaches the model. Invalid requests are rejected immediately with specific error messages.

**Graceful degradation philosophy:** If the ML prediction fails but I have valid input, I could still provide the Clinical Advisor's guideline-based recommendations. A patient with blood pressure of 180/110 doesn't need ML to know they have hypertensive crisis—the guidelines are explicit.

**Monitoring and alerting:** In production, I would integrate with monitoring systems (Datadog, New Relic) to track prediction latencies, error rates, and model performance over time. Any degradation triggers alerts before users are significantly impacted.

I designed for failure because in production, failure is inevitable. The question is how gracefully you handle it."

---

# SECTION 6: PRESENTATION BEST PRACTICES

## The STAR Method for Challenges

**When they ask:** *"Tell me about a challenge you faced."*

**Format:** Situation → Task → Action → Result

**Example Response:**

*"During development, I noticed my model had 85% accuracy but only 42% recall (Situation). I needed to catch more heart disease cases because false negatives are clinically unacceptable (Task). I analyzed the ROC-AUC curve, experimented with multiple thresholds, and determined that 0.25 was the optimal operating point for a screening tool (Action). This increased my sensitivity from 42% to 63%, catching 20% more true disease cases while accepting a manageable increase in false positives (Result)."*

## Visual Over Text

**Don't show:** Bullet points listing your technologies

**Do show:** A data flow diagram

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Medical   │───▶│     OCR      │───▶│   XGBoost    │───▶│   Clinical   │
│   Report    │    │  (Tesseract) │    │   Models     │    │   Advisor    │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                          │                   │                    │
                          ▼                   ▼                    ▼
                   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                   │ Preprocessing│    │ Risk Score   │    │ Personalized │
                   │  • Deskew    │    │ LOW/MED/HIGH │    │ Recommendations│
                   │  • Threshold │    │              │    │              │
                   └──────────────┘    └──────────────┘    └──────────────┘
```

## Handling Unknown Questions

**When you don't know:** Don't guess. Be honest.

**Template Response:**

*"That's an excellent point regarding [topic]. In the current version, I focused on [what you did focus on], but I acknowledge that [the topic they raised] is important for a production-grade system. To address it, I would explore [reasonable approach], potentially leveraging [relevant technology or method]. I'd be happy to research this further and provide a detailed proposal."*

**Example:**

*"That's an excellent point regarding model drift detection. In the current version, I focused on building a robust initial model with proper validation. For a production-grade system, I would implement monitoring that tracks prediction distributions over time, alerting us when input patterns shift significantly from training data. I'd likely use a tool like Evidently AI or implement statistical tests like the Kolmogorov-Smirnov test. I'd be happy to add this to my roadmap."*

---

# QUICK REFERENCE CARD

## My Technology Choices and Why

| Component | I Chose | Over | Because |
|-----------|---------|------|---------|
| Algorithm | XGBoost | Neural Networks | 16k samples, interpretability, tabular data |
| Auth | JWT | Sessions | Stateless, scalable, fast |
| Hashing | PBKDF2 600k | Simple hash | Brute-force prevention |
| OCR | Tesseract | Cloud APIs | Privacy (HIPAA), cost |
| Threshold | 0.25 | 0.50 | Recall > Precision for screening |

## My Key Metrics

| What | Value |
|------|-------|
| Detection Accuracy | 91.45% |
| Prediction Accuracy | 91.63% |
| Backend Tests | 41 (Django) |
| Frontend Tests | 60 (Jest) |
| **Total Tests** | **101** |
| API Endpoints | 32+ |
| React Components | 19 |
| Caching | Redis |

## Files They Might Ask About

| Topic | File | Key Lines |
|-------|------|-----------|
| Singleton | `services/ml_service.py` | 20-36 |
| JWT Config | `cardiodetect/settings.py` | 178-184 |
| Rate Limiting | `cardiodetect/middleware.py` | 12-77 |
| Threshold | `integrated_pipeline.py` | 557-564 |

---

# CLOSING STATEMENT

*"CardioDetect represents not just technical implementation, but thoughtful architectural decision-making. Every component—from my choice of XGBoost over neural networks to my decision to lower the prediction threshold—was a deliberate choice backed by data, domain knowledge, and engineering judgment. I'm proud of what I built, and I'm excited to answer any questions you have about my approach."*

---

**Good luck! You've got this. 🎯**
