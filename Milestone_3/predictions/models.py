"""
Prediction and Medical Document models for CardioDetect.
Enhanced with complete features for Milestone 3.
"""

from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid


class MedicalDocument(models.Model):
    """Uploaded medical documents for OCR processing."""
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        SUCCESS = 'success', 'Success'
        FAILED = 'failed', 'Failed'
    
    class FileType(models.TextChoices):
        PDF = 'pdf', 'PDF Document'
        PNG = 'png', 'PNG Image'
        JPG = 'jpg', 'JPEG Image'
        OTHER = 'other', 'Other'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='documents'
    )
    filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/%Y/%m/')
    file_type = models.CharField(
        max_length=20,
        choices=FileType.choices,
        default=FileType.OTHER
    )
    file_size = models.IntegerField(validators=[MinValueValidator(0)])  # in bytes
    
    # OCR Processing
    ocr_status = models.CharField(
        max_length=15,
        choices=Status.choices,
        default=Status.PENDING
    )
    ocr_confidence = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    ocr_method = models.CharField(max_length=50, blank=True)
    extracted_text = models.TextField(blank=True)
    extracted_fields = models.JSONField(default=dict, blank=True)
    
    # Metadata
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    processing_time_ms = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['user', '-uploaded_at']),
            models.Index(fields=['ocr_status']),
        ]
    
    def __str__(self):
        return f"{self.filename} - {self.ocr_status}"
    
    @property
    def file_size_formatted(self):
        """Return human-readable file size."""
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class Prediction(models.Model):
    """Heart disease risk predictions with complete analysis."""
    
    class InputMethod(models.TextChoices):
        MANUAL = 'manual', 'Manual Input'
        OCR = 'ocr', 'OCR Document'
        HYBRID = 'hybrid', 'Hybrid (OCR + Manual)'
    
    class RiskCategory(models.TextChoices):
        LOW = 'LOW', 'Low Risk'
        MODERATE = 'MODERATE', 'Moderate Risk'
        HIGH = 'HIGH', 'High Risk'
    
    class ModelType(models.TextChoices):
        DETECTION = 'detection', 'Disease Detection'
        PREDICTION = 'prediction', '10-Year Risk Prediction'
        BOTH = 'both', 'Both Models'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    document = models.ForeignKey(
        MedicalDocument,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='predictions'
    )
    
    # Input method
    input_method = models.CharField(
        max_length=10,
        choices=InputMethod.choices,
        default=InputMethod.MANUAL
    )
    model_used = models.CharField(
        max_length=20,
        choices=ModelType.choices,
        default=ModelType.PREDICTION
    )
    
    # Complete input data (stored as JSON for flexibility)
    input_data = models.JSONField(default=dict)
    
    # ========== PATIENT DEMOGRAPHICS ==========
    age = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(18), MaxValueValidator(120)]
    )
    sex = models.IntegerField(
        null=True, blank=True,
        choices=[(0, 'Female'), (1, 'Male')]
    )
    
    # ========== VITAL SIGNS ==========
    systolic_bp = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(70), MaxValueValidator(250)]
    )
    diastolic_bp = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(40), MaxValueValidator(150)]
    )
    heart_rate = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(40), MaxValueValidator(200)]
    )
    
    # ========== LAB RESULTS ==========
    cholesterol = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(100), MaxValueValidator(400)]
    )
    hdl = models.FloatField(null=True, blank=True)
    ldl = models.FloatField(null=True, blank=True)
    triglycerides = models.FloatField(null=True, blank=True)
    glucose = models.FloatField(null=True, blank=True)
    
    # ========== LIFESTYLE ==========
    smoking = models.BooleanField(null=True, blank=True)
    diabetes = models.BooleanField(null=True, blank=True)
    bp_medication = models.BooleanField(null=True, blank=True)
    
    # ========== PHYSICAL MEASUREMENTS ==========
    bmi = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(15), MaxValueValidator(60)]
    )
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    
    # ========== STRESS TEST DATA (for Detection Model) ==========
    chest_pain_type = models.IntegerField(null=True, blank=True)  # 0-3
    max_heart_rate = models.IntegerField(null=True, blank=True)   # thalach
    exercise_angina = models.BooleanField(null=True, blank=True)  # exang
    st_depression = models.FloatField(null=True, blank=True)      # oldpeak
    st_slope = models.IntegerField(null=True, blank=True)         # slope
    major_vessels = models.IntegerField(null=True, blank=True)    # ca
    thalassemia = models.IntegerField(null=True, blank=True)      # thal
    resting_ecg = models.IntegerField(null=True, blank=True)      # restecg
    
    # ========== PREDICTION RESULTS ==========
    # 10-Year Risk Prediction
    risk_score = models.FloatField(
        null=True, blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    risk_percentage = models.FloatField(null=True, blank=True)
    risk_category = models.CharField(
        max_length=10,
        choices=RiskCategory.choices,
        null=True,
        blank=True
    )
    prediction_confidence = models.FloatField(null=True, blank=True)
    
    # Disease Detection
    detection_result = models.BooleanField(null=True, blank=True)
    detection_probability = models.FloatField(null=True, blank=True)
    detection_confidence = models.FloatField(null=True, blank=True)
    
    # Clinical Assessment
    clinical_score = models.IntegerField(null=True, blank=True)
    clinical_max_score = models.IntegerField(null=True, blank=True)
    clinical_risk_level = models.CharField(max_length=20, blank=True)
    
    # ========== CLINICAL OVERRIDE ==========
    clinical_override_applied = models.BooleanField(default=False)
    override_reason = models.TextField(blank=True)
    original_risk_category = models.CharField(max_length=10, blank=True)
    
    # ========== RECOMMENDATIONS ==========
    recommendations = models.TextField(blank=True)
    risk_factors = models.JSONField(default=list, blank=True)
    lifestyle_modifications = models.JSONField(default=list, blank=True)
    follow_up_actions = models.JSONField(default=list, blank=True)
    
    # ========== METADATA ==========
    model_version = models.CharField(max_length=20, default='v2.0')
    processing_time_ms = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Report generation
    report_generated = models.BooleanField(default=False)
    report_file = models.FileField(upload_to='reports/%Y/%m/', blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Risk Prediction'
        verbose_name_plural = 'Risk Predictions'
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['risk_category']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Prediction #{str(self.id)[:8]} - {self.risk_category or 'Pending'}"
    
    @property
    def risk_color(self):
        """Return color code for risk category."""
        colors = {
            'LOW': '#00D9C4',      # Teal
            'MODERATE': '#FFB800', # Yellow
            'HIGH': '#FF4757',     # Red
        }
        return colors.get(self.risk_category, '#808080')
    
    @property
    def risk_emoji(self):
        """Return emoji for risk category."""
        emojis = {
            'LOW': '🟢',
            'MODERATE': '🟡',
            'HIGH': '🔴',
        }
        return emojis.get(self.risk_category, '⚪')
    
    def get_summary(self):
        """Return a summary of the prediction."""
        return {
            'id': str(self.id),
            'risk_category': self.risk_category,
            'risk_percentage': self.risk_percentage,
            'detection_result': self.detection_result,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'model_used': self.model_used,
        }


class UnitPreference(models.Model):
    """User preferences for measurement units."""
    
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='unit_preferences'
    )
    
    # Unit choices
    cholesterol_unit = models.CharField(
        max_length=10,
        choices=[('mg/dL', 'mg/dL'), ('mmol/L', 'mmol/L')],
        default='mg/dL'
    )
    glucose_unit = models.CharField(
        max_length=10,
        choices=[('mg/dL', 'mg/dL'), ('mmol/L', 'mmol/L')],
        default='mg/dL'
    )
    weight_unit = models.CharField(
        max_length=5,
        choices=[('kg', 'Kilograms'), ('lbs', 'Pounds')],
        default='kg'
    )
    height_unit = models.CharField(
        max_length=5,
        choices=[('cm', 'Centimeters'), ('ft', 'Feet/Inches')],
        default='cm'
    )
    temperature_unit = models.CharField(
        max_length=5,
        choices=[('C', 'Celsius'), ('F', 'Fahrenheit')],
        default='C'
    )
    
    def __str__(self):
        return f"Units for {self.user.email}"


class DoctorPatient(models.Model):
    """Doctor-Patient relationship for role-based access."""
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending Approval'
        ACTIVE = 'active', 'Active'
        REVOKED = 'revoked', 'Access Revoked'
    
    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='patients'
    )
    patient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='doctors'
    )
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.PENDING
    )
    
    # Permissions
    can_view_history = models.BooleanField(default=True)
    can_add_notes = models.BooleanField(default=True)
    can_modify_predictions = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['doctor', 'patient']
        verbose_name = 'Doctor-Patient Relationship'
        verbose_name_plural = 'Doctor-Patient Relationships'
    
    def __str__(self):
        return f"Dr. {self.doctor.last_name} -> {self.patient.get_full_name()}"


class PredictionNote(models.Model):
    """Doctor notes on predictions."""
    
    prediction = models.ForeignKey(
        Prediction,
        on_delete=models.CASCADE,
        related_name='notes'
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='prediction_notes'
    )
    note = models.TextField()
    is_private = models.BooleanField(default=False)  # If true, only visible to doctor
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Note by {self.author.get_full_name()} on {self.prediction}"


class AuditLog(models.Model):
    """Audit log for tracking user actions."""
    
    class Action(models.TextChoices):
        CREATE = 'create', 'Created'
        READ = 'read', 'Viewed'
        UPDATE = 'update', 'Updated'
        DELETE = 'delete', 'Deleted'
        EXPORT = 'export', 'Exported'
        LOGIN = 'login', 'Logged In'
        LOGOUT = 'logout', 'Logged Out'
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='audit_logs'
    )
    action = models.CharField(max_length=10, choices=Action.choices)
    resource_type = models.CharField(max_length=50)  # e.g., 'Prediction', 'Document'
    resource_id = models.CharField(max_length=100, blank=True)
    details = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
            models.Index(fields=['resource_type', 'resource_id']),
        ]
    
    def __str__(self):
        return f"{self.user} - {self.action} - {self.resource_type}"


class SystemNotification(models.Model):
    """System notifications for users."""
    
    class NotificationType(models.TextChoices):
        INFO = 'info', 'Information'
        WARNING = 'warning', 'Warning'
        SUCCESS = 'success', 'Success'
        ERROR = 'error', 'Error'
        PREDICTION = 'prediction', 'New Prediction'
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='notifications'
    )
    title = models.CharField(max_length=200)
    message = models.TextField()
    notification_type = models.CharField(
        max_length=15,
        choices=NotificationType.choices,
        default=NotificationType.INFO
    )
    related_prediction = models.ForeignKey(
        Prediction,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'is_read', '-created_at']),
        ]
    
    def __str__(self):
        return f"{self.title} - {'Read' if self.is_read else 'Unread'}"
