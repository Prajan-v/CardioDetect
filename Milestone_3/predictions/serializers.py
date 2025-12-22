"""
Enhanced Serializers for CardioDetect predictions app.
"""

from rest_framework import serializers
from .models import (
    Prediction, MedicalDocument, UnitPreference,
    DoctorPatient, PredictionNote, SystemNotification
)


class MedicalDocumentSerializer(serializers.ModelSerializer):
    """Serializer for medical documents."""
    file_size_formatted = serializers.ReadOnlyField()
    
    class Meta:
        model = MedicalDocument
        fields = [
            'id', 'filename', 'file', 'file_type', 'file_size',
            'file_size_formatted', 'ocr_status', 'ocr_confidence',
            'ocr_method', 'extracted_fields', 'uploaded_at', 'processed_at'
        ]
        read_only_fields = ['id', 'ocr_status', 'ocr_confidence', 'extracted_fields']


class ManualInputSerializer(serializers.Serializer):
    """Serializer for manual input prediction."""
    # Required fields
    age = serializers.IntegerField(min_value=18, max_value=120)
    sex = serializers.IntegerField(min_value=0, max_value=1)
    systolic_bp = serializers.FloatField(min_value=70, max_value=250)
    
    # Common optional fields
    diastolic_bp = serializers.FloatField(min_value=40, max_value=150, required=False)
    cholesterol = serializers.FloatField(min_value=100, max_value=400, required=False)
    hdl = serializers.FloatField(min_value=20, max_value=150, required=False)
    glucose = serializers.FloatField(min_value=50, max_value=500, required=False)
    bmi = serializers.FloatField(min_value=15, max_value=60, required=False)
    heart_rate = serializers.IntegerField(min_value=40, max_value=200, required=False)
    
    # Lifestyle factors
    smoking = serializers.BooleanField(required=False, default=False)
    diabetes = serializers.BooleanField(required=False, default=False)
    bp_medication = serializers.BooleanField(required=False, default=False)
    
    # Stress test fields (for Detection model)
    chest_pain_type = serializers.IntegerField(min_value=0, max_value=3, required=False)
    max_heart_rate = serializers.IntegerField(min_value=60, max_value=220, required=False)
    exercise_angina = serializers.BooleanField(required=False)
    st_depression = serializers.FloatField(min_value=0, max_value=10, required=False)
    st_slope = serializers.IntegerField(min_value=0, max_value=2, required=False)
    major_vessels = serializers.IntegerField(min_value=0, max_value=3, required=False)
    thalassemia = serializers.IntegerField(min_value=1, max_value=3, required=False)
    resting_ecg = serializers.IntegerField(min_value=0, max_value=2, required=False)


class OCRUploadSerializer(serializers.Serializer):
    """Serializer for OCR file upload."""
    file = serializers.FileField()
    additional_data = serializers.JSONField(required=False, default=dict)


class PredictionSerializer(serializers.ModelSerializer):
    """Basic prediction serializer."""
    risk_color = serializers.ReadOnlyField()
    risk_emoji = serializers.ReadOnlyField()
    
    class Meta:
        model = Prediction
        fields = [
            'id', 'input_method', 'model_used',
            'risk_score', 'risk_percentage', 'risk_category',
            'risk_color', 'risk_emoji',
            'detection_result', 'detection_probability',
            'recommendations', 'risk_factors',
            'created_at'
        ]
        read_only_fields = fields


class PredictionHistorySerializer(serializers.ModelSerializer):
    """Serializer for prediction history list."""
    risk_color = serializers.ReadOnlyField()
    risk_emoji = serializers.ReadOnlyField()
    
    class Meta:
        model = Prediction
        fields = [
            'id', 'input_method', 'risk_category', 'risk_percentage',
            'risk_color', 'risk_emoji', 'detection_result',
            'created_at', 'processing_time_ms'
        ]


class PredictionDetailSerializer(serializers.ModelSerializer):
    """Detailed prediction serializer with all fields."""
    risk_color = serializers.ReadOnlyField()
    risk_emoji = serializers.ReadOnlyField()
    document = MedicalDocumentSerializer(read_only=True)
    notes = serializers.SerializerMethodField()
    
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['id', 'user', 'created_at', 'updated_at']
    
    def get_notes(self, obj):
        user = self.context.get('request').user if self.context.get('request') else None
        if user and user.is_doctor:
            notes = obj.notes.all()
        else:
            notes = obj.notes.filter(is_private=False)
        return PredictionNoteSerializer(notes, many=True).data


class PredictionNoteSerializer(serializers.ModelSerializer):
    """Serializer for prediction notes."""
    author_name = serializers.CharField(source='author.get_full_name', read_only=True)
    
    class Meta:
        model = PredictionNote
        fields = ['id', 'note', 'author_name', 'is_private', 'created_at', 'updated_at']
        read_only_fields = ['id', 'author_name', 'created_at', 'updated_at']


class UnitConversionSerializer(serializers.Serializer):
    """Serializer for unit conversion."""
    value = serializers.FloatField()
    from_unit = serializers.CharField(max_length=20)
    to_unit = serializers.CharField(max_length=20)


class UnitPreferenceSerializer(serializers.ModelSerializer):
    """Serializer for unit preferences."""
    class Meta:
        model = UnitPreference
        fields = ['cholesterol_unit', 'glucose_unit', 'weight_unit', 'height_unit', 'temperature_unit']


class StatisticsSerializer(serializers.Serializer):
    """Serializer for statistics response."""
    total_predictions = serializers.IntegerField()
    last_30_days = serializers.IntegerField()
    last_90_days = serializers.IntegerField()
    risk_distribution = serializers.DictField()
    input_methods = serializers.DictField()
    average_processing_time_ms = serializers.FloatField()
    latest_prediction = serializers.DictField(allow_null=True)
    trend = serializers.CharField()


class NotificationSerializer(serializers.ModelSerializer):
    """Serializer for notifications."""
    class Meta:
        model = SystemNotification
        fields = ['id', 'title', 'message', 'notification_type', 'is_read', 'created_at']
        read_only_fields = fields


class DoctorPatientSerializer(serializers.ModelSerializer):
    """Serializer for doctor-patient relationships."""
    doctor_name = serializers.CharField(source='doctor.get_full_name', read_only=True)
    patient_name = serializers.CharField(source='patient.get_full_name', read_only=True)
    
    class Meta:
        model = DoctorPatient
        fields = [
            'id', 'doctor', 'doctor_name', 'patient', 'patient_name',
            'status', 'can_view_history', 'can_add_notes', 'can_modify_predictions',
            'created_at'
        ]
        read_only_fields = ['id', 'doctor_name', 'patient_name', 'created_at']
