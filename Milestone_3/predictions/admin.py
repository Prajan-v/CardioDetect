"""
Admin configuration for CardioDetect predictions app.
Simplified for Django 6.0 compatibility.
"""

from django.contrib import admin
from django.urls import reverse
from django.db.models import Count
from .models import (
    MedicalDocument, Prediction, UnitPreference,
    DoctorPatient, PredictionNote, AuditLog, SystemNotification
)


@admin.register(MedicalDocument)
class MedicalDocumentAdmin(admin.ModelAdmin):
    """Admin for medical documents."""
    list_display = ['id_short', 'filename', 'get_user', 'file_type', 'ocr_status', 'uploaded_at']
    list_filter = ['ocr_status', 'file_type', 'uploaded_at']
    search_fields = ['filename', 'user__email', 'user__first_name', 'user__last_name']
    readonly_fields = ['id', 'extracted_text', 'extracted_fields', 'processed_at', 'processing_time_ms']
    ordering = ['-uploaded_at']
    
    @admin.display(description='ID')
    def id_short(self, obj):
        return str(obj.id)[:8]
    
    @admin.display(description='User')
    def get_user(self, obj):
        if obj.user:
            return obj.user.email
        return '-'


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    """Admin for predictions."""
    list_display = ['id_short', 'get_user', 'risk_category', 'risk_percentage', 'input_method', 'created_at']
    list_filter = ['risk_category', 'input_method', 'model_used', 'clinical_override_applied', 'created_at']
    search_fields = ['user__email', 'user__first_name', 'user__last_name']
    readonly_fields = ['id', 'created_at', 'updated_at', 'processing_time_ms']
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('id', 'user', 'document', 'input_method', 'model_used')
        }),
        ('Patient Data', {
            'fields': ('age', 'sex', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'hdl', 'glucose', 'bmi', 'heart_rate'),
            'classes': ('collapse',)
        }),
        ('Lifestyle', {
            'fields': ('smoking', 'diabetes', 'bp_medication'),
            'classes': ('collapse',)
        }),
        ('Stress Test Data', {
            'fields': ('chest_pain_type', 'max_heart_rate', 'exercise_angina', 'st_depression', 'st_slope', 'major_vessels', 'thalassemia', 'resting_ecg'),
            'classes': ('collapse',)
        }),
        ('Results', {
            'fields': ('risk_category', 'risk_score', 'risk_percentage', 'prediction_confidence', 'detection_result', 'detection_probability', 'detection_confidence')
        }),
        ('Clinical', {
            'fields': ('clinical_score', 'clinical_max_score', 'clinical_override_applied', 'override_reason', 'original_risk_category'),
            'classes': ('collapse',)
        }),
        ('Recommendations', {
            'fields': ('recommendations', 'risk_factors', 'lifestyle_modifications', 'follow_up_actions'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('model_version', 'processing_time_ms', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    @admin.display(description='ID')
    def id_short(self, obj):
        return str(obj.id)[:8]
    
    @admin.display(description='User')
    def get_user(self, obj):
        if obj.user:
            return obj.user.email
        return '-'
    
    # Admin Actions
    actions = ['export_to_csv', 'mark_high_risk', 'mark_low_risk']
    
    @admin.action(description='Export selected to CSV')
    def export_to_csv(self, request, queryset):
        import csv
        from django.http import HttpResponse
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        writer = csv.writer(response)
        writer.writerow(['ID', 'User', 'Risk', 'Percentage', 'Input Method', 'Created'])
        for pred in queryset:
            writer.writerow([
                str(pred.id)[:8],
                pred.user.email if pred.user else 'N/A',
                pred.risk_category,
                pred.risk_percentage,
                pred.input_method,
                pred.created_at.strftime('%Y-%m-%d %H:%M')
            ])
        return response
    
    @admin.action(description='Mark as HIGH risk')
    def mark_high_risk(self, request, queryset):
        count = queryset.update(risk_category='HIGH')
        self.message_user(request, f'{count} prediction(s) marked as HIGH risk.')
    
    @admin.action(description='Mark as LOW risk')
    def mark_low_risk(self, request, queryset):
        count = queryset.update(risk_category='LOW')
        self.message_user(request, f'{count} prediction(s) marked as LOW risk.')


@admin.register(UnitPreference)
class UnitPreferenceAdmin(admin.ModelAdmin):
    """Admin for unit preferences."""
    list_display = ['get_user', 'cholesterol_unit', 'glucose_unit', 'weight_unit', 'height_unit']
    search_fields = ['user__email']
    
    @admin.display(description='User')
    def get_user(self, obj):
        if obj.user:
            return obj.user.email
        return '-'


@admin.register(DoctorPatient)
class DoctorPatientAdmin(admin.ModelAdmin):
    """Admin for doctor-patient relationships."""
    list_display = ['get_doctor', 'get_patient', 'status', 'can_view_history', 'can_add_notes', 'created_at']
    list_filter = ['status', 'can_view_history', 'can_add_notes']
    search_fields = ['doctor__email', 'patient__email', 'doctor__last_name', 'patient__last_name']
    
    @admin.display(description='Doctor')
    def get_doctor(self, obj):
        if obj.doctor:
            return obj.doctor.email
        return '-'
    
    @admin.display(description='Patient')
    def get_patient(self, obj):
        if obj.patient:
            return obj.patient.email
        return '-'


@admin.register(PredictionNote)
class PredictionNoteAdmin(admin.ModelAdmin):
    """Admin for prediction notes."""
    list_display = ['prediction_short', 'get_author', 'is_private', 'created_at']
    list_filter = ['is_private', 'created_at']
    search_fields = ['note', 'author__email']
    
    @admin.display(description='Prediction')
    def prediction_short(self, obj):
        return str(obj.prediction.id)[:8]
    
    @admin.display(description='Author')
    def get_author(self, obj):
        if obj.author:
            return obj.author.email
        return '-'


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    """Admin for audit logs."""
    list_display = ['timestamp', 'get_user', 'action', 'resource_type', 'resource_id_short', 'ip_address']
    list_filter = ['action', 'resource_type', 'timestamp']
    search_fields = ['user__email', 'resource_id', 'ip_address']
    readonly_fields = ['user', 'action', 'resource_type', 'resource_id', 'details', 'ip_address', 'user_agent', 'timestamp']
    ordering = ['-timestamp']
    
    @admin.display(description='User')
    def get_user(self, obj):
        if obj.user:
            return obj.user.email
        return '-'
    
    @admin.display(description='Resource ID')
    def resource_id_short(self, obj):
        return obj.resource_id[:8] if obj.resource_id else '-'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(SystemNotification)
class SystemNotificationAdmin(admin.ModelAdmin):
    """Admin for system notifications."""
    list_display = ['title', 'get_user', 'notification_type', 'is_read', 'created_at']
    list_filter = ['notification_type', 'is_read', 'created_at']
    search_fields = ['title', 'message', 'user__email']
    ordering = ['-created_at']
    
    @admin.display(description='User')
    def get_user(self, obj):
        if obj.user:
            return obj.user.email
        return '-'


# Admin site customization
admin.site.site_header = 'CardioDetect Administration'
admin.site.site_title = 'CardioDetect Admin'
admin.site.index_title = 'Dashboard'
admin.site.site_url = 'http://localhost:3000'  # Point to React frontend
