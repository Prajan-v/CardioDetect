"""
API URL Configuration for CardioDetect predictions app.
Complete routing for all endpoints.
"""

from django.urls import path
from .views import (
    ManualPredictionView,
    OCRPredictionView,
    PredictionHistoryView,
    PredictionDetailView,
    PredictionStatisticsView,
    ExportPredictionView,
    DocumentListView,
    DocumentDetailView,
    UnitConversionView,
    NotificationListView,
    NotificationMarkReadView,
    DashboardView,
    HealthCheckView,
    AdminStatsView,
)
from .barcode_views import (
    BarcodeVerifyView,
    DeviceAuthorizationView,
    DeviceCheckView,
    DeviceListView,
    DeviceRevokeView,
    ScanHistoryView,
    ScanStatsView,
)
from .role_views import (
    DoctorDashboardView,
    DoctorPatientsView,
    DoctorPatientDetailView,
    PatientDashboardView,
)
from .export_views import (
    ExportPredictionsExcelView,
    ExportDoctorPatientsExcelView,
)
from .pdf_generator import GeneratePredictionPDFView

app_name = 'predictions'

urlpatterns = [
    # ========== PREDICTION ENDPOINTS ==========
    # Manual input prediction
    path('predict/manual/', ManualPredictionView.as_view(), name='predict-manual'),
    
    # OCR document prediction
    path('predict/ocr/', OCRPredictionView.as_view(), name='predict-ocr'),
    
    # Prediction history list (with filtering and pagination)
    path('history/', PredictionHistoryView.as_view(), name='prediction-history'),
    
    # Export history to Excel
    path('history/export/excel/', ExportPredictionsExcelView.as_view(), name='export-predictions-excel'),
    
    # Prediction detail, update, delete
    path('predictions/<uuid:id>/', PredictionDetailView.as_view(), name='prediction-detail'),
    
    # Export prediction as PDF (legacy)
    path('predictions/<uuid:id>/export/', ExportPredictionView.as_view(), name='prediction-export'),
    
    # Generate PDF report
    path('predictions/<uuid:id>/pdf/', GeneratePredictionPDFView.as_view(), name='prediction-pdf'),
    
    # User statistics and analytics
    path('statistics/', PredictionStatisticsView.as_view(), name='statistics'),
    
    # ========== DOCUMENT ENDPOINTS ==========
    # Document list
    path('documents/', DocumentListView.as_view(), name='document-list'),
    
    # Document detail
    path('documents/<uuid:id>/', DocumentDetailView.as_view(), name='document-detail'),
    
    # ========== UTILITY ENDPOINTS ==========
    # Unit conversion
    path('convert/', UnitConversionView.as_view(), name='unit-convert'),
    
    # ========== NOTIFICATION ENDPOINTS ==========
    # Notification list
    path('notifications/', NotificationListView.as_view(), name='notification-list'),
    
    # Mark notifications as read
    path('notifications/read/', NotificationMarkReadView.as_view(), name='notification-read'),
    
    # ========== DASHBOARD ENDPOINTS ==========
    # Dashboard data (generic)
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    
    # Admin stats (for doctor/admin dashboard)
    path('admin/stats/', AdminStatsView.as_view(), name='admin-stats'),
    
    # ========== ROLE-BASED DASHBOARDS ==========
    # Doctor dashboard
    path('doctor/dashboard/', DoctorDashboardView.as_view(), name='doctor-dashboard'),
    
    # Doctor patient list and add
    path('doctor/patients/', DoctorPatientsView.as_view(), name='doctor-patients'),
    
    # Export doctor's patients to Excel
    path('doctor/patients/export/excel/', ExportDoctorPatientsExcelView.as_view(), name='export-doctor-patients-excel'),
    
    # Doctor patient detail
    path('doctor/patients/<uuid:patient_id>/', DoctorPatientDetailView.as_view(), name='doctor-patient-detail'),
    
    # Patient dashboard
    path('patient/dashboard/', PatientDashboardView.as_view(), name='patient-dashboard'),
    
    # ========== HEALTH CHECK ==========
    # Public health check endpoint
    path('health/', HealthCheckView.as_view(), name='health-check'),
    
    # ========== BARCODE VERIFICATION API ==========
    path('barcode/verify/', BarcodeVerifyView.as_view(), name='barcode-verify'),
    path('barcode/device/auth/', DeviceAuthorizationView.as_view(), name='device-auth'),
    path('barcode/device/check/', DeviceCheckView.as_view(), name='device-check'),
    path('barcode/devices/', DeviceListView.as_view(), name='device-list'),
    path('barcode/device/revoke/', DeviceRevokeView.as_view(), name='device-revoke'),
    path('barcode/scans/', ScanHistoryView.as_view(), name='scan-history'),
    path('barcode/stats/', ScanStatsView.as_view(), name='scan-stats'),
]
