"""
HIPAA/GDPR Compliance Utilities for CardioDetect.

Implements:
- Data Export (GDPR Article 15 - Right of Access)
- Data Deletion with 7-day Grace Period (GDPR Article 17 - Right to Erasure)
- Data Anonymization
- Consent Tracking
- Compliance Audit Trail

Note: Models (DataDeletionRequest, ConsentRecord) are defined in accounts.models
"""

import hashlib
import json
from datetime import timedelta
from typing import Dict, Any, List

from django.db import transaction
from django.utils import timezone

# Import models from the central models file
from .models import DataDeletionRequest, ConsentRecord, User


def export_user_data(user) -> Dict[str, Any]:
    """
    Export all user data in portable JSON format.
    GDPR Article 15 - Right of Access.
    
    Includes:
    - Profile information
    - Prediction history with recommendations
    - Uploaded documents metadata
    - Consent history
    - Login history
    """
    from predictions.models import Prediction, MedicalDocument
    
    # Basic profile
    profile_data = {
        'id': str(user.id),
        'email': user.email,
        'username': user.username,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'role': user.role,
        'phone': user.phone,
        'date_of_birth': str(user.date_of_birth) if user.date_of_birth else None,
        'gender': user.gender,
        'address': user.address,
        'city': user.city,
        'country': user.country,
        'bio': user.bio,
        'email_verified': user.email_verified,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'last_login': user.last_login.isoformat() if user.last_login else None,
    }
    
    # Doctor-specific fields
    if user.role == 'doctor':
        profile_data['license_number'] = user.license_number
        profile_data['specialization'] = user.specialization
        profile_data['hospital'] = user.hospital
    
    # Predictions with recommendations
    predictions = Prediction.objects.filter(user=user).order_by('-created_at')
    predictions_data = []
    for pred in predictions:
        predictions_data.append({
            'id': str(pred.id),
            'created_at': pred.created_at.isoformat(),
            'input_method': pred.input_method,
            'model_used': pred.model_used,
            'age': pred.age,
            'sex': pred.sex,
            'systolic_bp': pred.systolic_bp,
            'diastolic_bp': pred.diastolic_bp,
            'heart_rate': pred.heart_rate,
            'cholesterol': pred.cholesterol,
            'hdl': pred.hdl,
            'ldl': pred.ldl,
            'glucose': pred.glucose,
            'bmi': pred.bmi,
            'smoking': pred.smoking,
            'diabetes': pred.diabetes,
            'risk_category': pred.risk_category,
            'risk_percentage': pred.risk_percentage,
            'recommendations': pred.recommendations,
            'risk_factors': pred.risk_factors,
        })
    
    # Documents metadata (not file content for security)
    documents = MedicalDocument.objects.filter(user=user).order_by('-uploaded_at')
    documents_data = []
    for doc in documents:
        documents_data.append({
            'id': str(doc.id),
            'filename': doc.filename,
            'file_type': doc.file_type,
            'file_size': doc.file_size,
            'uploaded_at': doc.uploaded_at.isoformat(),
            'ocr_status': doc.ocr_status,
            'ocr_confidence': doc.ocr_confidence,
        })
    
    # Consent history
    consent_records = ConsentRecord.objects.filter(user=user).order_by('-recorded_at')
    consent_data = []
    for record in consent_records:
        consent_data.append({
            'type': record.consent_type,
            'granted': record.granted,
            'version': record.version,
            'recorded_at': record.recorded_at.isoformat(),
        })
    
    # Login history
    login_history = user.login_history.all().order_by('-timestamp')[:50]
    login_data = []
    for login in login_history:
        login_data.append({
            'timestamp': login.timestamp.isoformat(),
            'ip_address': login.ip_address,
            'location': login.location,
            'success': login.success,
        })
    
    return {
        'export_date': timezone.now().isoformat(),
        'data_controller': 'CardioDetect',
        'data_subject': {
            'email': user.email,
            'id': str(user.id),
        },
        'profile': profile_data,
        'predictions': predictions_data,
        'predictions_count': len(predictions_data),
        'documents': documents_data,
        'documents_count': len(documents_data),
        'consent_history': consent_data,
        'login_history': login_data,
    }


def anonymize_user_data(user) -> Dict[str, Any]:
    """
    Anonymize user data by replacing PII with hashed values.
    Used for analytics retention after deletion.
    """
    anon_id = hashlib.sha256(str(user.id).encode()).hexdigest()[:16]
    
    return {
        'anonymized_id': anon_id,
        'role': user.role,
        'created_year': user.created_at.year if user.created_at else None,
        'predictions_count': user.predictions.count(),
        'country': user.country[:2] if user.country else None,
    }


@transaction.atomic
def execute_data_deletion(deletion_request: DataDeletionRequest) -> Dict[str, Any]:
    """
    Execute permanent data deletion after grace period.
    """
    from predictions.models import Prediction, MedicalDocument, AuditLog
    
    user = deletion_request.user
    result = {
        'user_id': str(user.id),
        'email_hash': hashlib.sha256(user.email.encode()).hexdigest(),
        'deleted_at': timezone.now().isoformat(),
        'items_deleted': {},
    }
    
    try:
        anon_data = anonymize_user_data(user)
        result['anonymized_stats'] = anon_data
        
        # Delete predictions
        pred_count = Prediction.objects.filter(user=user).count()
        Prediction.objects.filter(user=user).delete()
        result['items_deleted']['predictions'] = pred_count
        
        # Delete documents
        docs = MedicalDocument.objects.filter(user=user)
        doc_count = docs.count()
        for doc in docs:
            if doc.file:
                try:
                    doc.file.delete(save=False)
                except Exception:
                    pass
        docs.delete()
        result['items_deleted']['documents'] = doc_count
        
        # Delete consent records
        consent_count = ConsentRecord.objects.filter(user=user).count()
        ConsentRecord.objects.filter(user=user).delete()
        result['items_deleted']['consent_records'] = consent_count
        
        # Delete login history
        login_count = user.login_history.count()
        user.login_history.all().delete()
        result['items_deleted']['login_history'] = login_count
        
        # Anonymize user record
        user.email = f"deleted_{anon_data['anonymized_id']}@deleted.local"
        user.username = f"deleted_{anon_data['anonymized_id']}"
        user.first_name = "Deleted"
        user.last_name = "User"
        user.phone = ""
        user.date_of_birth = None
        user.address = ""
        user.city = ""
        user.bio = ""
        user.is_active = False
        user.save()
        
        # Mark request as completed
        deletion_request.status = DataDeletionRequest.Status.COMPLETED
        deletion_request.completed_at = timezone.now()
        deletion_request.save()
        
        # Create audit log
        AuditLog.objects.create(
            user=None,
            action='delete',
            resource_type='User',
            resource_id=str(result['user_id']),
            details={'deletion_result': result},
        )
        
        result['success'] = True
        
    except Exception as e:
        deletion_request.status = DataDeletionRequest.Status.FAILED
        deletion_request.save()
        result['success'] = False
        result['error'] = str(e)
    
    return result


def get_consent_history(user) -> List[Dict[str, Any]]:
    """Get full consent history for a user."""
    records = ConsentRecord.objects.filter(user=user).order_by('-recorded_at')
    return [
        {
            'type': r.consent_type,
            'type_display': r.get_consent_type_display(),
            'granted': r.granted,
            'version': r.version,
            'recorded_at': r.recorded_at.isoformat(),
        }
        for r in records
    ]


def record_consent(
    user,
    consent_type: str,
    granted: bool,
    version: str,
    ip_address: str = None,
    user_agent: str = ''
) -> ConsentRecord:
    """Record a consent action (grant or revoke)."""
    return ConsentRecord.objects.create(
        user=user,
        consent_type=consent_type,
        granted=granted,
        version=version,
        ip_address=ip_address,
        user_agent=user_agent[:500] if user_agent else '',
    )
