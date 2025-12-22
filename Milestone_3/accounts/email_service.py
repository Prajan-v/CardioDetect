"""
Email Service for CardioDetect.
Centralized email sending with templates.
"""

from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.utils import timezone
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def send_templated_email(
    subject: str,
    template_name: str,
    context: dict,
    recipient_email: str,
    fail_silently: bool = True
) -> bool:
    """
    Send an email using a template.
    
    Args:
        subject: Email subject line
        template_name: Name of template in templates/emails/ (without .html)
        context: Context dict for template rendering
        recipient_email: Recipient's email address
        fail_silently: If True, don't raise exceptions on failure
    
    Returns:
        True if email was sent successfully, False otherwise
    """
    try:
        # Add common context
        context['user_email'] = recipient_email
        context['frontend_url'] = settings.FRONTEND_URL
        
        # Render HTML content
        html_content = render_to_string(f'emails/{template_name}.html', context)
        
        # Send email
        send_mail(
            subject=subject,
            message='',  # Plain text fallback (empty, using HTML)
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient_email],
            html_message=html_content,
            fail_silently=fail_silently,
        )
        
        logger.info(f"Email sent: {template_name} to {recipient_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email {template_name} to {recipient_email}: {e}")
        if not fail_silently:
            raise
        return False


# ========== ACCOUNT EMAILS ==========

def send_welcome_email(user) -> bool:
    """Send welcome email after account creation."""
    return send_templated_email(
        subject="Welcome to CardioDetect! 🫀",
        template_name="welcome",
        context={
            'first_name': user.first_name,
            'login_url': f"{settings.FRONTEND_URL}/login",
        },
        recipient_email=user.email,
    )


def send_password_changed_email(user, ip_address: str = None) -> bool:
    """Send email after password change."""
    return send_templated_email(
        subject="🔐 Your Password Was Changed - CardioDetect",
        template_name="password_changed",
        context={
            'first_name': user.first_name,
            'changed_at': timezone.now().strftime('%B %d, %Y at %I:%M %p'),
            'ip_address': ip_address or 'Unknown',
            'reset_url': f"{settings.FRONTEND_URL}/forgot-password",
        },
        recipient_email=user.email,
    )


# ========== APPROVAL EMAILS ==========

def send_change_approved_email(pending_change) -> bool:
    """Send email when profile change is approved."""
    user = pending_change.user
    return send_templated_email(
        subject="✅ Your Profile Change Was Approved - CardioDetect",
        template_name="change_approved",
        context={
            'first_name': user.first_name,
            'field_name': pending_change.field_name.replace('_', ' ').title(),
            'old_value': pending_change.old_value or '(empty)',
            'new_value': pending_change.new_value,
            'approved_by': pending_change.reviewed_by.get_full_name() if pending_change.reviewed_by else 'Admin',
            'review_notes': pending_change.review_notes,
            'approved_at': pending_change.reviewed_at.strftime('%B %d, %Y at %I:%M %p') if pending_change.reviewed_at else '',
            'dashboard_url': f"{settings.FRONTEND_URL}/profile",
        },
        recipient_email=user.email,
    )


def send_change_rejected_email(pending_change) -> bool:
    """Send email when profile change is rejected."""
    user = pending_change.user
    return send_templated_email(
        subject="Profile Change Request Update - CardioDetect",
        template_name="change_rejected",
        context={
            'first_name': user.first_name,
            'field_name': pending_change.field_name.replace('_', ' ').title(),
            'old_value': pending_change.old_value or '(empty)',
            'new_value': pending_change.new_value,
            'reviewed_by': pending_change.reviewed_by.get_full_name() if pending_change.reviewed_by else 'Admin',
            'review_notes': pending_change.review_notes,
            'reviewed_at': pending_change.reviewed_at.strftime('%B %d, %Y at %I:%M %p') if pending_change.reviewed_at else '',
            'profile_url': f"{settings.FRONTEND_URL}/profile",
        },
        recipient_email=user.email,
    )


# ========== RISK ALERT EMAILS ==========

def send_high_risk_alert_to_patient(prediction) -> bool:
    """Send high-risk alert email to patient."""
    user = prediction.user
    return send_templated_email(
        subject="⚠️ High Risk Alert - CardioDetect Assessment",
        template_name="high_risk_alert",
        context={
            'is_doctor': False,
            'first_name': user.first_name,
            'risk_category': prediction.risk_category,
            'risk_percentage': round(prediction.risk_percentage, 1) if prediction.risk_percentage else 'N/A',
            'assessment_date': prediction.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'input_method': prediction.input_method.title(),
            'dashboard_url': f"{settings.FRONTEND_URL}/dashboard",
        },
        recipient_email=user.email,
    )


def send_high_risk_alert_to_doctor(prediction, doctor) -> bool:
    """Send high-risk alert email to assigned doctor."""
    patient = prediction.user
    return send_templated_email(
        subject=f"⚠️ High Risk Alert for Patient {patient.get_full_name()} - CardioDetect",
        template_name="high_risk_alert",
        context={
            'is_doctor': True,
            'doctor_name': doctor.get_full_name(),
            'patient_name': patient.get_full_name(),
            'patient_email': patient.email,
            'risk_category': prediction.risk_category,
            'risk_percentage': round(prediction.risk_percentage, 1) if prediction.risk_percentage else 'N/A',
            'assessment_date': prediction.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'input_method': prediction.input_method.title(),
            'doctor_dashboard_url': f"{settings.FRONTEND_URL}/doctor/dashboard",
        },
        recipient_email=doctor.email,
    )


def send_high_risk_alerts(prediction) -> dict:
    """
    Send high-risk alerts to patient and all assigned doctors.
    Only sends if risk_category is 'HIGH' or risk_percentage > 60.
    
    Returns dict with results.
    """
    results = {'patient_notified': False, 'doctors_notified': []}
    
    # Check if this is a high-risk result
    is_high_risk = (
        prediction.risk_category and prediction.risk_category.upper() == 'HIGH'
    ) or (
        prediction.risk_percentage and prediction.risk_percentage > 60
    )
    
    if not is_high_risk:
        return results
    
    # Notify patient
    results['patient_notified'] = send_high_risk_alert_to_patient(prediction)
    
    # Notify all assigned doctors
    from .models import DoctorPatient
    doctor_relations = DoctorPatient.objects.filter(patient=prediction.user)
    
    for relation in doctor_relations:
        if send_high_risk_alert_to_doctor(prediction, relation.doctor):
            results['doctors_notified'].append(relation.doctor.email)
    
    return results


# ========== SECURITY & LOGIN EMAILS ==========

def send_new_login_alert(user, ip_address: str = None, location: str = None, device: str = None) -> bool:
    """Send email when user logs in from a new device/location."""
    return send_templated_email(
        subject="🔔 New Login to Your CardioDetect Account",
        template_name="new_login_alert",
        context={
            'first_name': user.first_name,
            'login_time': timezone.now().strftime('%B %d, %Y at %I:%M %p'),
            'ip_address': ip_address or 'Unknown',
            'location': location or 'Unknown location',
            'device': device or 'Unknown device',
            'security_url': f"{settings.FRONTEND_URL}/settings",
        },
        recipient_email=user.email,
    )


def send_account_locked_email(user, reason: str = None) -> bool:
    """Send email when account is locked due to security reasons."""
    # Convert to IST (UTC+5:30)
    ist = timezone.now() + timedelta(hours=5, minutes=30)
    return send_templated_email(
        subject="🔒 Your CardioDetect Account Has Been Locked",
        template_name="account_locked",
        context={
            'first_name': user.first_name,
            'locked_at': ist.strftime('%B %d, %Y at %I:%M %p') + ' IST',
            'reason': reason or 'Too many failed login attempts',
            'unlock_url': f"{settings.FRONTEND_URL}/forgot-password",
            'support_email': 'support@cardiodetect.com',
        },
        recipient_email=user.email,
    )


def send_account_unlocked_email(user) -> bool:
    """Send email when account is unlocked by admin."""
    return send_templated_email(
        subject="✅ Your CardioDetect Account Has Been Unlocked",
        template_name="account_unlocked",
        context={
            'first_name': user.first_name,
            'unlocked_at': timezone.now().strftime('%B %d, %Y at %I:%M %p'),
            'login_url': f"{settings.FRONTEND_URL}/login",
        },
        recipient_email=user.email,
    )


# ========== PROFILE CHANGE EMAILS ==========

def send_change_submitted_email(pending_change) -> bool:
    """Send confirmation email when user submits a profile change for approval."""
    user = pending_change.user
    return send_templated_email(
        subject="📝 Profile Change Request Submitted - CardioDetect",
        template_name="change_submitted",
        context={
            'first_name': user.first_name,
            'field_name': pending_change.field_name.replace('_', ' ').title(),
            'old_value': pending_change.old_value or '(empty)',
            'new_value': pending_change.new_value,
            'submitted_at': pending_change.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'profile_url': f"{settings.FRONTEND_URL}/profile",
        },
        recipient_email=user.email,
    )


# ========== DOCTOR-PATIENT RELATIONSHIP EMAILS ==========

def send_doctor_assigned_email(patient, doctor) -> bool:
    """Send email to patient when a doctor is assigned to them."""
    return send_templated_email(
        subject="👨‍⚕️ A Doctor Has Been Assigned to Your Account - CardioDetect",
        template_name="doctor_assigned",
        context={
            'patient_name': patient.first_name,
            'doctor_name': doctor.get_full_name(),
            'doctor_specialization': doctor.specialization or 'Cardiology',
            'doctor_hospital': doctor.hospital or 'CardioDetect Partner Hospital',
            'dashboard_url': f"{settings.FRONTEND_URL}/dashboard",
        },
        recipient_email=patient.email,
    )


def send_patient_assigned_email(doctor, patient) -> bool:
    """Send email to doctor when a new patient is assigned to them."""
    return send_templated_email(
        subject=f"👤 New Patient Assigned: {patient.get_full_name()} - CardioDetect",
        template_name="patient_assigned",
        context={
            'doctor_name': doctor.first_name,
            'patient_name': patient.get_full_name(),
            'patient_email': patient.email,
            'doctor_dashboard_url': f"{settings.FRONTEND_URL}/doctor/dashboard",
        },
        recipient_email=doctor.email,
    )


# ========== PREDICTION & HEALTH EMAILS ==========

def send_prediction_complete_email(prediction) -> bool:
    """Send email when a prediction is completed with results summary."""
    user = prediction.user
    risk_emoji = "🟢" if prediction.risk_category == "LOW" else ("🟡" if prediction.risk_category == "MODERATE" else "🔴")
    
    return send_templated_email(
        subject=f"{risk_emoji} Your CardioDetect Assessment Results Are Ready",
        template_name="prediction_complete",
        context={
            'first_name': user.first_name,
            'risk_category': prediction.risk_category,
            'risk_percentage': round(prediction.risk_percentage, 1) if prediction.risk_percentage else 'N/A',
            'detection_result': 'Positive' if prediction.detection_result else 'Negative',
            'assessment_date': prediction.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'input_method': prediction.input_method.title(),
            'dashboard_url': f"{settings.FRONTEND_URL}/dashboard",
            'is_high_risk': prediction.risk_category == 'HIGH',
        },
        recipient_email=user.email,
    )


def send_weekly_health_summary(user, predictions_count: int, latest_risk: str = None) -> bool:
    """Send weekly health activity summary email."""
    return send_templated_email(
        subject="📊 Your Weekly CardioDetect Health Summary",
        template_name="weekly_summary",
        context={
            'first_name': user.first_name,
            'predictions_count': predictions_count,
            'latest_risk': latest_risk or 'No assessments this week',
            'week_start': (timezone.now() - timezone.timedelta(days=7)).strftime('%B %d'),
            'week_end': timezone.now().strftime('%B %d, %Y'),
            'dashboard_url': f"{settings.FRONTEND_URL}/dashboard",
        },
        recipient_email=user.email,
    )


# ========== ADMIN NOTIFICATION EMAILS ==========

def send_admin_new_user_notification(admin_email: str, new_user) -> bool:
    """Notify admin when a new user registers."""
    return send_templated_email(
        subject=f"👤 New User Registration: {new_user.get_full_name()} - CardioDetect",
        template_name="admin_new_user",
        context={
            'user_name': new_user.get_full_name(),
            'user_email': new_user.email,
            'user_role': new_user.role.title(),
            'registered_at': new_user.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'admin_url': f"{settings.FRONTEND_URL}/admin",
        },
        recipient_email=admin_email,
    )


def send_admin_change_pending_notification(admin_email: str, pending_change) -> bool:
    """Notify admin when a new profile change is pending approval."""
    return send_templated_email(
        subject=f"📋 New Change Request from {pending_change.user.get_full_name()} - CardioDetect",
        template_name="admin_change_pending",
        context={
            'user_name': pending_change.user.get_full_name(),
            'user_email': pending_change.user.email,
            'field_name': pending_change.field_name.replace('_', ' ').title(),
            'old_value': pending_change.old_value or '(empty)',
            'new_value': pending_change.new_value,
            'reason': pending_change.reason or 'No reason provided',
            'submitted_at': pending_change.created_at.strftime('%B %d, %Y at %I:%M %p'),
            'approval_url': f"{settings.FRONTEND_URL}/admin/approvals",
        },
        recipient_email=admin_email,
    )

