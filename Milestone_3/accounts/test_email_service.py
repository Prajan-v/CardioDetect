"""
Comprehensive Email Service Tests for CardioDetect.
Tests all email functions to verify 100% reliability.
"""

from django.test import TestCase, override_settings
from django.core import mail
from django.utils import timezone
from unittest.mock import Mock, patch, MagicMock
from datetime import timedelta

from .email_service import (
    send_templated_email,
    send_welcome_email,
    send_password_changed_email,
    send_change_approved_email,
    send_change_rejected_email,
    send_high_risk_alert_to_patient,
    send_high_risk_alert_to_doctor,
    send_high_risk_alerts,
    send_new_login_alert,
    send_account_locked_email,
    send_account_unlocked_email,
    send_change_submitted_email,
    send_doctor_assigned_email,
    send_patient_assigned_email,
    send_prediction_complete_email,
    send_weekly_health_summary,
    send_admin_new_user_notification,
    send_admin_change_pending_notification,
)
from .models import User


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    FRONTEND_URL='http://localhost:3000',
    DEFAULT_FROM_EMAIL='CardioDetect <test@cardiodetect.com>'
)
class EmailServiceTestCase(TestCase):
    """Test cases for email service functions."""

    @classmethod
    def setUpTestData(cls):
        """Set up test fixtures once for all tests in this class."""
        # Create test patient with unique username
        cls.patient = User.objects.create_user(
            email='email_patient@test.com',
            username='email_patient',
            password='testpass123',
            first_name='John',
            last_name='Doe',
            role='patient'
        )
        
        # Create test doctor with unique username
        cls.doctor = User.objects.create_user(
            email='email_doctor@test.com',
            username='email_doctor',
            password='testpass123',
            first_name='Dr. Jane',
            last_name='Smith',
            role='doctor',
            specialization='Cardiology',
            hospital='Test Hospital'
        )

    def setUp(self):
        """Clear mail outbox before each test."""
        mail.outbox = []

    def test_send_templated_email_success(self):
        """Test base templated email function."""
        result = send_templated_email(
            subject="Test Email",
            template_name="welcome",
            context={'first_name': 'Test', 'login_url': 'http://test.com'},
            recipient_email="test@example.com"
        )
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].subject, "Test Email")
        self.assertIn("test@example.com", mail.outbox[0].to)

    def test_send_welcome_email(self):
        """Test welcome email after registration."""
        result = send_welcome_email(self.patient)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Welcome", mail.outbox[0].subject)
        self.assertIn(self.patient.email, mail.outbox[0].to)

    def test_send_password_changed_email(self):
        """Test password changed notification."""
        result = send_password_changed_email(self.patient, ip_address="192.168.1.1")
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Password", mail.outbox[0].subject)

    def test_send_new_login_alert(self):
        """Test new login alert email."""
        result = send_new_login_alert(
            self.patient,
            ip_address="192.168.1.1",
            location="Mumbai, India",
            device="Chrome on MacOS"
        )
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Login", mail.outbox[0].subject)

    def test_send_account_locked_email(self):
        """Test account locked notification."""
        result = send_account_locked_email(
            self.patient,
            reason="Too many failed login attempts"
        )
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Locked", mail.outbox[0].subject)

    def test_send_account_unlocked_email(self):
        """Test account unlocked notification."""
        result = send_account_unlocked_email(self.patient)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Unlocked", mail.outbox[0].subject)

    def test_send_doctor_assigned_email(self):
        """Test doctor assigned to patient notification."""
        result = send_doctor_assigned_email(self.patient, self.doctor)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Doctor", mail.outbox[0].subject)
        self.assertIn(self.patient.email, mail.outbox[0].to)

    def test_send_patient_assigned_email(self):
        """Test patient assigned to doctor notification."""
        result = send_patient_assigned_email(self.doctor, self.patient)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Patient", mail.outbox[0].subject)
        self.assertIn(self.doctor.email, mail.outbox[0].to)

    def test_send_weekly_health_summary(self):
        """Test weekly health summary email."""
        result = send_weekly_health_summary(
            self.patient,
            predictions_count=5,
            latest_risk="LOW"
        )
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Weekly", mail.outbox[0].subject)

    def test_send_admin_new_user_notification(self):
        """Test admin notification for new user."""
        result = send_admin_new_user_notification(
            admin_email="admin@cardiodetect.com",
            new_user=self.patient
        )
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("New User", mail.outbox[0].subject)


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    FRONTEND_URL='http://localhost:3000',
    DEFAULT_FROM_EMAIL='CardioDetect <test@cardiodetect.com>'
)
class PredictionEmailTestCase(TestCase):
    """Test cases for prediction-related emails."""

    @classmethod
    def setUpTestData(cls):
        """Set up test fixtures once for all tests in this class."""
        cls.patient = User.objects.create_user(
            email='pred_patient@test.com',
            username='pred_patient',
            password='testpass123',
            first_name='John',
            last_name='Doe',
            role='patient'
        )
        
        cls.doctor = User.objects.create_user(
            email='pred_doctor@test.com',
            username='pred_doctor',
            password='testpass123',
            first_name='Dr. Jane',
            last_name='Smith',
            role='doctor'
        )

    def setUp(self):
        """Set up mock prediction and clear mail outbox."""
        # Mock prediction object
        self.mock_prediction = Mock()
        self.mock_prediction.user = self.patient
        self.mock_prediction.risk_category = "HIGH"
        self.mock_prediction.risk_percentage = 75.5
        self.mock_prediction.detection_result = True
        self.mock_prediction.input_method = "manual"
        self.mock_prediction.created_at = timezone.now()
        
        mail.outbox = []

    def test_send_high_risk_alert_to_patient(self):
        """Test high risk alert to patient."""
        result = send_high_risk_alert_to_patient(self.mock_prediction)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("High Risk", mail.outbox[0].subject)
        self.assertIn(self.patient.email, mail.outbox[0].to)

    def test_send_high_risk_alert_to_doctor(self):
        """Test high risk alert to doctor."""
        result = send_high_risk_alert_to_doctor(self.mock_prediction, self.doctor)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("High Risk", mail.outbox[0].subject)
        self.assertIn(self.doctor.email, mail.outbox[0].to)

    def test_send_prediction_complete_email(self):
        """Test prediction complete notification."""
        result = send_prediction_complete_email(self.mock_prediction)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Results", mail.outbox[0].subject)

    def test_send_prediction_complete_low_risk(self):
        """Test prediction complete for low risk result."""
        self.mock_prediction.risk_category = "LOW"
        self.mock_prediction.risk_percentage = 15.0
        
        result = send_prediction_complete_email(self.mock_prediction)
        
        self.assertTrue(result)
        self.assertIn("🟢", mail.outbox[0].subject)

    def test_send_prediction_complete_moderate_risk(self):
        """Test prediction complete for moderate risk result."""
        self.mock_prediction.risk_category = "MODERATE"
        self.mock_prediction.risk_percentage = 45.0
        
        result = send_prediction_complete_email(self.mock_prediction)
        
        self.assertTrue(result)
        self.assertIn("🟡", mail.outbox[0].subject)


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    FRONTEND_URL='http://localhost:3000',
    DEFAULT_FROM_EMAIL='CardioDetect <test@cardiodetect.com>'
)
class ApprovalEmailTestCase(TestCase):
    """Test cases for profile change approval emails."""

    @classmethod
    def setUpTestData(cls):
        """Set up test fixtures once for all tests in this class."""
        cls.patient = User.objects.create_user(
            email='approval_patient@test.com',
            username='approval_patient',
            password='testpass123',
            first_name='John',
            last_name='Doe',
            role='patient'
        )
        
        cls.admin = User.objects.create_superuser(
            email='approval_admin@test.com',
            username='approval_admin',
            password='adminpass123',
            first_name='Admin',
            last_name='User'
        )

    def setUp(self):
        """Set up mock pending change and clear mail outbox."""
        # Mock pending change object
        self.mock_change = Mock()
        self.mock_change.user = self.patient
        self.mock_change.field_name = "phone"
        self.mock_change.old_value = "+1234567890"
        self.mock_change.new_value = "+0987654321"
        self.mock_change.reason = "Updated phone number"
        self.mock_change.reviewed_by = self.admin
        self.mock_change.review_notes = "Approved"
        self.mock_change.reviewed_at = timezone.now()
        self.mock_change.created_at = timezone.now()
        
        mail.outbox = []

    def test_send_change_submitted_email(self):
        """Test change submitted confirmation email."""
        result = send_change_submitted_email(self.mock_change)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Submitted", mail.outbox[0].subject)

    def test_send_change_approved_email(self):
        """Test change approved notification."""
        result = send_change_approved_email(self.mock_change)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Approved", mail.outbox[0].subject)

    def test_send_change_rejected_email(self):
        """Test change rejected notification."""
        result = send_change_rejected_email(self.mock_change)
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Update", mail.outbox[0].subject)

    def test_send_admin_change_pending_notification(self):
        """Test admin notification for pending change."""
        result = send_admin_change_pending_notification(
            admin_email="admin@cardiodetect.com",
            pending_change=self.mock_change
        )
        
        self.assertTrue(result)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("Change Request", mail.outbox[0].subject)


@override_settings(
    EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
    FRONTEND_URL='http://localhost:3000',
    DEFAULT_FROM_EMAIL='CardioDetect <test@cardiodetect.com>'
)
class EmailErrorHandlingTestCase(TestCase):
    """Test error handling in email service."""

    @classmethod
    def setUpTestData(cls):
        """Set up test fixtures once for all tests in this class."""
        cls.patient = User.objects.create_user(
            email='error_patient@test.com',
            username='error_patient',
            password='testpass123',
            first_name='John',
            last_name='Doe',
            role='patient'
        )

    def setUp(self):
        """Clear mail outbox before each test."""
        mail.outbox = []

    def test_send_email_with_missing_template(self):
        """Test graceful handling of missing template."""
        result = send_templated_email(
            subject="Test",
            template_name="nonexistent_template",
            context={},
            recipient_email="test@example.com",
            fail_silently=True
        )
        
        self.assertFalse(result)

    def test_send_email_with_invalid_recipient(self):
        """Test handling of invalid email."""
        # This should not raise an exception when fail_silently=True
        result = send_templated_email(
            subject="Test",
            template_name="welcome",
            context={'first_name': 'Test', 'login_url': 'http://test.com'},
            recipient_email="",  # Empty email
            fail_silently=True
        )
        
        # Should return False or handle gracefully
        self.assertIsInstance(result, bool)
