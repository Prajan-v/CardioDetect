"""
Gold Standard User Model for CardioDetect.
Implements comprehensive security features and role-based access.
"""

import uuid
import secrets
from datetime import timedelta
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models
from django.utils import timezone
from django.core.validators import RegexValidator


class UserManager(BaseUserManager):
    """Custom user manager for email-based authentication."""
    
    def create_user(self, email, password=None, **extra_fields):
        """Create and save a regular user."""
        if not email:
            raise ValueError('Email is required')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        """Create and save a superuser."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('email_verified', True)
        extra_fields.setdefault('role', User.Role.ADMIN)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    """
    Gold Standard User Model with comprehensive security features.
    
    Features:
    - UUID primary key for security
    - Email-based authentication
    - Role-based access control
    - Email verification
    - Password reset tokens
    - Login tracking
    - Account lockout protection
    """
    
    class Role(models.TextChoices):
        PATIENT = 'patient', 'Patient'
        DOCTOR = 'doctor', 'Doctor'
        ADMIN = 'admin', 'Administrator'
    
    class Gender(models.TextChoices):
        MALE = 'M', 'Male'
        FEMALE = 'F', 'Female'
        OTHER = 'O', 'Other'
        PREFER_NOT = 'N', 'Prefer not to say'
    
    # ========== PRIMARY KEY ==========
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # ========== AUTHENTICATION ==========
    email = models.EmailField(
        unique=True,
        error_messages={'unique': 'A user with this email already exists.'}
    )
    username = models.CharField(
        max_length=150,
        unique=True,
        validators=[
            RegexValidator(
                regex=r'^[\w.@+-]+$',
                message='Username may only contain letters, digits, and @/./+/-/_ characters.'
            )
        ]
    )
    
    # Make email the login field
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'first_name', 'last_name']
    
    # ========== PROFILE ==========
    role = models.CharField(
        max_length=10,
        choices=Role.choices,
        default=Role.PATIENT
    )
    phone = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        validators=[
            RegexValidator(
                regex=r'^\+?1?\d{9,15}$',
                message='Phone number must be in format: +999999999. Up to 15 digits.'
            )
        ]
    )
    date_of_birth = models.DateField(blank=True, null=True)
    gender = models.CharField(
        max_length=1,
        choices=Gender.choices,
        blank=True,
        null=True
    )
    address = models.TextField(blank=True)
    city = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)
    profile_picture = models.ImageField(
        upload_to='profiles/%Y/%m/',
        blank=True,
        null=True
    )
    bio = models.TextField(max_length=500, blank=True)
    
    # ========== DOCTOR-SPECIFIC ==========
    license_number = models.CharField(max_length=50, blank=True)
    specialization = models.CharField(max_length=100, blank=True)
    hospital = models.CharField(max_length=200, blank=True)
    
    # ========== EMAIL VERIFICATION ==========
    email_verified = models.BooleanField(default=False)
    email_verification_token = models.CharField(max_length=64, blank=True)
    email_verification_sent_at = models.DateTimeField(null=True, blank=True)
    
    # ========== PASSWORD RESET ==========
    password_reset_token = models.CharField(max_length=64, blank=True)
    password_reset_sent_at = models.DateTimeField(null=True, blank=True)
    password_changed_at = models.DateTimeField(null=True, blank=True)
    
    # ========== SECURITY ==========
    failed_login_attempts = models.PositiveIntegerField(default=0)
    locked_until = models.DateTimeField(null=True, blank=True)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    last_login_user_agent = models.TextField(blank=True)
    
    # ========== TWO-FACTOR AUTH (Future) ==========
    two_factor_enabled = models.BooleanField(default=False)
    two_factor_secret = models.CharField(max_length=32, blank=True)
    two_factor_method = models.CharField(
        max_length=10,
        choices=[('app', 'Authenticator App'), ('email', 'Email')],
        default='app'
    )
    two_factor_email_code = models.CharField(max_length=6, blank=True)
    two_factor_email_code_sent_at = models.DateTimeField(null=True, blank=True)
    
    # ========== CONSENT & COMPLIANCE ==========
    terms_accepted = models.BooleanField(default=False)
    terms_accepted_at = models.DateTimeField(null=True, blank=True)
    privacy_accepted = models.BooleanField(default=False)
    privacy_accepted_at = models.DateTimeField(null=True, blank=True)
    marketing_consent = models.BooleanField(default=False)
    
    # ========== TIMESTAMPS ==========
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Custom manager
    objects = UserManager()
    
    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['email']),
            models.Index(fields=['role']),
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        return f"{self.get_full_name() or self.email} ({self.get_role_display()})"
    
    # ========== PROPERTIES ==========
    @property
    def is_doctor(self):
        return self.role == self.Role.DOCTOR
    
    @property
    def is_patient(self):
        return self.role == self.Role.PATIENT
    
    @property
    def is_admin_user(self):
        return self.role == self.Role.ADMIN
    
    @property
    def is_locked(self):
        """Check if account is locked due to failed login attempts."""
        if self.locked_until:
            return timezone.now() < self.locked_until
        return False
    
    @property
    def age(self):
        """Calculate age from date of birth."""
        if self.date_of_birth:
            today = timezone.now().date()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None
    
    # ========== METHODS ==========
    def generate_email_verification_token(self):
        """Generate a secure token for email verification."""
        self.email_verification_token = secrets.token_urlsafe(48)
        self.email_verification_sent_at = timezone.now()
        self.save(update_fields=['email_verification_token', 'email_verification_sent_at'])
        return self.email_verification_token
    
    def verify_email(self, token):
        """Verify email with token."""
        if not self.email_verification_token:
            return False
        
        # Check token expiry (24 hours)
        if self.email_verification_sent_at:
            expiry = self.email_verification_sent_at + timedelta(hours=24)
            if timezone.now() > expiry:
                return False
        
        if secrets.compare_digest(self.email_verification_token, token):
            self.email_verified = True
            self.email_verification_token = ''
            self.save(update_fields=['email_verified', 'email_verification_token'])
            return True
        return False
    
    def generate_password_reset_token(self):
        """Generate a secure token for password reset."""
        self.password_reset_token = secrets.token_urlsafe(48)
        self.password_reset_sent_at = timezone.now()
        self.save(update_fields=['password_reset_token', 'password_reset_sent_at'])
        return self.password_reset_token
    
    def verify_password_reset_token(self, token):
        """Verify password reset token."""
        if not self.password_reset_token:
            return False
        
        # Check token expiry (1 hour)
        if self.password_reset_sent_at:
            expiry = self.password_reset_sent_at + timedelta(hours=1)
            if timezone.now() > expiry:
                return False
        
        return secrets.compare_digest(self.password_reset_token, token)
    
    def clear_password_reset_token(self):
        """Clear password reset token after use."""
        self.password_reset_token = ''
        self.password_reset_sent_at = None
        self.password_changed_at = timezone.now()
        self.save(update_fields=['password_reset_token', 'password_reset_sent_at', 'password_changed_at'])
    
    def record_login_attempt(self, success, ip_address=None, user_agent=''):
        """Record login attempt for security tracking."""
        was_unlocked = not self.is_locked
        
        if success:
            self.failed_login_attempts = 0
            self.locked_until = None
            self.last_login = timezone.now()
            self.last_login_ip = ip_address
            self.last_login_user_agent = user_agent[:500] if user_agent else ''
        else:
            self.failed_login_attempts += 1
            # Lock account after 5 failed attempts for 30 minutes
            if self.failed_login_attempts >= 5:
                self.locked_until = timezone.now() + timedelta(minutes=30)
                
                # Send account locked email only when account just got locked
                if was_unlocked:
                    from .email_service import send_account_locked_email
                    send_account_locked_email(self, 'Too many failed login attempts (5 or more)')
        
        self.save(update_fields=[
            'failed_login_attempts', 'locked_until', 
            'last_login', 'last_login_ip', 'last_login_user_agent'
        ])
    
    def accept_terms(self):
        """Record terms acceptance."""
        self.terms_accepted = True
        self.terms_accepted_at = timezone.now()
        self.save(update_fields=['terms_accepted', 'terms_accepted_at'])
    
    def accept_privacy(self):
        """Record privacy policy acceptance."""
        self.privacy_accepted = True
        self.privacy_accepted_at = timezone.now()
        self.save(update_fields=['privacy_accepted', 'privacy_accepted_at'])


class LoginHistory(models.Model):
    """Track login history for security auditing."""
    
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='login_history'
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    location = models.CharField(max_length=200, blank=True)
    success = models.BooleanField(default=True)
    failure_reason = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'Login histories'
        indexes = [
            models.Index(fields=['user', '-timestamp']),
        ]
    
    def __str__(self):
        status = 'Success' if self.success else 'Failed'
        email = self.user.email if self.user else 'Unknown'
        return f"{email} - {status} - {self.timestamp}"


class RefreshTokenBlacklist(models.Model):
    """Blacklist for invalidated refresh tokens."""
    
    token = models.CharField(max_length=500, unique=True)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='blacklisted_tokens'
    )
    blacklisted_at = models.DateTimeField(auto_now_add=True)
    reason = models.CharField(max_length=100, blank=True)
    
    class Meta:
        ordering = ['-blacklisted_at']
    
    
    def __str__(self):
        return f"Blacklisted token for {self.user.email}"


class DataDeletionRequest(models.Model):
    """
    Tracks data deletion requests with 7-day grace period.
    GDPR Article 17 compliant.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending (7-day grace period)'
        CANCELLED = 'cancelled', 'Cancelled by User'
        COMPLETED = 'completed', 'Deletion Completed'
        FAILED = 'failed', 'Deletion Failed'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='deletion_requests'
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    reason = models.TextField(blank=True, help_text="Optional reason for deletion")
    
    # Timestamps
    requested_at = models.DateTimeField(auto_now_add=True)
    scheduled_deletion_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    
    # Audit
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-requested_at']
        verbose_name = 'Data Deletion Request'
        verbose_name_plural = 'Data Deletion Requests'
    
    def save(self, *args, **kwargs):
        if not self.scheduled_deletion_at:
            self.scheduled_deletion_at = timezone.now() + timedelta(days=7)
        super().save(*args, **kwargs)
    
    @property
    def is_cancellable(self) -> bool:
        return (
            self.status == self.Status.PENDING and
            timezone.now() < self.scheduled_deletion_at
        )
    
    @property
    def grace_period_remaining(self) -> timedelta:
        if self.status != self.Status.PENDING:
            return timedelta(0)
        remaining = self.scheduled_deletion_at - timezone.now()
        return max(remaining, timedelta(0))
    
    def cancel(self):
        if self.is_cancellable:
            self.status = self.Status.CANCELLED
            self.cancelled_at = timezone.now()
            self.save()
            return True
        return False
    
    def __str__(self):
        return f"Deletion Request: {self.user.email} - {self.status}"


class ConsentRecord(models.Model):
    """
    Tracks user consent for GDPR compliance.
    Immutable audit trail of all consent changes.
    """
    
    class ConsentType(models.TextChoices):
        TERMS = 'terms', 'Terms of Service'
        PRIVACY = 'privacy', 'Privacy Policy'
        MARKETING = 'marketing', 'Marketing Communications'
        DATA_PROCESSING = 'data_processing', 'Medical Data Processing'
        THIRD_PARTY = 'third_party', 'Third-Party Sharing'
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='consent_records'
    )
    consent_type = models.CharField(max_length=20, choices=ConsentType.choices)
    granted = models.BooleanField()
    version = models.CharField(max_length=20, help_text="Policy version consented to")
    
    # Audit fields
    recorded_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-recorded_at']
        verbose_name = 'Consent Record'
        verbose_name_plural = 'Consent Records'
    
    def __str__(self):
        action = "granted" if self.granted else "revoked"
        return f"{self.user.email} {action} {self.consent_type} (v{self.version})"
