"""
Gold Standard Authentication Serializers.
Implements comprehensive validation and security for user authentication.
"""

import re
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import User, LoginHistory
from predictions.models import SystemNotification


class UserRegistrationSerializer(serializers.ModelSerializer):
    """
    User registration with comprehensive validation.
    
    Validates:
    - Email format and uniqueness
    - Password strength (min 8 chars, uppercase, lowercase, number, special char)
    - Password confirmation match
    - Terms and privacy acceptance
    """
    
    email = serializers.EmailField(
        required=True,
        error_messages={'invalid': 'Please enter a valid email address.'}
    )
    password = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'},
        min_length=8,
        error_messages={
            'min_length': 'Password must be at least 8 characters long.'
        }
    )
    password_confirm = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    first_name = serializers.CharField(required=True, max_length=150)
    last_name = serializers.CharField(required=False, max_length=150, allow_blank=True)
    terms_accepted = serializers.BooleanField(required=True)
    privacy_accepted = serializers.BooleanField(required=True)
    role = serializers.ChoiceField(
        choices=[('patient', 'Patient'), ('doctor', 'Doctor')],
        default='patient'
    )
    
    class Meta:
        model = User
        fields = [
            'email', 'username', 'password', 'password_confirm',
            'first_name', 'last_name', 'phone', 'date_of_birth', 'gender',
            'role', 'terms_accepted', 'privacy_accepted',
            # Doctor-specific
            'license_number', 'specialization', 'hospital'
        ]
        extra_kwargs = {
            'username': {'required': False},
            'phone': {'required': False},
            'date_of_birth': {'required': False},
            'gender': {'required': False},
            'license_number': {'required': False},
            'specialization': {'required': False},
            'hospital': {'required': False},
        }
    
    def validate_email(self, value):
        """Validate email uniqueness (case-insensitive)."""
        email = value.lower()
        if User.objects.filter(email__iexact=email).exists():
            raise serializers.ValidationError('A user with this email already exists.')
        return email
    
    def validate_password(self, value):
        """Validate password strength."""
        errors = []
        
        if len(value) < 8:
            errors.append('Password must be at least 8 characters long.')
        if not re.search(r'[A-Z]', value):
            errors.append('Password must contain at least one uppercase letter.')
        if not re.search(r'[a-z]', value):
            errors.append('Password must contain at least one lowercase letter.')
        if not re.search(r'\d', value):
            errors.append('Password must contain at least one number.')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            errors.append('Password must contain at least one special character (!@#$%^&*(),.?":{}|<>).')
        
        # Django's built-in password validation
        try:
            validate_password(value)
        except Exception as e:
            errors.extend(list(e.messages))
        
        if errors:
            raise serializers.ValidationError(errors)
        return value
    
    def validate_terms_accepted(self, value):
        """Ensure terms are accepted."""
        if not value:
            raise serializers.ValidationError('You must accept the Terms of Service.')
        return value
    
    def validate_privacy_accepted(self, value):
        """Ensure privacy policy is accepted."""
        if not value:
            raise serializers.ValidationError('You must accept the Privacy Policy.')
        return value
    
    def validate(self, attrs):
        """Cross-field validation."""
        # Password confirmation
        if attrs.get('password') != attrs.get('password_confirm'):
            raise serializers.ValidationError({
                'password_confirm': 'Passwords do not match.'
            })
        
        # Doctor-specific validation
        if attrs.get('role') == 'doctor':
            if not attrs.get('license_number'):
                raise serializers.ValidationError({
                    'license_number': 'License number is required for doctors.'
                })
        
        return attrs
    
    def create(self, validated_data):
        """Create user with proper password hashing."""
        # Remove password_confirm as it's not a model field
        validated_data.pop('password_confirm', None)
        
        # Generate username from email if not provided
        if not validated_data.get('username'):
            base_username = validated_data['email'].split('@')[0]
            username = base_username
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{base_username}{counter}"
                counter += 1
            validated_data['username'] = username
        
        # Extract password
        password = validated_data.pop('password')
        
        # Handle terms and privacy
        terms = validated_data.pop('terms_accepted', False)
        privacy = validated_data.pop('privacy_accepted', False)
        
        # Create user
        user = User.objects.create_user(password=password, **validated_data)
        
        # Record consent
        if terms:
            user.accept_terms()
        if privacy:
            user.accept_privacy()
        
        # Generate email verification token
        user.generate_email_verification_token()
        
        return user


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """
    Custom JWT token serializer with enhanced security.
    
    Features:
    - Account lockout check
    - Email verification check
    - Login attempt tracking
    - Custom token claims
    """
    
    def validate(self, attrs):
        email = attrs.get('email', '').lower()
        password = attrs.get('password', '')
        
        # Get request for IP tracking
        request = self.context.get('request')
        ip_address = request.META.get('REMOTE_ADDR') if request else None
        user_agent = request.META.get('HTTP_USER_AGENT', '') if request else ''
        
        # Find user by email
        try:
            user = User.objects.get(email__iexact=email)
        except User.DoesNotExist:
            # Log failed attempt with unknown user
            raise serializers.ValidationError({
                'detail': 'Invalid email or password.'
            })
        
        # Check if account is locked
        if user.is_locked:
            lock_remaining = (user.locked_until - timezone.now()).seconds // 60
            raise serializers.ValidationError({
                'detail': f'Account is locked. Try again in {lock_remaining} minutes.',
                'code': 'account_locked'
            })
        
        # Check if account is active
        if not user.is_active:
            raise serializers.ValidationError({
                'detail': 'This account has been deactivated.',
                'code': 'account_inactive'
            })
        
        # Authenticate
        authenticated_user = authenticate(
            request=request,
            email=email,
            password=password
        )
        
        if authenticated_user is None:
            # Record failed attempt
            user.record_login_attempt(success=False, ip_address=ip_address)
            LoginHistory.objects.create(
                user=user,
                ip_address=ip_address,
                user_agent=user_agent[:500],
                success=False,
                failure_reason='Invalid password'
            )
            
            # Calculate remaining attempts
            remaining = 5 - user.failed_login_attempts
            if remaining > 0:
                raise serializers.ValidationError({
                    'detail': f'Invalid password. {remaining} attempts remaining.',
                    'code': 'invalid_credentials'
                })
            else:
                raise serializers.ValidationError({
                    'detail': 'Account locked due to too many failed attempts.',
                    'code': 'account_locked'
                })
        
        # Check email verification (optional - can be enforced)
        # if not authenticated_user.email_verified:
        #     raise serializers.ValidationError({
        #         'detail': 'Please verify your email address.',
        #         'code': 'email_not_verified'
        #     })
        
        # Record successful login
        authenticated_user.record_login_attempt(
            success=True,
            ip_address=ip_address,
            user_agent=user_agent
        )
        LoginHistory.objects.create(
            user=authenticated_user,
            ip_address=ip_address,
            user_agent=user_agent[:500],
            success=True
        )
        
        # Get tokens
        data = super().validate(attrs)
        
        # Add custom response data
        data['user'] = {
            'id': str(authenticated_user.id),
            'email': authenticated_user.email,
            'first_name': authenticated_user.first_name,
            'last_name': authenticated_user.last_name,
            'full_name': authenticated_user.get_full_name(),
            'role': authenticated_user.role,
            'email_verified': authenticated_user.email_verified,
        }
        
        return data
    
    @classmethod
    def get_token(cls, user):
        """Add custom claims to JWT token."""
        token = super().get_token(user)
        
        # Add custom claims
        token['email'] = user.email
        token['role'] = user.role
        token['name'] = user.get_full_name()
        
        return token


class PasswordResetRequestSerializer(serializers.Serializer):
    """Serializer for password reset request."""
    
    email = serializers.EmailField(required=True)
    
    def validate_email(self, value):
        """Check if user exists."""
        # Always return success even if email doesn't exist (security)
        return value.lower()


class PasswordResetConfirmSerializer(serializers.Serializer):
    """Serializer for password reset confirmation."""
    
    token = serializers.CharField(required=True)
    email = serializers.EmailField(required=True)
    new_password = serializers.CharField(
        write_only=True,
        required=True,
        min_length=8
    )
    new_password_confirm = serializers.CharField(
        write_only=True,
        required=True
    )
    
    def validate_new_password(self, value):
        """Validate new password strength."""
        errors = []
        
        if len(value) < 8:
            errors.append('Password must be at least 8 characters long.')
        if not re.search(r'[A-Z]', value):
            errors.append('Password must contain at least one uppercase letter.')
        if not re.search(r'[a-z]', value):
            errors.append('Password must contain at least one lowercase letter.')
        if not re.search(r'\d', value):
            errors.append('Password must contain at least one number.')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            errors.append('Password must contain at least one special character.')
        
        if errors:
            raise serializers.ValidationError(errors)
        return value
    
    def validate(self, attrs):
        """Validate token and password confirmation."""
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError({
                'new_password_confirm': 'Passwords do not match.'
            })
        
        # Verify token
        try:
            user = User.objects.get(email__iexact=attrs['email'])
        except User.DoesNotExist:
            raise serializers.ValidationError({
                'email': 'Invalid email or token.'
            })
        
        if not user.verify_password_reset_token(attrs['token']):
            raise serializers.ValidationError({
                'token': 'Invalid or expired token. Please request a new password reset.'
            })
        
        attrs['user'] = user
        return attrs


class PasswordChangeSerializer(serializers.Serializer):
    """Serializer for authenticated password change."""
    
    current_password = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    new_password = serializers.CharField(
        write_only=True,
        required=True,
        min_length=8,
        style={'input_type': 'password'}
    )
    new_password_confirm = serializers.CharField(
        write_only=True,
        required=True,
        style={'input_type': 'password'}
    )
    
    def validate_new_password(self, value):
        """Validate new password strength."""
        errors = []
        
        if len(value) < 8:
            errors.append('Password must be at least 8 characters long.')
        if not re.search(r'[A-Z]', value):
            errors.append('Password must contain at least one uppercase letter.')
        if not re.search(r'[a-z]', value):
            errors.append('Password must contain at least one lowercase letter.')
        if not re.search(r'\d', value):
            errors.append('Password must contain at least one number.')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            errors.append('Password must contain at least one special character.')
        
        if errors:
            raise serializers.ValidationError(errors)
        return value
    
    def validate(self, attrs):
        """Validate current password and confirmation."""
        user = self.context['request'].user
        
        # Verify current password
        if not user.check_password(attrs['current_password']):
            raise serializers.ValidationError({
                'current_password': 'Current password is incorrect.'
            })
        
        # Check passwords match
        if attrs['new_password'] != attrs['new_password_confirm']:
            raise serializers.ValidationError({
                'new_password_confirm': 'Passwords do not match.'
            })
        
        # Prevent reusing the current password
        if attrs['current_password'] == attrs['new_password']:
            raise serializers.ValidationError({
                'new_password': 'New password must be different from current password.'
            })
        
        # Run Django's built-in validators (includes CommonPasswordValidator)
        try:
            validate_password(attrs['new_password'], user=user)
        except Exception as e:
            raise serializers.ValidationError({
                'new_password': list(e.messages)
            })
        
        return attrs



class EmailVerificationSerializer(serializers.Serializer):
    """Serializer for email verification."""
    
    email = serializers.EmailField(required=True)
    token = serializers.CharField(required=True)


class ResendVerificationSerializer(serializers.Serializer):
    """Serializer for resending email verification."""
    
    email = serializers.EmailField(required=True)


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for user profile view/update."""
    
    age = serializers.ReadOnlyField()
    full_name = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = [
            'id', 'email', 'username', 'first_name', 'last_name', 'full_name',
            'phone', 'date_of_birth', 'age', 'gender', 'role',
            'address', 'city', 'country', 'profile_picture', 'bio',
            'email_verified', 'created_at', 'last_login',
            # Doctor-specific
            'license_number', 'specialization', 'hospital',
            # Consent
            'terms_accepted', 'privacy_accepted', 'marketing_consent'
        ]
        read_only_fields = [
            'id', 'email', 'role', 'email_verified', 
            'created_at', 'last_login', 'terms_accepted', 'privacy_accepted'
        ]
    
    def get_full_name(self, obj):
        return obj.get_full_name()


class UserProfileUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user profile."""
    
    class Meta:
        model = User
        fields = [
            'first_name', 'last_name', 'phone', 'date_of_birth', 'gender',
            'address', 'city', 'country', 'profile_picture', 'bio',
            'license_number', 'specialization', 'hospital', 'marketing_consent'
        ]
    
    def validate_phone(self, value):
        """Validate phone format."""
        if value and not re.match(r'^\+?1?\d{9,15}$', value):
            raise serializers.ValidationError(
                'Phone number must be in format: +999999999. Up to 15 digits.'
            )
        return value


class LoginHistorySerializer(serializers.ModelSerializer):
    """Serializer for login history."""
    
    class Meta:
        model = LoginHistory
        fields = ['ip_address', 'login_time', 'success', 'user_agent']


class NotificationSerializer(serializers.ModelSerializer):
    time_ago = serializers.SerializerMethodField()
    type = serializers.CharField(source='notification_type')
    
    class Meta:
        model = SystemNotification
        fields = ['id', 'title', 'message', 'type', 'is_read', 'created_at', 'time_ago', 'related_prediction']
        
    def get_time_ago(self, obj):
        from django.utils.timesince import timesince
        return timesince(obj.created_at) + " ago"
