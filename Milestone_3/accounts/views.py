"""
Gold Standard Authentication Views.
Implements secure endpoints for user authentication workflows.
"""

from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils import timezone
from rest_framework import status, generics, viewsets
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from rest_framework_simplejwt.tokens import RefreshToken

from .models import User, LoginHistory, RefreshTokenBlacklist
from predictions.models import SystemNotification
from .serializers import (
    UserRegistrationSerializer,
    CustomTokenObtainPairSerializer,
    PasswordResetRequestSerializer,
    PasswordResetConfirmSerializer,
    PasswordChangeSerializer,
    EmailVerificationSerializer,
    ResendVerificationSerializer,
    UserProfileSerializer,
    UserProfileUpdateSerializer,
    LoginHistorySerializer,
    NotificationSerializer,
)


class RegisterView(APIView):
    """
    User registration endpoint.
    
    POST /api/auth/register/
    
    Creates a new user account and sends email verification.
    """
    permission_classes = [AllowAny]
    throttle_scope = 'registration'
    
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.save()
            
            # Send welcome email
            from .email_service import send_welcome_email
            send_welcome_email(user)

            # Send verification email
            self._send_verification_email(user, request)
            
            # Generate tokens for immediate login (optional)
            refresh = RefreshToken.for_user(user)
            
            return Response({
                'status': 'success',
                'message': 'Registration successful. Please check your email to verify your account.',
                'user': {
                    'id': str(user.id),
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'role': user.role,
                    'email_verified': user.email_verified,
                },
                'tokens': {
                    'access': str(refresh.access_token),
                    'refresh': str(refresh),
                }
            }, status=status.HTTP_201_CREATED)
        
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    def _send_verification_email(self, user, request):
        """Send email verification link with professional HTML template."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Build frontend URL for verification using settings
            frontend_url = getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')
            verification_url = f"{frontend_url}/verify-email?email={user.email}&token={user.email_verification_token}"
            
            logger.info(f"Sending verification email to {user.email}")
            logger.info(f"Verification URL: {verification_url}")
            
            subject = 'Cardio Detect - Verify Your Email'
            
            # Render HTML template
            html_message = render_to_string('emails/verify_email.html', {
                'first_name': user.first_name or 'there',
                'verification_url': verification_url,
                'user_email': user.email,
            })
            
            # Plain text fallback
            text_message = f"""
Hello {user.first_name or 'there'},

Thank you for registering with Cardio Detect!

Please verify your email by clicking the link below:
{verification_url}

This link will expire in 24 hours.

If you did not create this account, please ignore this email.

Best regards,
The Cardio Detect Team
            """
            
            from django.core.mail import EmailMultiAlternatives
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[user.email],
            )
            email.attach_alternative(html_message, "text/html")
            result = email.send(fail_silently=False)
            logger.info(f"Verification email sent successfully to {user.email}, result: {result}")
            
        except Exception as e:
            # Log error but don't fail registration
            import traceback
            logger.error(f"Failed to send verification email to {user.email}: {e}")
            logger.error(traceback.format_exc())
            print(f"Failed to send verification email: {e}")




class LoginView(TokenObtainPairView):
    """
    User login endpoint with enhanced security.
    
    POST /api/auth/login/
    
    Features:
    - Account lockout protection
    - Login attempt tracking
    - Custom JWT claims
    """
    serializer_class = CustomTokenObtainPairSerializer
    throttle_scope = 'login'


class LogoutView(APIView):
    """
    User logout endpoint.
    
    POST /api/auth/logout/
    
    Blacklists the refresh token to prevent reuse.
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            refresh_token = request.data.get('refresh')
            
            if refresh_token:
                # Blacklist the token
                RefreshTokenBlacklist.objects.create(
                    token=refresh_token,
                    user=request.user,
                    reason='User logout'
                )
                
                # Invalidate the token
                token = RefreshToken(refresh_token)
                token.blacklist()
            
            return Response({
                'status': 'success',
                'message': 'Successfully logged out.'
            })
        except Exception as e:
            return Response({
                'status': 'success',
                'message': 'Logged out.'
            })


class VerifyEmailView(APIView):
    """
    Email verification endpoint.
    
    GET /api/auth/verify-email/?email=...&token=...
    POST /api/auth/verify-email/ (with body)
    """
    permission_classes = [AllowAny]
    
    def get(self, request):
        """Verify email via URL query params."""
        email = request.query_params.get('email')
        token = request.query_params.get('token')
        
        return self._verify(email, token)
    
    def post(self, request):
        """Verify email via POST body."""
        serializer = EmailVerificationSerializer(data=request.data)
        if serializer.is_valid():
            return self._verify(
                serializer.validated_data['email'],
                serializer.validated_data['token']
            )
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    def _verify(self, email, token):
        if not email or not token:
            return Response({
                'status': 'error',
                'message': 'Email and token are required.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email__iexact=email)
        except User.DoesNotExist:
            return Response({
                'status': 'error',
                'message': 'Invalid verification link.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if user.email_verified:
            return Response({
                'status': 'success',
                'message': 'Email already verified.'
            })
        
        if user.verify_email(token):
            # Send welcome email now that account is verified
            from .email_service import send_welcome_email
            send_welcome_email(user)
            
            return Response({
                'status': 'success',
                'message': 'Email verified successfully. You can now login.'
            })
        else:
            return Response({
                'status': 'error',
                'message': 'Invalid or expired verification link. Please request a new one.'
            }, status=status.HTTP_400_BAD_REQUEST)


class ResendVerificationView(APIView):
    """
    Resend email verification link.
    
    POST /api/auth/resend-verification/
    """
    permission_classes = [AllowAny]
    throttle_scope = 'email'
    
    def post(self, request):
        serializer = ResendVerificationSerializer(data=request.data)
        
        if serializer.is_valid():
            email = serializer.validated_data['email']
            
            try:
                user = User.objects.get(email__iexact=email)
                
                if user.email_verified:
                    return Response({
                        'status': 'success',
                        'message': 'Email already verified.'
                    })
                
                # Check rate limit (don't send more than once per 5 minutes)
                if user.email_verification_sent_at:
                    time_diff = timezone.now() - user.email_verification_sent_at
                    if time_diff.seconds < 300:
                        return Response({
                            'status': 'error',
                            'message': f'Please wait {300 - time_diff.seconds} seconds before requesting again.'
                        }, status=status.HTTP_429_TOO_MANY_REQUESTS)
                
                # Generate new token and send email
                user.generate_email_verification_token()
                self._send_verification_email(user, request)
                
            except User.DoesNotExist:
                pass  # Don't reveal if email exists
            
            return Response({
                'status': 'success',
                'message': 'If an account exists with this email, a verification link has been sent.'
            })
        
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    def _send_verification_email(self, user, request):
        """Send email verification link."""
        try:
            verification_url = f"{request.build_absolute_uri('/')[:-1]}/api/auth/verify-email/?email={user.email}&token={user.email_verification_token}"
            
            subject = 'CardioDetect - Verify Your Email'
            message = f"""
Hello {user.first_name},

Please verify your email by clicking the link below:
{verification_url}

This link will expire in 24 hours.

Best regards,
The CardioDetect Team
            """
            
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL if hasattr(settings, 'DEFAULT_FROM_EMAIL') else 'noreply@cardiodetect.com',
                recipient_list=[user.email],
                fail_silently=True,
            )
        except Exception as e:
            print(f"Failed to send verification email: {e}")


class PasswordResetRequestView(APIView):
    """
    Password reset request endpoint.
    
    POST /api/auth/password-reset/
    
    Sends password reset link to email.
    """
    permission_classes = [AllowAny]
    throttle_scope = 'password_reset'
    
    def post(self, request):
        serializer = PasswordResetRequestSerializer(data=request.data)
        
        if serializer.is_valid():
            email = serializer.validated_data['email']
            
            try:
                user = User.objects.get(email__iexact=email)
                
                # Check rate limit
                if user.password_reset_sent_at:
                    time_diff = timezone.now() - user.password_reset_sent_at
                    if time_diff.seconds < 300:
                        # Still return success to not reveal user existence
                        pass
                    else:
                        user.generate_password_reset_token()
                        self._send_password_reset_email(user, request)
                else:
                    user.generate_password_reset_token()
                    self._send_password_reset_email(user, request)
                    
            except User.DoesNotExist:
                pass  # Don't reveal if email exists
            
            return Response({
                'status': 'success',
                'message': 'If an account exists with this email, a password reset link has been sent.'
            })
        
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)
    
    def _send_password_reset_email(self, user, request):
        """Send password reset email with professional HTML template."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Build frontend URL for reset using settings
            frontend_url = getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')
            reset_url = f"{frontend_url}/reset-password?email={user.email}&token={user.password_reset_token}"
            
            logger.info(f"Sending password reset email to {user.email}")
            logger.info(f"Reset URL: {reset_url}")
            
            subject = 'Cardio Detect - Password Reset Request'
            
            # Render HTML template
            html_message = render_to_string('emails/password_reset.html', {
                'first_name': user.first_name or 'there',
                'reset_url': reset_url,
                'user_email': user.email,
            })
            
            # Plain text fallback
            text_message = f"""
Hello {user.first_name or 'there'},

We received a request to reset your password for your Cardio Detect account.

Click the link below to reset your password:
{reset_url}

This link will expire in 1 hour.

If you did not request this, please ignore this email or contact support if you have concerns.

Best regards,
The Cardio Detect Team
            """
            
            from django.core.mail import EmailMultiAlternatives
            email = EmailMultiAlternatives(
                subject=subject,
                body=text_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                to=[user.email],
            )
            email.attach_alternative(html_message, "text/html")
            result = email.send(fail_silently=False)
            logger.info(f"Password reset email sent successfully to {user.email}, result: {result}")
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to send password reset email to {user.email}: {e}")
            logger.error(traceback.format_exc())
            print(f"Failed to send password reset email: {e}")



class PasswordResetConfirmView(APIView):
    """
    Password reset confirmation endpoint.
    
    POST /api/auth/password-reset/confirm/
    
    Resets password using token.
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = PasswordResetConfirmSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.validated_data['user']
            
            # Set new password
            user.set_password(serializer.validated_data['new_password'])
            user.clear_password_reset_token()
            user.save()
            
            # Invalidate all existing sessions/tokens
            # (Optional: blacklist all refresh tokens for this user)
            
            return Response({
                'status': 'success',
                'message': 'Password reset successfully. You can now login with your new password.'
            })
        
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Password reset confirm errors: {serializer.errors}")
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class PasswordChangeView(APIView):
    """
    Password change endpoint for authenticated users.
    
    POST /api/auth/password-change/
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = PasswordChangeSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            # Fetch fresh user from database to avoid stale data
            user = User.objects.get(pk=request.user.pk)
            user.set_password(serializer.validated_data['new_password'])
            user.password_changed_at = timezone.now()
            # Explicitly save password field to ensure it persists
            user.save(update_fields=['password', 'password_changed_at'])
            
            # Send password changed email notification
            from .email_service import send_password_changed_email
            ip_address = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', ''))
            send_password_changed_email(user, ip_address)
            
            return Response({
                'status': 'success',
                'message': 'Password changed successfully.'
            })
        
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class ProfileView(APIView):
    """
    User profile endpoint.
    
    GET /api/auth/profile/ - Get profile
    PUT /api/auth/profile/ - Update profile
    PATCH /api/auth/profile/ - Partial update
    
    For doctors: Changes go through approval workflow
    For patients: Changes are applied instantly
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        serializer = UserProfileSerializer(request.user)
        return Response({
            'status': 'success',
            'data': serializer.data
        })
    
    def put(self, request):
        return self._update_profile(request, partial=False)
    
    def patch(self, request):
        return self._update_profile(request, partial=True)
    
    def _update_profile(self, request, partial=False):
        user = request.user
        
        # For doctors: Create pending changes that require admin approval
        if user.role == 'doctor':
            from .pending_changes import PendingProfileChange
            pending_created = []
            
            for field, new_value in request.data.items():
                # Skip fields that shouldn't be changed via profile
                if field in ['email', 'password', 'role', 'is_staff', 'is_superuser']:
                    continue
                
                # Get current value
                old_value = getattr(user, field, None)
                if old_value is None and not hasattr(user, field):
                    continue
                    
                # Only create pending change if value actually changed
                if str(old_value) != str(new_value):
                    PendingProfileChange.objects.create(
                        user=user,
                        field_name=field,
                        old_value=str(old_value) if old_value else '',
                        new_value=str(new_value),
                        reason='Profile update via dashboard'
                    )
                    pending_created.append(field)
            
            if pending_created:
                return Response({
                    'status': 'pending',
                    'message': f'Profile change(s) submitted for admin approval: {", ".join(pending_created)}',
                    'pending_fields': pending_created,
                    'data': UserProfileSerializer(user).data
                })
            else:
                return Response({
                    'status': 'success',
                    'message': 'No changes detected.',
                    'data': UserProfileSerializer(user).data
                })
        
        # For patients and admins: Apply changes instantly
        serializer = UserProfileUpdateSerializer(
            user,
            data=request.data,
            partial=partial
        )
        
        if serializer.is_valid():
            serializer.save()
            return Response({
                'status': 'success',
                'message': 'Profile updated successfully.',
                'data': UserProfileSerializer(user).data
            })
        
        return Response({
            'status': 'error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class LoginHistoryView(generics.ListAPIView):
    """
    User login history endpoint.
    
    GET /api/auth/login-history/
    """
    permission_classes = [IsAuthenticated]
    serializer_class = LoginHistorySerializer
    
    def get_queryset(self):
        return LoginHistory.objects.filter(user=self.request.user)[:50]


class DeleteAccountView(APIView):
    """
    Account deletion endpoint.
    
    DELETE /api/auth/delete-account/
    
    Requires password confirmation for security.
    """
    permission_classes = [IsAuthenticated]
    
    def delete(self, request):
        password = request.data.get('password')
        
        if not password:
            return Response({
                'status': 'error',
                'message': 'Password is required to delete account.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not request.user.check_password(password):
            return Response({
                'status': 'error',
                'message': 'Incorrect password.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Soft delete by deactivating (or hard delete if preferred)
        request.user.is_active = False
        request.user.save()
        
        return Response({
            'status': 'success',
            'message': 'Account deleted successfully.'
        })


class CheckEmailView(APIView):
    """
    Check if email is already registered.
    
    POST /api/auth/check-email/
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        email = request.data.get('email', '').lower()
        
        if not email:
            return Response({
                'status': 'error',
                'message': 'Email is required.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        exists = User.objects.filter(email__iexact=email).exists()
        
        return Response({
            'status': 'success',
            'available': not exists
        })


class WhoAmIView(APIView):
    """
    Get current authenticated user info.
    
    GET /api/auth/me/
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        return Response({
            'status': 'success',
            'data': {
                'id': str(user.id),
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'full_name': user.get_full_name(),
                'role': user.role,
                'email_verified': user.email_verified,
                'created_at': user.created_at.isoformat(),
            }
        })


class NotificationViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = NotificationSerializer
    
    def get_queryset(self):
        return SystemNotification.objects.filter(user=self.request.user)
    
    @action(detail=False, methods=['post'])
    def mark_all_read(self, request):
        self.get_queryset().update(is_read=True)
        return Response({'status': 'success'})
        
    @action(detail=True, methods=['post'])
    def mark_read(self, request, pk=None):
        notification = self.get_object()
        notification.is_read = True
        notification.save()
        return Response({'status': 'success'})
