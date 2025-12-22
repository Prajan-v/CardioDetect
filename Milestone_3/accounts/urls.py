"""
URL Configuration for accounts app.
Complete authentication routes + GDPR compliance endpoints.
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    RegisterView,
    LoginView,
    LogoutView,
    VerifyEmailView,
    ResendVerificationView,
    PasswordResetRequestView,
    PasswordResetConfirmView,
    PasswordChangeView,
    ProfileView,
    LoginHistoryView,
    DeleteAccountView,
    CheckEmailView,
    WhoAmIView,
)
from .approval_views import (
    SubmitProfileChangeView,
    MyPendingChangesView,
    AdminPendingChangesView,
    ApproveChangeView,
    RejectChangeView,
)
from .gdpr_views import (
    DataExportView,
    DataDeletionView,
    ConsentHistoryView,
    ConsentActionView,
    AdminDeletionRequestsView,
)
from .views import NotificationViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'notifications', NotificationViewSet, basename='notification')

app_name = 'accounts'

urlpatterns = [
    # ========== REGISTRATION ==========
    path('register/', RegisterView.as_view(), name='register'),
    path('check-email/', CheckEmailView.as_view(), name='check-email'),
    
    # ========== LOGIN / LOGOUT ==========
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
    
    # ========== EMAIL VERIFICATION ==========
    path('verify-email/', VerifyEmailView.as_view(), name='verify-email'),
    path('resend-verification/', ResendVerificationView.as_view(), name='resend-verification'),
    
    # ========== PASSWORD MANAGEMENT ==========
    path('password-reset/', PasswordResetRequestView.as_view(), name='password-reset'),
    path('password-reset/confirm/', PasswordResetConfirmView.as_view(), name='password-reset-confirm'),
    path('password-change/', PasswordChangeView.as_view(), name='password-change'),
    
    # ========== PROFILE ==========
    path('profile/', ProfileView.as_view(), name='profile'),
    path('me/', WhoAmIView.as_view(), name='whoami'),
    
    # ========== ADMIN APPROVAL ==========
    path('profile-changes/', MyPendingChangesView.as_view(), name='my-profile-changes'),
    path('profile-changes/submit/', SubmitProfileChangeView.as_view(), name='submit-profile-change'),
    path('admin/pending-changes/', AdminPendingChangesView.as_view(), name='admin-pending-changes'),
    path('admin/approve/<int:change_id>/', ApproveChangeView.as_view(), name='approve-change'),
    path('admin/reject/<int:change_id>/', RejectChangeView.as_view(), name='reject-change'),
    path('admin/deletion-requests/', AdminDeletionRequestsView.as_view(), name='admin-deletion-requests'),
    
    # ========== SECURITY ==========
    path('login-history/', LoginHistoryView.as_view(), name='login-history'),
    path('delete-account/', DeleteAccountView.as_view(), name='delete-account'),
    
    # ========== GDPR/HIPAA COMPLIANCE ==========
    path('data-export/', DataExportView.as_view(), name='data-export'),
    path('data-deletion/', DataDeletionView.as_view(), name='data-deletion'),
    path('consent-history/', ConsentHistoryView.as_view(), name='consent-history'),
    path('consent/', ConsentActionView.as_view(), name='consent-action'),
] + router.urls
