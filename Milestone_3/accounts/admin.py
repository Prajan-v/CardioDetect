"""
Admin configuration for accounts app.
Simplified for Django 6.0 compatibility.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils.html import format_html
from .models import User, LoginHistory, RefreshTokenBlacklist
from .pending_changes import PendingProfileChange


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Custom admin for User model."""
    
    list_display = ['email', 'first_name', 'last_name', 'role', 'is_active', 'is_staff', 'created_at']
    list_filter = ['role', 'is_active', 'email_verified', 'is_staff', 'created_at']
    search_fields = ['email', 'first_name', 'last_name', 'phone']
    ordering = ['-created_at']
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal Info', {
            'fields': (
                'username', 'first_name', 'last_name', 'phone',
                'date_of_birth', 'gender', 'profile_picture', 'bio'
            )
        }),
        ('Address', {
            'fields': ('address', 'city', 'country'),
            'classes': ('collapse',)
        }),
        ('Role & Permissions', {
            'fields': ('role', 'is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')
        }),
        ('Doctor Info', {
            'fields': ('license_number', 'specialization', 'hospital'),
            'classes': ('collapse',)
        }),
        ('Email Verification', {
            'fields': ('email_verified', 'email_verification_token', 'email_verification_sent_at'),
            'classes': ('collapse',)
        }),
        ('Password Reset', {
            'fields': ('password_reset_token', 'password_reset_sent_at', 'password_changed_at'),
            'classes': ('collapse',)
        }),
        ('Security', {
            'fields': (
                'failed_login_attempts', 'locked_until',
                'last_login', 'last_login_ip', 'last_login_user_agent'
            ),
            'classes': ('collapse',)
        }),
        ('Consent', {
            'fields': (
                'terms_accepted', 'terms_accepted_at',
                'privacy_accepted', 'privacy_accepted_at',
                'marketing_consent'
            ),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': (
                'email', 'username', 'password1', 'password2',
                'first_name', 'last_name', 'role'
            ),
        }),
    )
    
    readonly_fields = [
        'created_at', 'updated_at', 'last_login', 'last_login_ip',
        'password_changed_at', 'terms_accepted_at', 'privacy_accepted_at'
    ]
    
    # Admin Actions
    actions = ['activate_users', 'deactivate_users', 'unlock_accounts', 'verify_emails', 'make_doctor', 'make_patient']
    
    @admin.action(description='Activate selected users')
    def activate_users(self, request, queryset):
        count = queryset.update(is_active=True)
        self.message_user(request, f'{count} user(s) activated.')
    
    @admin.action(description='Deactivate selected users')
    def deactivate_users(self, request, queryset):
        count = queryset.update(is_active=False)
        self.message_user(request, f'{count} user(s) deactivated.')
    
    @admin.action(description='Unlock selected accounts')
    def unlock_accounts(self, request, queryset):
        from .email_service import send_account_unlocked_email
        count = 0
        for user in queryset.filter(locked_until__isnull=False):
            user.failed_login_attempts = 0
            user.locked_until = None
            user.save(update_fields=['failed_login_attempts', 'locked_until'])
            send_account_unlocked_email(user)
            count += 1
        self.message_user(request, f'{count} account(s) unlocked and notified.')
    
    @admin.action(description='Mark emails as verified')
    def verify_emails(self, request, queryset):
        count = queryset.update(email_verified=True)
        self.message_user(request, f'{count} email(s) verified.')
    
    @admin.action(description='Change role to Doctor')
    def make_doctor(self, request, queryset):
        count = queryset.update(role='doctor')
        self.message_user(request, f'{count} user(s) changed to Doctor role.')
    
    @admin.action(description='Change role to Patient')
    def make_patient(self, request, queryset):
        count = queryset.update(role='patient')
        self.message_user(request, f'{count} user(s) changed to Patient role.')


@admin.register(LoginHistory)
class LoginHistoryAdmin(admin.ModelAdmin):
    """Admin for login history."""
    
    list_display = ['get_user_email', 'timestamp', 'success', 'ip_address', 'failure_reason']
    list_filter = ['success', 'timestamp']
    search_fields = ['user__email', 'ip_address']
    ordering = ['-timestamp']
    readonly_fields = ['user', 'timestamp', 'ip_address', 'user_agent', 'location', 'success', 'failure_reason']
    
    @admin.display(description='User')
    def get_user_email(self, obj):
        if obj.user:
            return obj.user.email
        return '-'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(RefreshTokenBlacklist)
class RefreshTokenBlacklistAdmin(admin.ModelAdmin):
    """Admin for token blacklist."""
    
    list_display = ['get_user_email', 'reason', 'blacklisted_at']
    list_filter = ['reason', 'blacklisted_at']
    search_fields = ['user__email']
    ordering = ['-blacklisted_at']
    readonly_fields = ['token', 'user', 'blacklisted_at', 'reason']
    
    @admin.display(description='User')
    def get_user_email(self, obj):
        if obj.user:
            return obj.user.email
        return '-'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False


@admin.register(PendingProfileChange)
class PendingProfileChangeAdmin(admin.ModelAdmin):
    """Admin for pending profile change approvals."""
    
    list_display = ['get_user_email', 'field_name', 'old_value', 'new_value', 'status', 'created_at']
    list_filter = ['status', 'field_name', 'created_at']
    search_fields = ['user__email', 'user__first_name', 'user__last_name', 'field_name']
    ordering = ['-created_at']
    readonly_fields = ['user', 'field_name', 'old_value', 'new_value', 'reason', 'created_at', 'reviewed_at', 'reviewed_by']
    
    @admin.display(description='User')
    def get_user_email(self, obj):
        if obj.user:
            return f"{obj.user.email} ({obj.user.get_full_name()})"
        return '-'
    
    # Admin Actions
    actions = ['approve_changes', 'reject_changes']
    
    @admin.action(description='Approve selected profile changes')
    def approve_changes(self, request, queryset):
        from django.utils import timezone
        count = 0
        for change in queryset.filter(status='pending'):
            change.approve(request.user, 'Approved via admin panel')
            count += 1
        self.message_user(request, f'{count} change(s) approved and applied.')
    
    @admin.action(description='Reject selected profile changes')
    def reject_changes(self, request, queryset):
        count = 0
        for change in queryset.filter(status='pending'):
            change.reject(request.user, 'Rejected via admin panel')
            count += 1
        self.message_user(request, f'{count} change(s) rejected.')
