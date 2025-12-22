"""
Pending Profile Changes Model for Admin Approval System.
"""

from django.db import models
from django.utils import timezone


class PendingProfileChange(models.Model):
    """
    Stores profile change requests that need admin approval.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending Review'
        APPROVED = 'approved', 'Approved'
        REJECTED = 'rejected', 'Rejected'
    
    user = models.ForeignKey(
        'accounts.User',
        on_delete=models.CASCADE,
        related_name='pending_changes'
    )
    
    # What fields are being changed
    field_name = models.CharField(max_length=50)
    old_value = models.TextField(blank=True)
    new_value = models.TextField()
    
    # Change request details
    reason = models.TextField(blank=True, help_text="User's reason for change")
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.PENDING
    )
    
    # Admin response
    reviewed_by = models.ForeignKey(
        'accounts.User',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_changes'
    )
    review_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Pending Profile Change'
        verbose_name_plural = 'Pending Profile Changes'
    
    def __str__(self):
        return f"{self.user.email}: {self.field_name} ({self.status})"
    
    def approve(self, admin_user, notes=''):
        """Approve the change and apply it to user profile."""
        self.status = self.Status.APPROVED
        self.reviewed_by = admin_user
        self.review_notes = notes
        self.reviewed_at = timezone.now()
        self.save()
        
        # Apply the change to user
        if hasattr(self.user, self.field_name):
            setattr(self.user, self.field_name, self.new_value)
            self.user.save(update_fields=[self.field_name])
        
        # Send approval email notification
        from .email_service import send_change_approved_email
        send_change_approved_email(self)
    
    def reject(self, admin_user, notes=''):
        """Reject the change request."""
        self.status = self.Status.REJECTED
        self.reviewed_by = admin_user
        self.review_notes = notes
        self.reviewed_at = timezone.now()
        self.save()
        
        # Send rejection email notification
        from .email_service import send_change_rejected_email
        send_change_rejected_email(self)
