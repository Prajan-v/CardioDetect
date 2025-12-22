"""
Barcode Verification API for CardioDetect.
Handles device authorization and barcode scanning.
"""

from django.db import models
from django.conf import settings
import uuid
import hmac
import hashlib
import time


class AuthorizedDevice(models.Model):
    """Authorized verification terminal devices."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    device_fingerprint = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100, blank=True)
    authorized_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True
    )
    is_active = models.BooleanField(default=True)
    authorized_at = models.DateTimeField(auto_now_add=True)
    last_scan_at = models.DateTimeField(null=True, blank=True)
    scan_count = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-authorized_at']
    
    def __str__(self):
        return f"{self.device_fingerprint} - {'Active' if self.is_active else 'Revoked'}"


class BarcodeScan(models.Model):
    """Audit log for barcode scans."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    device = models.ForeignKey(
        AuthorizedDevice, 
        on_delete=models.SET_NULL, 
        null=True, 
        related_name='scans'
    )
    patient_id = models.CharField(max_length=50)
    patient_name = models.CharField(max_length=100)
    risk_level = models.CharField(max_length=20)
    probability = models.FloatField(null=True)
    accession_number = models.CharField(max_length=50, blank=True)
    is_valid = models.BooleanField(default=True)
    scanned_at = models.DateTimeField(auto_now_add=True)
    barcode_version = models.IntegerField(default=3)
    
    class Meta:
        ordering = ['-scanned_at']
    
    def __str__(self):
        return f"{self.patient_name} - {self.risk_level} @ {self.scanned_at}"
