"""
Custom SMTP Email Backend with SSL Context Support.
Fixes macOS certificate verification issues by using certifi's certificate bundle.
"""

from django.core.mail.backends.smtp import EmailBackend as SMTPBackend
from django.conf import settings


class CertifiedEmailBackend(SMTPBackend):
    """
    SMTP email backend that uses certifi's SSL context for certificate verification.
    This fixes the SSL certificate verification errors on macOS.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the SSL context from settings if available
        if hasattr(settings, 'EMAIL_SSL_CONTEXT') and settings.EMAIL_SSL_CONTEXT:
            self.ssl_context = settings.EMAIL_SSL_CONTEXT
