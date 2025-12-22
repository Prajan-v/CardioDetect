"""
Rate Limiting Middleware for CardioDetect.
Protects authentication endpoints from brute force attacks.
"""

import time
from collections import defaultdict
from django.http import JsonResponse
from django.conf import settings


class RateLimitMiddleware:
    """
    Simple in-memory rate limiting for critical endpoints.
    For production, consider using Redis with django-ratelimit.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limits = defaultdict(list)  # IP -> [(timestamp, endpoint), ...]
        
        # Rate limit configuration - relaxed for development
        self.limits = {
            '/api/auth/login/': (20, 300),        # 20 attempts per 5 minutes
            '/api/auth/register/': (20, 3600),    # 20 per hour
            '/api/auth/password-reset/': (10, 3600),  # 10 per hour
            '/api/auth/resend-verification/': (10, 3600),  # 10 per hour
        }

    
    def __call__(self, request):
        # Get client IP
        ip = self._get_client_ip(request)
        path = request.path
        
        # Check if this endpoint is rate limited
        if path in self.limits and request.method == 'POST':
            max_requests, window_seconds = self.limits[path]
            
            if self._is_rate_limited(ip, path, max_requests, window_seconds):
                return JsonResponse({
                    'status': 'error',
                    'error': 'Too many requests. Please try again later.',
                    'code': 'rate_limited',
                    'retry_after': window_seconds
                }, status=429)
            
            # Record this request
            self._record_request(ip, path)
        
        return self.get_response(request)
    
    def _get_client_ip(self, request):
        """Get the real client IP, considering proxies."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', 'unknown')
    
    def _is_rate_limited(self, ip, path, max_requests, window_seconds):
        """Check if IP has exceeded rate limit for this endpoint."""
        now = time.time()
        key = f"{ip}:{path}"
        
        # Clean old entries
        self.rate_limits[key] = [
            ts for ts in self.rate_limits[key] 
            if now - ts < window_seconds
        ]
        
        return len(self.rate_limits[key]) >= max_requests
    
    def _record_request(self, ip, path):
        """Record a request for rate limiting."""
        key = f"{ip}:{path}"
        self.rate_limits[key].append(time.time())


class SecurityHeadersMiddleware:
    """
    Add security headers to all responses.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Security headers
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Content Security Policy (adjust as needed)
        if not getattr(settings, 'DEBUG', False):
            response['Content-Security-Policy'] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://fonts.gstatic.com; "
                "connect-src 'self' https:; "
            )
        
        return response


class RequestLoggingMiddleware:
    """
    Log all API requests for monitoring and debugging.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        import logging
        logger = logging.getLogger('django.request')
        
        # Skip static files and health checks
        if not request.path.startswith('/static/') and request.path != '/api/health/':
            start_time = time.time()
            response = self.get_response(request)
            duration = (time.time() - start_time) * 1000  # ms
            
            # Log the request
            user = getattr(request, 'user', None)
            user_info = user.email if user and user.is_authenticated else 'anonymous'
            
            logger.info(
                f"{request.method} {request.path} - {response.status_code} "
                f"- {duration:.0f}ms - {user_info}"
            )
            
            return response
        
        return self.get_response(request)
