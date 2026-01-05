"""
Django settings for CardioDetect Milestone 3.
AI-Powered Cardiovascular Disease Risk Prediction System
"""

from pathlib import Path
from datetime import timedelta
import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use os.environ directly

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Add parent directories to path for importing Milestone_2 modules
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'Milestone_2'))
sys.path.insert(0, str(PROJECT_ROOT / 'Milestone_2' / 'pipeline'))

# =============================================================================
# SECURITY SETTINGS - Load from environment
# =============================================================================

# SECRET_KEY: Load from environment, fallback to insecure key for development only
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-CHANGE-THIS-IN-PRODUCTION'  # Development fallback
)

# DEBUG: Load from environment, default True for development
# Set DEBUG=False or 0 in production environment
DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 'yes')

# ALLOWED_HOSTS: Load from environment as comma-separated
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')



# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party apps
    'rest_framework',
    'rest_framework_simplejwt',
    'corsheaders',
    
    # Local apps
    'accounts',
    'predictions',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # Custom middleware
    'cardiodetect.middleware.RateLimitMiddleware',
    'cardiodetect.middleware.SecurityHeadersMiddleware',
]


ROOT_URLCONF = 'cardiodetect.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'cardiodetect.wsgi.application'


# Database - PostgreSQL Configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.environ.get('DB_NAME', 'cardiodetect_db'),
        'USER': os.environ.get('DB_USER', 'cardiodetect_user'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST', 'localhost'),
        'PORT': os.environ.get('DB_PORT', '5432'),
        'CONN_MAX_AGE': 600,  # Connection pooling for performance
        'OPTIONS': {
            'connect_timeout': 10,
        }
    }
}

# =============================================================================
# REDIS CACHE CONFIGURATION
# =============================================================================
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'KEY_PREFIX': 'cardiodetect',
    }
}

# Use Redis for session storage (optional but recommended)
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'


# Password validation - Enhanced security
AUTH_PASSWORD_VALIDATORS = [
    # Checks password isn't too similar to user attributes (email, name)
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
        'OPTIONS': {'max_similarity': 0.7}
    },
    # Minimum 8 characters
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {'min_length': 8}
    },
    # Checks against 20,000 common passwords (password123, qwerty, etc.)
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    # Prevents all-numeric passwords
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True


# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files (uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'


# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# Custom User Model
AUTH_USER_MODEL = 'accounts.User'


# Django REST Framework Configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    # Rate limiting (generous limits for development)
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '1000/hour',       # Anonymous: 1000 requests/hour
        'user': '5000/hour',       # Authenticated: 5000 requests/hour  
        'predictions': '500/hour', # Prediction endpoint specific
    }
}


# JWT Settings
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=24),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
}

# CORS Settings - Load from environment for flexibility
# SECURITY: CORS_ALLOW_ALL_ORIGINS removed for production safety
_cors_origins = os.environ.get('CORS_ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:8000')
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in _cors_origins.split(',')]
CORS_ALLOW_CREDENTIALS = True

# Allow all CORS methods and headers for API compatibility
CORS_ALLOW_METHODS = [
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
]

CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]

# Enable CORS_ALLOW_ALL_ORIGINS for development (DEBUG mode or localhost)
# This is safe for development and ensures CORS doesn't block local testing
CORS_ALLOW_ALL_ORIGINS = True  # Enable for development


# =============================================================================
# FILE UPLOAD SECURITY
# =============================================================================
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB

# Allowed file types for OCR upload (validated by magic bytes)
ALLOWED_UPLOAD_TYPES = {
    'application/pdf': ['.pdf'],
    'image/png': ['.png'],
    'image/jpeg': ['.jpg', '.jpeg'],
}

# Magic bytes for file type validation
MAGIC_BYTES = {
    b'%PDF': 'application/pdf',
    b'\x89PNG': 'image/png',
    b'\xff\xd8\xff': 'image/jpeg',
}


# ML Model Paths (from Milestone 2)
ML_MODELS_DIR = PROJECT_ROOT / 'Milestone_2' / 'models' / 'Final_models'
OCR_PIPELINE_DIR = PROJECT_ROOT / 'Milestone_2' / 'pipeline'


# Login/Logout URLs
LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/dashboard/'
LOGOUT_REDIRECT_URL = '/'


# =============================================================================
# EMAIL CONFIGURATION
# =============================================================================
# Fix SSL certificate verification on macOS BEFORE importing email modules
import ssl
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    # Create custom SSL context with certifi certificates for Django email
    EMAIL_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    EMAIL_SSL_CONTEXT = None  # certifi not installed, use system defaults

# Gmail SMTP configuration for real email sending
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', 'cardiodetect.care@gmail.com')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
DEFAULT_FROM_EMAIL = 'CardioDetect <cardiodetect.care@gmail.com>'

# Set the SSL context for email backend (fixes macOS certificate issues)
if EMAIL_SSL_CONTEXT:
    EMAIL_SSL_CERTFILE = None  # Not needed when using context
    EMAIL_SSL_KEYFILE = None
    # Django 4.2+ uses this for TLS connections
    EMAIL_USE_SSL = False  # We use TLS, not SSL

# Use console backend for development when EMAIL_HOST_PASSWORD is not set
# This prints emails to the terminal instead of sending them
if EMAIL_HOST_PASSWORD:
    # Use custom backend with SSL context support (fixes macOS certificate issues)
    EMAIL_BACKEND = 'cardiodetect.email_backend.CertifiedEmailBackend'
    print("[EMAIL] Using CertifiedEmailBackend - emails will be sent via Gmail")
else:
    EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
    print("[EMAIL] WARNING: EMAIL_HOST_PASSWORD not set - using console backend (emails printed to terminal)")




# =============================================================================
# PRODUCTION SECURITY SETTINGS
# =============================================================================
# These settings are applied when DEBUG=False

if not DEBUG:
    # HTTPS Settings
    SECURE_SSL_REDIRECT = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    
    # HSTS (HTTP Strict Transport Security)
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    
    # Secure Cookies
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    CSRF_COOKIE_HTTPONLY = True
    
    # Content Security
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_BROWSER_XSS_FILTER = True
    X_FRAME_OPTIONS = 'DENY'
    
    # CSRF
    CSRF_TRUSTED_ORIGINS = os.environ.get(
        'CSRF_TRUSTED_ORIGINS', 
        'https://localhost'
    ).split(',')


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '[{levelname}] {asctime} - {message}',
            'style': '{',
        },
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs' / 'django.log',
            'formatter': 'verbose',
        },
        'mail_admins': {
            'level': 'ERROR',
            'filters': ['require_debug_false'],
            'class': 'django.utils.log.AdminEmailHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'django.request': {
            'handlers': ['console', 'mail_admins'],
            'level': 'ERROR',
            'propagate': False,
        },
        'accounts': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
        'predictions': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# Create logs directory if it doesn't exist
(BASE_DIR / 'logs').mkdir(exist_ok=True)


# =============================================================================
# RATE LIMITING (using django-ratelimit or custom middleware)
# =============================================================================
RATE_LIMIT_LOGIN = '5/5m'      # 5 attempts per 5 minutes
RATE_LIMIT_REGISTER = '3/h'     # 3 registrations per hour per IP
RATE_LIMIT_PASSWORD_RESET = '3/h'  # 3 reset requests per hour


# =============================================================================
# API DOCUMENTATION
# =============================================================================
API_TITLE = 'Cardio Detect API'
API_VERSION = 'v1.0.0'
API_DESCRIPTION = """
AI-Powered Cardiovascular Disease Risk Prediction System

## Features
- 🩺 Heart Disease Detection (91.45% accuracy)
- 📊 10-Year Risk Prediction (91.63% accuracy)
- 📄 OCR Medical Report Processing
- 🔐 Secure Authentication with JWT
- 👨‍⚕️ Role-Based Access (Patient/Doctor/Admin)
"""


# =============================================================================
# FRONTEND URL (for email links)
# =============================================================================
FRONTEND_URL = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
