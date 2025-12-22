"""
Main URL configuration for CardioDetect.
Complete routing for API and frontend pages.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    # ========== ADMIN ==========
    path('admin/', admin.site.urls),
    
    # ========== JWT AUTH ==========
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # ========== API ENDPOINTS ==========
    # Authentication (register, login, profile)
    path('api/auth/', include('accounts.urls')),
    
    # Predictions API
    path('api/', include('predictions.urls')),
    
    # ========== FRONTEND PAGES ==========
    path('', include('predictions.frontend_urls')),
]

# Serve media and static files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    if settings.STATICFILES_DIRS:
        urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
