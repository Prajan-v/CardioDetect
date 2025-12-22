"""
URL configuration for frontend pages.
"""

from django.urls import path
from . import frontend_views

urlpatterns = [
    path('', frontend_views.landing_page, name='home'),
    path('dashboard/', frontend_views.dashboard, name='dashboard'),
    path('predict/', frontend_views.predict_page, name='predict'),
    path('predict/manual/', frontend_views.manual_input, name='manual_input'),
    path('predict/upload/', frontend_views.ocr_upload, name='ocr_upload'),
    path('results/<int:prediction_id>/', frontend_views.results_page, name='results'),
    path('history/', frontend_views.history_page, name='history'),
    
    # Auth pages
    path('accounts/login/', frontend_views.login_page, name='login_page'),
    path('accounts/register/', frontend_views.register_page, name='register_page'),
    path('accounts/logout/', frontend_views.logout_page, name='logout_page'),
]
