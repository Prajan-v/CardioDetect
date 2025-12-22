"""
Frontend views for template rendering.
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .models import Prediction, MedicalDocument


def landing_page(request):
    """Landing page."""
    return render(request, 'index.html')


@login_required
def dashboard(request):
    """User dashboard."""
    predictions = Prediction.objects.filter(user=request.user)[:10]
    
    # Stats
    total_predictions = predictions.count()
    high_risk_count = Prediction.objects.filter(
        user=request.user, risk_category='HIGH'
    ).count()
    
    context = {
        'predictions': predictions,
        'total_predictions': total_predictions,
        'high_risk_count': high_risk_count,
    }
    return render(request, 'dashboard.html', context)


@login_required
def predict_page(request):
    """Prediction options page."""
    return render(request, 'predict.html')


@login_required
def manual_input(request):
    """Manual input form page."""
    return render(request, 'manual_input.html')


@login_required
def ocr_upload(request):
    """OCR upload page."""
    return render(request, 'ocr_upload.html')


@login_required
def results_page(request, prediction_id):
    """Results display page."""
    prediction = get_object_or_404(
        Prediction, id=prediction_id, user=request.user
    )
    return render(request, 'results.html', {'prediction': prediction})


@login_required
def history_page(request):
    """Prediction history page."""
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'history.html', {'predictions': predictions})


def login_page(request):
    """Login page."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, 'Login successful!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid email or password')
    
    return render(request, 'login.html')


def register_page(request):
    """Registration page."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        from accounts.models import User
        
        email = request.POST.get('email')
        password = request.POST.get('password')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        role = request.POST.get('role', 'patient')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
        else:
            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                role=role
            )
            login(request, user)
            messages.success(request, 'Registration successful!')
            return redirect('dashboard')
    
    return render(request, 'register.html')


def logout_page(request):
    """Logout and redirect."""
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('home')
