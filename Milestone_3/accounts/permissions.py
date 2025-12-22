"""
Role-Based Permissions for CardioDetect.
Custom DRF permission classes for patient/doctor/admin access control.
"""

from rest_framework import permissions


class IsPatient(permissions.BasePermission):
    """Allow access only to patients."""
    message = "Only patients can access this resource."
    
    def has_permission(self, request, view):
        return (
            request.user and 
            request.user.is_authenticated and 
            request.user.role == 'patient'
        )


class IsDoctor(permissions.BasePermission):
    """Allow access only to doctors."""
    message = "Only doctors can access this resource."
    
    def has_permission(self, request, view):
        return (
            request.user and 
            request.user.is_authenticated and 
            request.user.role == 'doctor'
        )


class IsDoctorOrAdmin(permissions.BasePermission):
    """Allow access to doctors and admins."""
    message = "Only doctors and administrators can access this resource."
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return request.user.role in ('doctor', 'admin') or request.user.is_staff


class IsOwnerOrDoctor(permissions.BasePermission):
    """
    Allow access to:
    - The user who owns the resource
    - Doctors assigned to the patient
    - Admins
    """
    message = "You don't have permission to access this resource."
    
    def has_object_permission(self, request, view, obj):
        # Admin can access anything
        if request.user.is_staff or request.user.role == 'admin':
            return True
        
        # Owner can access their own data
        if hasattr(obj, 'user') and obj.user == request.user:
            return True
        if hasattr(obj, 'patient') and obj.patient == request.user:
            return True
        
        # Doctor can access their patients' data
        if request.user.role == 'doctor':
            patient = getattr(obj, 'user', None) or getattr(obj, 'patient', None)
            if patient:
                from predictions.models import DoctorPatient
                return DoctorPatient.objects.filter(
                    doctor=request.user,
                    patient=patient,
                    status='active'
                ).exists()
        
        return False


class CanViewPatientData(permissions.BasePermission):
    """
    Permission to view patient data.
    - Patients can view their own data
    - Doctors can view assigned patients' data
    - Admins can view all
    """
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        return True
    
    def has_object_permission(self, request, view, obj):
        user = request.user
        
        # Admin access
        if user.is_staff or user.role == 'admin':
            return True
        
        # Get the patient from the object
        patient = None
        if hasattr(obj, 'user'):
            patient = obj.user
        elif hasattr(obj, 'patient'):
            patient = obj.patient
        
        if not patient:
            return False
        
        # Patient accessing own data
        if user == patient:
            return True
        
        # Doctor accessing assigned patient's data
        if user.role == 'doctor':
            from predictions.models import DoctorPatient
            return DoctorPatient.objects.filter(
                doctor=user,
                patient=patient,
                status='active'
            ).exists()
        
        return False


class IsEmailVerified(permissions.BasePermission):
    """Require email verification for sensitive actions."""
    message = "Please verify your email address to access this feature."
    
    def has_permission(self, request, view):
        return (
            request.user and 
            request.user.is_authenticated and 
            request.user.email_verified
        )
