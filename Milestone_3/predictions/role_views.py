"""
Doctor-specific views for CardioDetect.
Implements doctor dashboard, patient management, and admin features.
"""

from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.pagination import PageNumberPagination

from accounts.permissions import IsDoctor, IsDoctorOrAdmin
from predictions.models import Prediction, DoctorPatient


class StandardPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 50


class DoctorDashboardView(APIView):
    """
    Doctor dashboard with patient overview and statistics.
    
    GET /api/doctor/dashboard/
    """
    permission_classes = [IsAuthenticated, IsDoctor]
    
    def get(self, request):
        doctor = request.user
        
        # Get assigned patients
        patient_relations = DoctorPatient.objects.filter(
            doctor=doctor,
            status='active'
        ).select_related('patient')
        
        patient_ids = [rel.patient_id for rel in patient_relations]
        
        # Get statistics
        today = timezone.now().date()
        week_ago = today - timedelta(days=7)
        
        # Patient predictions
        patient_predictions = Prediction.objects.filter(user_id__in=patient_ids)
        recent_predictions = patient_predictions.filter(created_at__date__gte=week_ago)
        
        # Risk distribution
        risk_counts = patient_predictions.values('risk_category').annotate(count=Count('id'))
        risk_distribution = {item['risk_category']: item['count'] for item in risk_counts}
        
        # Build patient list with latest prediction
        patients_data = []
        for rel in patient_relations[:10]:  # Top 10
            patient = rel.patient
            latest_pred = Prediction.objects.filter(user=patient).first()
            patients_data.append({
                'id': str(patient.id),
                'name': patient.get_full_name() or patient.email,
                'email': patient.email,
                'assigned_at': rel.created_at.isoformat(),
                'latest_risk': latest_pred.risk_category if latest_pred else None,
                'latest_prediction': latest_pred.created_at.isoformat() if latest_pred else None,
            })
        
        return Response({
            'doctor': {
                'id': str(doctor.id),
                'name': doctor.get_full_name(),
                'email': doctor.email,
                'specialization': doctor.specialization or 'Cardiology',
                'hospital': doctor.hospital or 'CardioDetect Clinic',
            },
            'stats': {
                'total_patients': len(patient_ids),
                'total_predictions': patient_predictions.count(),
                'predictions_this_week': recent_predictions.count(),
                'high_risk_patients': patient_predictions.filter(risk_category='HIGH').values('user').distinct().count(),
            },
            'risk_distribution': risk_distribution,
            'patients': patients_data,  # Frontend expects 'patients'
            'recent_patients': patients_data,  # Keep for backward compat
            'actions': [
                {'label': 'Add Patient', 'url': '/doctor/patients/add'},
                {'label': 'View All Patients', 'url': '/doctor/patients'},
                {'label': 'Generate Report', 'url': '/doctor/reports'},
            ]
        })


class DoctorPatientsView(APIView):
    """
    List and manage doctor's patients.
    
    GET /api/doctor/patients/
    POST /api/doctor/patients/ - Assign patient by email
    """
    permission_classes = [IsAuthenticated, IsDoctor]
    
    def get(self, request):
        doctor = request.user
        
        # Filter parameters
        search = request.query_params.get('search', '')
        risk_filter = request.query_params.get('risk', '')
        
        # Get patient relations
        relations = DoctorPatient.objects.filter(
            doctor=doctor,
            status='active'
        ).select_related('patient')
        
        # Search by name or email
        if search:
            relations = relations.filter(
                Q(patient__first_name__icontains=search) |
                Q(patient__last_name__icontains=search) |
                Q(patient__email__icontains=search)
            )
        
        patients_data = []
        for rel in relations:
            patient = rel.patient
            latest_pred = Prediction.objects.filter(user=patient).first()
            
            # Apply risk filter
            if risk_filter and latest_pred and latest_pred.risk_category != risk_filter.upper():
                continue
            
            pred_count = Prediction.objects.filter(user=patient).count()
            
            patients_data.append({
                'id': str(patient.id),
                'name': patient.get_full_name() or patient.email,
                'email': patient.email,
                'phone': patient.phone,
                'age': patient.age,
                'gender': patient.get_gender_display() if patient.gender else None,
                'assigned_at': rel.created_at.isoformat(),
                'prediction_count': pred_count,
                'latest_risk': latest_pred.risk_category if latest_pred else None,
                'latest_prediction_date': latest_pred.created_at.isoformat() if latest_pred else None,
            })
        
        return Response({
            'count': len(patients_data),
            'patients': patients_data
        })
    
    def post(self, request):
        """Assign a new patient by email."""
        from accounts.models import User
        
        email = request.data.get('email', '').lower()
        
        if not email:
            return Response(
                {'error': 'Patient email is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            patient = User.objects.get(email__iexact=email, role='patient')
        except User.DoesNotExist:
            return Response(
                {'error': 'No patient found with this email'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check if already assigned
        if DoctorPatient.objects.filter(
            doctor=request.user,
            patient=patient,
            status='active'
        ).exists():
            return Response(
                {'error': 'Patient is already assigned to you'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create assignment
        DoctorPatient.objects.create(
            doctor=request.user,
            patient=patient,
            status='active'
        )
        
        return Response({
            'status': 'success',
            'message': f'Patient {patient.get_full_name() or patient.email} assigned successfully',
            'patient': {
                'id': str(patient.id),
                'name': patient.get_full_name() or patient.email,
                'email': patient.email,
            }
        }, status=status.HTTP_201_CREATED)


class DoctorPatientDetailView(APIView):
    """
    View patient details and predictions.
    
    GET /api/doctor/patients/<patient_id>/
    DELETE /api/doctor/patients/<patient_id>/ - Remove patient
    """
    permission_classes = [IsAuthenticated, IsDoctor]
    
    def get(self, request, patient_id):
        from accounts.models import User
        
        # Verify doctor has access to this patient
        try:
            relation = DoctorPatient.objects.get(
                doctor=request.user,
                patient_id=patient_id,
                status='active'
            )
        except DoctorPatient.DoesNotExist:
            return Response(
                {'error': 'Patient not found or not assigned to you'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        patient = relation.patient
        predictions = Prediction.objects.filter(user=patient).order_by('-created_at')[:20]
        
        return Response({
            'patient': {
                'id': str(patient.id),
                'name': patient.get_full_name(),
                'email': patient.email,
                'phone': patient.phone,
                'age': patient.age,
                'gender': patient.get_gender_display() if patient.gender else None,
                'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else None,
                'address': patient.address,
                'city': patient.city,
                'country': patient.country,
                'assigned_at': relation.created_at.isoformat(),
            },
            'predictions': [
                {
                    'id': str(p.id),
                    'risk_category': p.risk_category,
                    'risk_percentage': p.risk_percentage,
                    'risk_factors': p.risk_factors,
                    'recommendations': p.recommendations,
                    'input_method': p.input_method,
                    'created_at': p.created_at.isoformat(),
                }
                for p in predictions
            ],
            'stats': {
                'total_predictions': Prediction.objects.filter(user=patient).count(),
                'high_risk_count': Prediction.objects.filter(user=patient, risk_category='HIGH').count(),
            }
        })
    
    def delete(self, request, patient_id):
        """Remove patient from doctor's list."""
        try:
            relation = DoctorPatient.objects.get(
                doctor=request.user,
                patient_id=patient_id,
                status='active'
            )
            relation.status = 'removed'
            relation.save()
            
            return Response({
                'status': 'success',
                'message': 'Patient removed from your list'
            })
        except DoctorPatient.DoesNotExist:
            return Response(
                {'error': 'Patient not found'},
                status=status.HTTP_404_NOT_FOUND
            )


class PatientDashboardView(APIView):
    """
    Patient dashboard with their predictions and health summary.
    
    GET /api/patient/dashboard/
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Get predictions
        predictions = Prediction.objects.filter(user=user)
        latest = predictions.first()
        
        # Time filters
        today = timezone.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Risk distribution
        risk_counts = predictions.values('risk_category').annotate(count=Count('id'))
        risk_distribution = {item['risk_category']: item['count'] for item in risk_counts}
        
        # Get assigned doctor
        doctor_relation = DoctorPatient.objects.filter(
            patient=user,
            status='active'
        ).select_related('doctor').first()
        
        doctor_info = None
        if doctor_relation:
            doctor = doctor_relation.doctor
            doctor_info = {
                'id': str(doctor.id),
                'name': doctor.get_full_name(),
                'email': doctor.email,
                'specialization': doctor.specialization,
                'hospital': doctor.hospital,
            }
        
        # Recent predictions
        recent = predictions[:5]
        recent_data = [
            {
                'id': str(p.id),
                'risk_category': p.risk_category,
                'risk_percentage': p.risk_percentage,
                'created_at': p.created_at.isoformat(),
            }
            for p in recent
        ]
        
        return Response({
            'user': {
                'id': str(user.id),
                'name': user.get_full_name(),
                'email': user.email,
                'email_verified': user.email_verified,
            },
            'stats': {
                'total_predictions': predictions.count(),
                'this_week': predictions.filter(created_at__date__gte=week_ago).count(),
                'this_month': predictions.filter(created_at__date__gte=month_ago).count(),
            },
            'latest_prediction': {
                'risk_category': latest.risk_category,
                'risk_percentage': latest.risk_percentage,
                'risk_factors': latest.risk_factors,
                'recommendations': latest.recommendations,
                'created_at': latest.created_at.isoformat(),
            } if latest else None,
            'risk_distribution': risk_distribution,
            'recent_predictions': recent_data,
            'assigned_doctor': doctor_info,
            'actions': [
                {'label': 'New Prediction', 'url': '/dashboard/predict'},
                {'label': 'Upload Report', 'url': '/dashboard/upload'},
                {'label': 'View History', 'url': '/dashboard/history'},
            ]
        })
