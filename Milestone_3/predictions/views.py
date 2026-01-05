"""
Enhanced Views for predictions app - Complete API endpoints.
"""

import os
import sys
import io
from datetime import datetime, timedelta
from django.conf import settings
from django.http import HttpResponse, FileResponse
from django.db.models import Count, Avg, Q
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination

from .models import (
    Prediction, MedicalDocument, UnitPreference, 
    DoctorPatient, PredictionNote, AuditLog, SystemNotification
)
from .serializers import (
    PredictionSerializer, ManualInputSerializer, 
    OCRUploadSerializer, UnitConversionSerializer,
    MedicalDocumentSerializer, PredictionDetailSerializer,
    PredictionHistorySerializer, StatisticsSerializer,
    NotificationSerializer, PredictionNoteSerializer
)

# Import ML service
try:
    from services.ml_service import ml_service, unit_converter
    print(f"[views.py] ML service import: ml_service={ml_service is not None}, loaded={getattr(ml_service, 'loaded', False)}")
except Exception as e:
    print(f"[views.py] ERROR importing ml_service: {e}")
    import traceback
    traceback.print_exc()
    ml_service = None
    unit_converter = None


class StandardResultsPagination(PageNumberPagination):
    """Standard pagination for list views."""
    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100


def log_action(user, action, resource_type, resource_id='', details=None, request=None):
    """Helper to create audit log entries."""
    AuditLog.objects.create(
        user=user,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id),
        details=details or {},
        ip_address=request.META.get('REMOTE_ADDR') if request else None,
        user_agent=request.META.get('HTTP_USER_AGENT', '')[:500] if request else ''
    )


class ManualPredictionView(APIView):
    """Manual input prediction endpoint."""
    permission_classes = []  # Public for demo mode

    def post(self, request):
        serializer = ManualInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        data = serializer.validated_data
        start_time = datetime.now()
        
        try:
            # Create prediction record
            prediction = Prediction.objects.create(
                user=request.user,
                input_method=Prediction.InputMethod.MANUAL,
                input_data=data,
                age=data.get('age'),
                sex=data.get('sex'),
                systolic_bp=data.get('systolic_bp'),
                diastolic_bp=data.get('diastolic_bp'),
                cholesterol=data.get('cholesterol'),
                hdl=data.get('hdl'),
                glucose=data.get('glucose'),
                bmi=data.get('bmi'),
                heart_rate=data.get('heart_rate'),
                smoking=data.get('smoking'),
                diabetes=data.get('diabetes'),
                bp_medication=data.get('bp_medication'),
                # Stress test fields if provided
                chest_pain_type=data.get('chest_pain_type'),
                max_heart_rate=data.get('max_heart_rate'),
                exercise_angina=data.get('exercise_angina'),
                st_depression=data.get('st_depression'),
                st_slope=data.get('st_slope'),
                major_vessels=data.get('major_vessels'),
                thalassemia=data.get('thalassemia'),
                resting_ecg=data.get('resting_ecg'),
            )
            
            # Run prediction using ML service
            if ml_service:
                result = ml_service.predict(data)
                
                # Update prediction with results
                prediction.risk_score = result.get('risk_score')
                prediction.risk_percentage = result.get('risk_percentage')
                prediction.risk_category = result.get('risk_category')
                prediction.prediction_confidence = result.get('confidence')
                prediction.detection_result = result.get('detection_result')
                prediction.detection_probability = result.get('detection_probability')
                prediction.clinical_score = result.get('clinical_score')
                prediction.clinical_max_score = result.get('clinical_max_score')
                prediction.recommendations = result.get('recommendations', '')
                prediction.risk_factors = result.get('risk_factors', [])
                prediction.model_used = data.get('model_used', 'prediction')
            else:
                # Fallback: Clinical assessment only
                prediction.risk_category = self._calculate_clinical_risk(data)
                prediction.risk_factors = self._get_risk_factors(data)
            
            # Calculate processing time
            prediction.processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            prediction.save()
            
            # Log action
            log_action(request.user, 'create', 'Prediction', prediction.id, 
                      {'method': 'manual', 'risk': prediction.risk_category}, request)
            
            # Create notification
            SystemNotification.objects.create(
                user=request.user,
                title='New Prediction Complete',
                message=f'Your heart disease risk assessment is complete. Risk Level: {prediction.risk_category}',
                notification_type=SystemNotification.NotificationType.PREDICTION,
                related_prediction=prediction
            )
            
            # Send high-risk email alerts if applicable
            from accounts.email_service import send_high_risk_alerts
            send_high_risk_alerts(prediction)
            
            return Response({
                'status': 'success',
                'prediction_id': str(prediction.id),
                'risk_category': prediction.risk_category,
                'risk_percentage': prediction.risk_percentage,
                'risk_factors': prediction.risk_factors,
                'recommendations': prediction.recommendations,
                'processing_time_ms': prediction.processing_time_ms
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response({
                'status': 'error',
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _calculate_clinical_risk(self, data):
        """Calculate clinical risk category from data."""
        score = 0
        age = data.get('age', 0)
        
        # Age scoring
        if age >= 75: score += 20
        elif age >= 65: score += 15
        elif age >= 55: score += 10
        elif age >= 45: score += 5
        
        # BP scoring
        sbp = data.get('systolic_bp', 120)
        if sbp >= 180: score += 20
        elif sbp >= 160: score += 15
        elif sbp >= 140: score += 10
        elif sbp >= 130: score += 5
        
        # Cholesterol scoring
        chol = data.get('cholesterol', 200)
        if chol >= 280: score += 15
        elif chol >= 240: score += 10
        elif chol >= 200: score += 5
        
        # Risk factors
        if data.get('smoking'): score += 15
        if data.get('diabetes'): score += 15
        if data.get('sex') == 1: score += 5  # Male
        
        # Categorize
        if score >= 50: return 'HIGH'
        elif score >= 25: return 'MODERATE'
        return 'LOW'
    
    def _get_risk_factors(self, data):
        """Get list of risk factors."""
        factors = []
        if data.get('age', 0) >= 55: factors.append(f"Age {data.get('age')}")
        if data.get('systolic_bp', 0) >= 140: factors.append("High Blood Pressure")
        if data.get('cholesterol', 0) >= 240: factors.append("High Cholesterol")
        if data.get('smoking'): factors.append("Current Smoker")
        if data.get('diabetes'): factors.append("Diabetes")
        if data.get('bmi', 0) >= 30: factors.append("Obesity")
        return factors


class OCRPredictionView(APIView):
    """OCR document upload and prediction endpoint."""
    permission_classes = []  # Public for demo mode
    parser_classes = (MultiPartParser, FormParser)

    def get_or_create_demo_user(self):
        """Get or create a demo user for unauthenticated predictions."""
        from django.contrib.auth import get_user_model
        User = get_user_model()
        demo_email = 'demo@cardiodetect.com'
        try:
            user = User.objects.get(email=demo_email)
        except User.DoesNotExist:
            user = User.objects.create_user(
                email=demo_email,
                password='DemoUser123!',
                first_name='Demo',
                last_name='User'
            )
        return user

    def post(self, request):
        if 'file' not in request.FILES:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Extension check
        if file_ext not in ['pdf', 'png', 'jpg', 'jpeg']:
            return Response({'error': 'Unsupported file type. Allowed: PDF, PNG, JPG'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # File size check (10MB max)
        if uploaded_file.size > 10 * 1024 * 1024:
            return Response({'error': 'File too large. Maximum size: 10MB'}, 
                          status=status.HTTP_400_BAD_REQUEST)
        
        # Magic bytes validation (prevent spoofed files)
        try:
            header = uploaded_file.read(8)
            uploaded_file.seek(0)  # Reset file pointer
            
            valid_type = False
            if header.startswith(b'%PDF') and file_ext == 'pdf':
                valid_type = True
            elif header.startswith(b'\x89PNG') and file_ext == 'png':
                valid_type = True
            elif header.startswith(b'\xff\xd8\xff') and file_ext in ['jpg', 'jpeg']:
                valid_type = True
            
            if not valid_type:
                return Response({'error': 'File content does not match extension'}, 
                              status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            pass  # Continue if magic byte check fails
        
        start_time = datetime.now()
        
        # Determine user - use authenticated user or demo user
        if request.user.is_authenticated:
            user = request.user
        else:
            user = self.get_or_create_demo_user()
        
        try:
            import tempfile
            import os
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            try:
                # Direct import of DualModelPipeline to bypass ml_service singleton issue
                import sys
                from pathlib import Path
                
                # Ensure Milestone_2 paths are in sys.path
                project_root = Path(__file__).resolve().parent.parent.parent
                milestone_2 = project_root / 'Milestone_2'
                if str(milestone_2) not in sys.path:
                    sys.path.insert(0, str(milestone_2))
                if str(milestone_2 / 'pipeline') not in sys.path:
                    sys.path.insert(0, str(milestone_2 / 'pipeline'))
                
                from integrated_pipeline import DualModelPipeline
                pipeline = DualModelPipeline(verbose=False)
                
                # Process document
                result = pipeline.process_document(tmp_path)
                
                # Extract OCR fields
                ocr_data = result.get('ocr', {})
                fields = ocr_data.get('fields', {})
                clinical_risk = result.get('clinical_risk', {})
                
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                
                # Save to database for history
                try:
                    prediction = Prediction.objects.create(
                        user=user,
                        input_method=Prediction.InputMethod.OCR,
                        input_data=fields,
                        risk_category=clinical_risk.get('level_code', 'moderate').upper(),
                        risk_percentage=clinical_risk.get('percentage', 0),
                        risk_factors=clinical_risk.get('risk_factors', []),
                        recommendations=clinical_risk.get('recommendation', ''),
                        processing_time_ms=processing_time
                    )
                    prediction_id = str(prediction.id)
                    
                    # Send high-risk email alerts if applicable
                    from accounts.email_service import send_high_risk_alerts
                    send_high_risk_alerts(prediction)
                    
                except Exception as save_err:
                    print(f"Warning: Could not save prediction: {save_err}")
                    prediction_id = None
                
                # Get clinical recommendations from pipeline result
                clinical_recommendations = result.get('clinical_recommendations', {})
                
                return Response({
                    'status': 'success',
                    'prediction_id': prediction_id,
                    'ocr_confidence': ocr_data.get('confidence', 0),
                    'extracted_fields': fields,
                    'num_fields': len(fields),
                    'quality': ocr_data.get('quality', 'HIGH' if len(fields) >= 10 else 'MEDIUM' if len(fields) >= 5 else 'LOW'),
                    'risk_category': clinical_risk.get('level_code', 'moderate').upper(),
                    'risk_percentage': clinical_risk.get('percentage', 0),
                    'risk_factors': clinical_risk.get('risk_factors', []),
                    'recommendations': clinical_risk.get('recommendation', ''),
                    'clinical_recommendations': clinical_recommendations,
                    'processing_time_ms': processing_time
                }, status=status.HTTP_200_OK)
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
                
        except Exception as e:
            return Response({'status': 'error', 'message': str(e)}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictionHistoryView(generics.ListAPIView):
    """List user's prediction history with pagination."""
    permission_classes = (IsAuthenticated,)
    serializer_class = PredictionHistorySerializer
    pagination_class = StandardResultsPagination
    
    def get_queryset(self):
        queryset = Prediction.objects.filter(user=self.request.user)
        
        # Filter by risk category
        risk = self.request.query_params.get('risk')
        if risk:
            queryset = queryset.filter(risk_category=risk.upper())
        
        # Filter by date range
        from_date = self.request.query_params.get('from_date')
        to_date = self.request.query_params.get('to_date')
        if from_date:
            queryset = queryset.filter(created_at__gte=from_date)
        if to_date:
            queryset = queryset.filter(created_at__lte=to_date)
        
        # Filter by input method
        method = self.request.query_params.get('method')
        if method:
            queryset = queryset.filter(input_method=method)
        
        return queryset.order_by('-created_at')


class PredictionDetailView(generics.RetrieveDestroyAPIView):
    """Get or delete a specific prediction."""
    permission_classes = (IsAuthenticated,)
    serializer_class = PredictionDetailSerializer
    lookup_field = 'id'
    
    def get_queryset(self):
        # Users can view their own predictions
        # Doctors can view their patients' predictions
        user = self.request.user
        if user.is_doctor:
            patient_ids = DoctorPatient.objects.filter(
                doctor=user, status='active'
            ).values_list('patient_id', flat=True)
            return Prediction.objects.filter(
                Q(user=user) | Q(user_id__in=patient_ids)
            )
        return Prediction.objects.filter(user=user)
    
    def retrieve(self, request, *args, **kwargs):
        response = super().retrieve(request, *args, **kwargs)
        # Log view action
        log_action(request.user, 'read', 'Prediction', kwargs.get('id'), request=request)
        return response


class PredictionStatisticsView(APIView):
    """Get user's prediction statistics and analytics."""
    permission_classes = (IsAuthenticated,)
    
    def get(self, request):
        user = request.user
        predictions = Prediction.objects.filter(user=user)
        
        # Time-based filters
        last_30_days = timezone.now() - timedelta(days=30)
        last_90_days = timezone.now() - timedelta(days=90)
        
        stats = {
            'total_predictions': predictions.count(),
            'last_30_days': predictions.filter(created_at__gte=last_30_days).count(),
            'last_90_days': predictions.filter(created_at__gte=last_90_days).count(),
            
            'risk_distribution': {
                'LOW': predictions.filter(risk_category='LOW').count(),
                'MODERATE': predictions.filter(risk_category='MODERATE').count(),
                'HIGH': predictions.filter(risk_category='HIGH').count(),
            },
            
            'input_methods': {
                'manual': predictions.filter(input_method='manual').count(),
                'ocr': predictions.filter(input_method='ocr').count(),
                'hybrid': predictions.filter(input_method='hybrid').count(),
            },
            
            'average_processing_time_ms': predictions.aggregate(
                avg=Avg('processing_time_ms')
            )['avg'] or 0,
            
            'latest_prediction': None,
            'trend': self._calculate_trend(predictions)
        }
        
        # Latest prediction summary
        latest = predictions.first()
        if latest:
            stats['latest_prediction'] = {
                'id': str(latest.id),
                'risk_category': latest.risk_category,
                'created_at': latest.created_at.isoformat(),
            }
        
        return Response(stats)
    
    def _calculate_trend(self, predictions):
        """Calculate risk trend over time."""
        recent = predictions.filter(
            created_at__gte=timezone.now() - timedelta(days=90)
        ).order_by('created_at')
        
        if recent.count() < 2:
            return 'insufficient_data'
        
        # Compare first half vs second half
        count = recent.count()
        first_half = recent[:count//2]
        second_half = recent[count//2:]
        
        def avg_risk(qs):
            risk_map = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
            values = [risk_map.get(p.risk_category, 2) for p in qs if p.risk_category]
            return sum(values) / len(values) if values else 2
        
        first_avg = avg_risk(first_half)
        second_avg = avg_risk(second_half)
        
        if second_avg > first_avg + 0.3:
            return 'worsening'
        elif second_avg < first_avg - 0.3:
            return 'improving'
        return 'stable'


class ExportPredictionView(APIView):
    """Export prediction as PDF report."""
    permission_classes = (IsAuthenticated,)
    
    def get(self, request, id):
        try:
            prediction = Prediction.objects.get(id=id, user=request.user)
        except Prediction.DoesNotExist:
            return Response({'error': 'Prediction not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Generate PDF report
        pdf_content = self._generate_pdf_report(prediction)
        
        # Log export action
        log_action(request.user, 'export', 'Prediction', id, request=request)
        
        response = HttpResponse(pdf_content, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="CardioDetect_Report_{str(id)[:8]}.pdf"'
        return response
    
    def _generate_pdf_report(self, prediction):
        """Generate PDF report content."""
        # Simple text-based report (can be enhanced with reportlab)
        content = f"""
        ================================================================================
                            CARDIODETECT - CARDIOVASCULAR RISK REPORT
        ================================================================================
        
        Report ID: {prediction.id}
        Date: {prediction.created_at.strftime('%Y-%m-%d %H:%M:%S')}
        Patient: {prediction.user.get_full_name() or prediction.user.email}
        
        ================================================================================
                                    RISK ASSESSMENT RESULTS
        ================================================================================
        
        Risk Category: {prediction.risk_category}
        Risk Score: {prediction.risk_percentage:.1f}% if prediction.risk_percentage else 'N/A'
        
        Contributing Risk Factors:
        {chr(10).join(['  • ' + f for f in (prediction.risk_factors or [])])}
        
        ================================================================================
                                    RECOMMENDATIONS
        ================================================================================
        
        {prediction.recommendations or 'Please consult with your healthcare provider.'}
        
        ================================================================================
        
        This report was generated by CardioDetect v2.0
        Detection Model Accuracy: 91.45% | Prediction Model Accuracy: 91.63%
        
        DISCLAIMER: This is an AI-generated assessment and should not replace 
        professional medical advice. Please consult with a healthcare provider.
        """
        return content.encode('utf-8')


class DocumentListView(generics.ListAPIView):
    """List user's uploaded documents."""
    permission_classes = (IsAuthenticated,)
    serializer_class = MedicalDocumentSerializer
    pagination_class = StandardResultsPagination
    
    def get_queryset(self):
        return MedicalDocument.objects.filter(user=self.request.user)


class DocumentDetailView(generics.RetrieveDestroyAPIView):
    """Get or delete a specific document."""
    permission_classes = (IsAuthenticated,)
    serializer_class = MedicalDocumentSerializer
    lookup_field = 'id'
    
    def get_queryset(self):
        return MedicalDocument.objects.filter(user=self.request.user)


class UnitConversionView(APIView):
    """Unit conversion utility endpoint."""
    permission_classes = (IsAuthenticated,)
    
    def post(self, request):
        value = request.data.get('value')
        from_unit = request.data.get('from_unit')
        to_unit = request.data.get('to_unit')
        
        if not all([value, from_unit, to_unit]):
            return Response({'error': 'Missing parameters'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            if unit_converter:
                result = unit_converter.convert(float(value), from_unit, to_unit)
            else:
                result = self._convert(float(value), from_unit, to_unit)
            
            return Response({
                'original_value': value,
                'converted_value': round(result, 2),
                'from_unit': from_unit,
                'to_unit': to_unit
            })
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    def _convert(self, value, from_unit, to_unit):
        """Fallback unit conversion."""
        conversions = {
            ('mg/dL', 'mmol/L'): lambda x: x / 38.67,  # Cholesterol
            ('mmol/L', 'mg/dL'): lambda x: x * 38.67,
            ('kg', 'lbs'): lambda x: x * 2.205,
            ('lbs', 'kg'): lambda x: x / 2.205,
            ('cm', 'in'): lambda x: x / 2.54,
            ('in', 'cm'): lambda x: x * 2.54,
        }
        key = (from_unit, to_unit)
        if key in conversions:
            return conversions[key](value)
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")


class NotificationListView(generics.ListAPIView):
    """List user's notifications."""
    permission_classes = (IsAuthenticated,)
    serializer_class = NotificationSerializer
    pagination_class = StandardResultsPagination
    
    def get_queryset(self):
        return SystemNotification.objects.filter(user=self.request.user)


class NotificationMarkReadView(APIView):
    """Mark notifications as read."""
    permission_classes = (IsAuthenticated,)
    
    def post(self, request):
        notification_ids = request.data.get('ids', [])
        if notification_ids:
            SystemNotification.objects.filter(
                user=request.user, id__in=notification_ids
            ).update(is_read=True)
        else:
            # Mark all as read
            SystemNotification.objects.filter(user=request.user).update(is_read=True)
        
        return Response({'status': 'success'})


class DashboardView(APIView):
    """Dashboard data endpoint."""
    permission_classes = (IsAuthenticated,)
    
    def get(self, request):
        user = request.user
        
        # Get recent predictions
        recent_predictions = Prediction.objects.filter(user=user)[:5]
        
        # Get unread notifications
        unread_count = SystemNotification.objects.filter(user=user, is_read=False).count()
        
        # Get document count
        document_count = MedicalDocument.objects.filter(user=user).count()
        
        return Response({
            'user': {
                'name': user.get_full_name(),
                'email': user.email,
                'role': user.role,
            },
            'stats': {
                'total_predictions': Prediction.objects.filter(user=user).count(),
                'total_documents': document_count,
                'unread_notifications': unread_count,
            },
            'recent_predictions': [
                {
                    'id': str(p.id),
                    'risk_category': p.risk_category,
                    'created_at': p.created_at.isoformat(),
                    'input_method': p.input_method,
                }
                for p in recent_predictions
            ],
            'quick_actions': [
                {'label': 'New Manual Prediction', 'url': '/predict/manual/'},
                {'label': 'Upload Document', 'url': '/predict/ocr/'},
                {'label': 'View History', 'url': '/history/'},
            ]
        })


class HealthCheckView(APIView):
    """API health check endpoint."""
    permission_classes = []  # Public endpoint
    
    def get(self, request):
        return Response({
            'status': 'healthy',
            'version': '2.0',
            'models': {
                'detection': {'accuracy': '91.45%', 'loaded': ml_service is not None},
                'prediction': {'accuracy': '91.63%', 'loaded': ml_service is not None},
            },
            'timestamp': timezone.now().isoformat()
        })


class AdminStatsView(APIView):
    """Admin dashboard stats - accessible to doctors and admins."""
    permission_classes = []  # Public for demo mode
    
    def get(self, request):
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # Get time ranges
        today = timezone.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # System stats
        total_users = User.objects.count()
        total_doctors = User.objects.filter(role='doctor').count()
        total_patients = User.objects.filter(role='patient').count()
        total_predictions = Prediction.objects.count()
        predictions_today = Prediction.objects.filter(created_at__date=today).count()
        predictions_week = Prediction.objects.filter(created_at__date__gte=week_ago).count()
        
        # Risk distribution
        risk_dist = Prediction.objects.values('risk_category').annotate(count=Count('id'))
        risk_distribution = {item['risk_category']: item['count'] for item in risk_dist}
        
        # Doctor activity (last 7 days)
        doctor_activity = []
        doctors = User.objects.filter(role='doctor').order_by('-last_login')[:10]
        for doctor in doctors:
            pred_count = Prediction.objects.filter(user=doctor, created_at__date__gte=week_ago).count()
            doctor_activity.append({
                'id': str(doctor.id),
                'name': doctor.get_full_name() or doctor.email,
                'email': doctor.email,
                'last_login': doctor.last_login.isoformat() if doctor.last_login else None,
                'predictions_this_week': pred_count,
            })
        
        # Recent predictions (all users)
        recent_predictions = Prediction.objects.select_related('user').order_by('-created_at')[:10]
        recent_list = []
        for p in recent_predictions:
            recent_list.append({
                'id': str(p.id),
                'user_name': p.user.get_full_name() or p.user.email,
                'user_role': p.user.role,
                'risk_category': p.risk_category,
                'risk_percentage': p.risk_percentage,
                'input_method': p.input_method,
                'created_at': p.created_at.isoformat(),
            })
        
        return Response({
            'system_stats': {
                'total_users': total_users,
                'total_doctors': total_doctors,
                'total_patients': total_patients,
                'total_predictions': total_predictions,
                'predictions_today': predictions_today,
                'predictions_this_week': predictions_week,
            },
            'risk_distribution': risk_distribution,
            'doctor_activity': doctor_activity,
            'recent_predictions': recent_list,
            'generated_at': timezone.now().isoformat(),
        })

