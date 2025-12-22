"""
GDPR API Views for CardioDetect.

Endpoints:
- GET /api/auth/data-export/ - Download personal data (Article 15)
- POST /api/auth/data-deletion/ - Request account deletion (Article 17)
- DELETE /api/auth/data-deletion/ - Cancel deletion request
- GET /api/auth/consent-history/ - View consent records
- POST /api/auth/consent/ - Record consent action
"""

import json
from datetime import timedelta

from django.http import JsonResponse, HttpResponse
from django.utils import timezone
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import DataDeletionRequest, ConsentRecord
from .compliance import (
    export_user_data,
    get_consent_history,
    record_consent,
)


class DataExportView(APIView):
    """
    GDPR Article 15 - Right of Access.
    Export all user data as JSON file.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Download personal data as JSON."""
        try:
            user = request.user
            data = export_user_data(user)
            
            # Create downloadable JSON response
            response = HttpResponse(
                json.dumps(data, indent=2, ensure_ascii=False),
                content_type='application/json'
            )
            filename = f"cardiodetect_data_export_{user.id}_{timezone.now().strftime('%Y%m%d')}.json"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
            # Log the export action
            from predictions.models import AuditLog
            AuditLog.objects.create(
                user=user,
                action='export',
                resource_type='UserData',
                resource_id=str(user.id),
                details={'export_type': 'full', 'format': 'json'},
                ip_address=self._get_client_ip(request),
            )
            
            return response
            
        except Exception as e:
            return Response(
                {'error': f'Export failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')


class DataDeletionView(APIView):
    """
    GDPR Article 17 - Right to Erasure.
    Request or cancel account deletion with 7-day grace period.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get current deletion request status."""
        user = request.user
        
        # Check for existing pending request
        pending = DataDeletionRequest.objects.filter(
            user=user,
            status=DataDeletionRequest.Status.PENDING
        ).first()
        
        if pending:
            return Response({
                'has_pending_request': True,
                'request_id': str(pending.id),
                'requested_at': pending.requested_at.isoformat(),
                'scheduled_deletion_at': pending.scheduled_deletion_at.isoformat(),
                'grace_period_remaining_hours': pending.grace_period_remaining.total_seconds() / 3600,
                'is_cancellable': pending.is_cancellable,
            })
        
        return Response({'has_pending_request': False})
    
    def post(self, request):
        """Create a new deletion request."""
        user = request.user
        
        # Check for existing pending request
        existing = DataDeletionRequest.objects.filter(
            user=user,
            status=DataDeletionRequest.Status.PENDING
        ).exists()
        
        if existing:
            return Response(
                {'error': 'A deletion request is already pending for this account.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Create new deletion request with 7-day grace period
        deletion_request = DataDeletionRequest.objects.create(
            user=user,
            reason=request.data.get('reason', ''),
            ip_address=self._get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500],
        )
        
        # Log the action
        from predictions.models import AuditLog
        AuditLog.objects.create(
            user=user,
            action='create',
            resource_type='DataDeletionRequest',
            resource_id=str(deletion_request.id),
            details={'reason': deletion_request.reason},
            ip_address=self._get_client_ip(request),
        )
        
        # Send confirmation email
        try:
            from .email_service import send_deletion_request_email
            send_deletion_request_email(user, deletion_request)
        except Exception:
            pass  # Non-critical
        
        return Response({
            'success': True,
            'message': 'Deletion request created. Your data will be permanently deleted in 7 days.',
            'request_id': str(deletion_request.id),
            'scheduled_deletion_at': deletion_request.scheduled_deletion_at.isoformat(),
            'cancellation_deadline': deletion_request.scheduled_deletion_at.isoformat(),
        }, status=status.HTTP_201_CREATED)
    
    def delete(self, request):
        """Cancel a pending deletion request."""
        user = request.user
        
        pending = DataDeletionRequest.objects.filter(
            user=user,
            status=DataDeletionRequest.Status.PENDING
        ).first()
        
        if not pending:
            return Response(
                {'error': 'No pending deletion request found.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if not pending.is_cancellable:
            return Response(
                {'error': 'This deletion request can no longer be cancelled.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        pending.cancel()
        
        # Log the cancellation
        from predictions.models import AuditLog
        AuditLog.objects.create(
            user=user,
            action='update',
            resource_type='DataDeletionRequest',
            resource_id=str(pending.id),
            details={'action': 'cancelled'},
            ip_address=self._get_client_ip(request),
        )
        
        return Response({
            'success': True,
            'message': 'Deletion request cancelled successfully.',
        })
    
    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')


class ConsentHistoryView(APIView):
    """
    View consent history for the authenticated user.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get consent history."""
        history = get_consent_history(request.user)
        
        # Get current consent status
        current_status = {}
        for consent_type in ConsentRecord.ConsentType.values:
            latest = ConsentRecord.objects.filter(
                user=request.user,
                consent_type=consent_type
            ).order_by('-recorded_at').first()
            current_status[consent_type] = latest.granted if latest else False
        
        return Response({
            'current_status': current_status,
            'history': history,
        })


class ConsentActionView(APIView):
    """
    Record consent grant or revocation.
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Record a consent action."""
        consent_type = request.data.get('consent_type')
        granted = request.data.get('granted')
        version = request.data.get('version', '1.0')
        
        if consent_type not in ConsentRecord.ConsentType.values:
            return Response(
                {'error': f'Invalid consent type. Valid types: {list(ConsentRecord.ConsentType.values)}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if granted is None:
            return Response(
                {'error': 'granted field is required (true/false)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        record = record_consent(
            user=request.user,
            consent_type=consent_type,
            granted=bool(granted),
            version=version,
            ip_address=self._get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
        )
        
        return Response({
            'success': True,
            'message': f'Consent {"granted" if granted else "revoked"} for {consent_type}',
            'record_id': str(record.id),
            'recorded_at': record.recorded_at.isoformat(),
        }, status=status.HTTP_201_CREATED)
    
    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR')


class AdminDeletionRequestsView(APIView):
    """
    Admin view to manage all data deletion requests.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        """Get all deletion requests for admin review."""
        # Check if user is admin
        if not request.user.is_staff and request.user.role != 'admin':
            return Response(
                {'error': 'Admin access required'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        # Get all deletion requests
        requests_list = DataDeletionRequest.objects.all().order_by('-requested_at')
        
        data = []
        for req in requests_list:
            data.append({
                'id': str(req.id),
                'user_email': req.user.email,
                'user_name': f"{req.user.first_name} {req.user.last_name}".strip() or req.user.email,
                'status': req.status,
                'reason': req.reason,
                'requested_at': req.requested_at.isoformat(),
                'scheduled_deletion_at': req.scheduled_deletion_at.isoformat() if req.scheduled_deletion_at else None,
                'grace_period_remaining_hours': req.grace_period_remaining.total_seconds() / 3600 if req.status == 'pending' else 0,
                'is_cancellable': req.is_cancellable,
                'completed_at': req.completed_at.isoformat() if req.completed_at else None,
                'cancelled_at': req.cancelled_at.isoformat() if req.cancelled_at else None,
            })
        
        # Group by status
        pending = [r for r in data if r['status'] == 'pending']
        completed = [r for r in data if r['status'] == 'completed']
        cancelled = [r for r in data if r['status'] == 'cancelled']
        
        return Response({
            'deletion_requests': data,
            'counts': {
                'pending': len(pending),
                'completed': len(completed),
                'cancelled': len(cancelled),
                'total': len(data),
            }
        })
