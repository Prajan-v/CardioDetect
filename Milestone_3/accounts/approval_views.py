"""
Admin Approval Views for Profile Changes.
"""

from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone

from .pending_changes import PendingProfileChange
from .models import User


class SubmitProfileChangeView(APIView):
    """Submit a profile change for admin approval."""
    permission_classes = [IsAuthenticated]

    def post(self, request):
        field_name = request.data.get('field_name')
        new_value = request.data.get('new_value')
        reason = request.data.get('reason', '')

        if not field_name or not new_value:
            return Response(
                {'error': 'field_name and new_value are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get current value
        old_value = getattr(request.user, field_name, '')

        # Create pending change
        change = PendingProfileChange.objects.create(
            user=request.user,
            field_name=field_name,
            old_value=str(old_value) if old_value else '',
            new_value=new_value,
            reason=reason
        )

        # Send confirmation email to user only
        from .email_service import send_change_submitted_email
        send_change_submitted_email(change)

        return Response({
            'message': 'Change submitted for admin approval. You will be notified once reviewed.',
            'change_id': change.id,
            'status': change.status
        }, status=status.HTTP_201_CREATED)


class MyPendingChangesView(APIView):
    """List user's pending profile changes."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        changes = PendingProfileChange.objects.filter(user=request.user)
        
        return Response({
            'pending': [
                {
                    'id': c.id,
                    'field': c.field_name,
                    'old_value': c.old_value,
                    'new_value': c.new_value,
                    'status': c.status,
                    'created_at': c.created_at.isoformat(),
                    'review_notes': c.review_notes if c.status != 'pending' else None
                }
                for c in changes
            ]
        })


class AdminPendingChangesView(APIView):
    """Admin view to list all pending changes."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Only doctors/admins can see pending changes
        if request.user.role not in ['doctor', 'admin']:
            return Response(
                {'error': 'Admin access required'},
                status=status.HTTP_403_FORBIDDEN
            )

        changes = PendingProfileChange.objects.filter(status='pending')

        return Response({
            'pending_changes': [
                {
                    'id': c.id,
                    'user_email': c.user.email,
                    'user_name': c.user.get_full_name(),
                    'field': c.field_name,
                    'old_value': c.old_value,
                    'new_value': c.new_value,
                    'reason': c.reason,
                    'created_at': c.created_at.isoformat(),
                }
                for c in changes
            ],
            'total': changes.count()
        })


class ApproveChangeView(APIView):
    """Approve a pending profile change."""
    permission_classes = [IsAuthenticated]

    def post(self, request, change_id):
        if request.user.role not in ['doctor', 'admin']:
            return Response(
                {'error': 'Admin access required'},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            change = PendingProfileChange.objects.get(id=change_id, status='pending')
        except PendingProfileChange.DoesNotExist:
            return Response(
                {'error': 'Change not found or already processed'},
                status=status.HTTP_404_NOT_FOUND
            )

        notes = request.data.get('notes', '')
        change.approve(request.user, notes)

        return Response({
            'message': 'Change approved and applied',
            'change_id': change.id
        })


class RejectChangeView(APIView):
    """Reject a pending profile change."""
    permission_classes = [IsAuthenticated]

    def post(self, request, change_id):
        if request.user.role not in ['doctor', 'admin']:
            return Response(
                {'error': 'Admin access required'},
                status=status.HTTP_403_FORBIDDEN
            )

        try:
            change = PendingProfileChange.objects.get(id=change_id, status='pending')
        except PendingProfileChange.DoesNotExist:
            return Response(
                {'error': 'Change not found or already processed'},
                status=status.HTTP_404_NOT_FOUND
            )

        notes = request.data.get('notes', 'No reason provided')
        change.reject(request.user, notes)

        return Response({
            'message': 'Change rejected',
            'change_id': change.id
        })
