"""
Barcode Verification API Views.
"""

import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.utils import timezone
import hmac
import hashlib

from .barcode_models import AuthorizedDevice, BarcodeScan

# Security config - load from environment (must match report_generator.py)
BARCODE_SECRET_KEY = os.environ.get('BARCODE_SECRET_KEY', 'CardioDetect_SecureKey_2024_Hospital_Edition').encode()
ADMIN_PIN = os.environ.get('BARCODE_ADMIN_PIN', '258369')
VALIDITY_DAYS = int(os.environ.get('BARCODE_VALIDITY_DAYS', '365'))


class DeviceAuthorizationView(APIView):
    """Authorize a verification terminal device."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        device_id = request.data.get('device_id')
        pin = request.data.get('pin')
        
        if not device_id or not pin:
            return Response({'error': 'device_id and pin required'}, status=400)
        
        if pin != ADMIN_PIN:
            return Response({'error': 'Invalid PIN', 'authorized': False}, status=403)
        
        device, created = AuthorizedDevice.objects.get_or_create(
            device_fingerprint=device_id,
            defaults={'name': request.data.get('name', f'Terminal {device_id[:8]}')}
        )
        device.is_active = True
        device.save()
        
        return Response({
            'authorized': True,
            'device_id': device_id,
            'message': 'Device authorized successfully'
        })


class DeviceCheckView(APIView):
    """Check if device is authorized."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        device_id = request.data.get('device_id')
        if not device_id:
            return Response({'authorized': False}, status=400)
        
        try:
            device = AuthorizedDevice.objects.get(
                device_fingerprint=device_id, 
                is_active=True
            )
            return Response({'authorized': True, 'device_id': device_id})
        except AuthorizedDevice.DoesNotExist:
            return Response({'authorized': False})


class BarcodeVerifyView(APIView):
    """Verify barcode and log scan."""
    permission_classes = [AllowAny]
    
    def post(self, request):
        device_id = request.data.get('device_id')
        barcode = request.data.get('barcode')
        
        if not barcode:
            return Response({'error': 'barcode required', 'valid': False}, status=400)
        
        # Check device authorization
        device = None
        if device_id:
            try:
                device = AuthorizedDevice.objects.get(
                    device_fingerprint=device_id, 
                    is_active=True
                )
            except AuthorizedDevice.DoesNotExist:
                return Response({'error': 'Device not authorized', 'valid': False}, status=403)
        
        # Parse barcode
        try:
            parts = barcode.split('|')
            if not parts[0] == 'CD':
                return Response({'error': 'Invalid barcode format', 'valid': False}, status=400)
            
            version = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            is_secure = False
            data = {}
            
            if version == 3 and len(parts) >= 11:
                # V3: CD|3|ID|NAME|DATE|RISK|PROB|ACC|NPI|TIMESTAMP|HMAC
                data = {
                    'patient_id': parts[2],
                    'patient_name': parts[3],
                    'date': parts[4],
                    'risk': parts[5],
                    'probability': int(parts[6]) / 10,
                    'accession': parts[7],
                    'npi': parts[8],
                    'timestamp': int(parts[9]),
                    'signature': parts[10]
                }
                
                # Verify HMAC
                payload = f"{data['patient_id']}|{data['patient_name']}|{data['date']}|{data['risk']}|{parts[6]}|{data['accession']}|{data['npi']}|{data['timestamp']}"
                expected_sig = hmac.new(BARCODE_SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()[:12].upper()
                is_secure = expected_sig == data['signature']
                
            elif version == 2 and len(parts) >= 10:
                data = {
                    'patient_id': parts[2],
                    'patient_name': parts[3],
                    'date': parts[4],
                    'risk': parts[5],
                    'probability': int(parts[6]) / 10,
                    'accession': parts[7],
                    'npi': parts[8],
                }
            else:
                data = {
                    'patient_id': parts[1] if len(parts) > 1 else '',
                    'patient_name': parts[2] if len(parts) > 2 else '',
                    'date': parts[3] if len(parts) > 3 else '',
                    'risk': parts[4] if len(parts) > 4 else '',
                }
            
            # Log scan
            if device:
                BarcodeScan.objects.create(
                    device=device,
                    patient_id=data.get('patient_id', ''),
                    patient_name=data.get('patient_name', ''),
                    risk_level=data.get('risk', ''),
                    probability=data.get('probability'),
                    accession_number=data.get('accession', ''),
                    is_valid=is_secure,
                    barcode_version=version
                )
                device.scan_count += 1
                device.last_scan_at = timezone.now()
                device.save()
            
            return Response({
                'valid': True,
                'secure': is_secure,
                'version': version,
                'data': data
            })
            
        except Exception as e:
            return Response({'error': str(e), 'valid': False}, status=400)


class DeviceListView(APIView):
    """List all authorized devices (admin only)."""
    permission_classes = [AllowAny]  # In production: IsAdminUser
    
    def get(self, request):
        devices = AuthorizedDevice.objects.all()
        return Response({
            'devices': [
                {
                    'id': str(d.id),
                    'fingerprint': d.device_fingerprint,
                    'name': d.name,
                    'is_active': d.is_active,
                    'scan_count': d.scan_count,
                    'authorized_at': d.authorized_at.isoformat(),
                    'last_scan_at': d.last_scan_at.isoformat() if d.last_scan_at else None
                }
                for d in devices
            ]
        })


class DeviceRevokeView(APIView):
    """Revoke a device's authorization."""
    permission_classes = [AllowAny]  # In production: IsAdminUser
    
    def post(self, request):
        device_id = request.data.get('device_id')
        if not device_id:
            return Response({'error': 'device_id required'}, status=400)
        
        try:
            if device_id == 'all':
                AuthorizedDevice.objects.update(is_active=False)
                return Response({'message': 'All devices revoked'})
            else:
                device = AuthorizedDevice.objects.get(device_fingerprint=device_id)
                device.is_active = False
                device.save()
                return Response({'message': f'Device {device_id} revoked'})
        except AuthorizedDevice.DoesNotExist:
            return Response({'error': 'Device not found'}, status=404)


class ScanHistoryView(APIView):
    """Get scan history (admin only)."""
    permission_classes = [AllowAny]  # In production: IsAdminUser
    
    def get(self, request):
        scans = BarcodeScan.objects.all()[:100]
        return Response({
            'scans': [
                {
                    'id': str(s.id),
                    'patient_id': s.patient_id,
                    'patient_name': s.patient_name,
                    'risk_level': s.risk_level,
                    'probability': s.probability,
                    'is_valid': s.is_valid,
                    'scanned_at': s.scanned_at.isoformat(),
                    'device': s.device.device_fingerprint if s.device else None
                }
                for s in scans
            ]
        })


class ScanStatsView(APIView):
    """Get scan statistics."""
    permission_classes = [AllowAny]
    
    def get(self, request):
        from django.db.models import Count
        from datetime import timedelta
        
        today = timezone.now().date()
        week_ago = today - timedelta(days=7)
        
        total = BarcodeScan.objects.count()
        today_count = BarcodeScan.objects.filter(scanned_at__date=today).count()
        week_count = BarcodeScan.objects.filter(scanned_at__date__gte=week_ago).count()
        high_risk = BarcodeScan.objects.filter(risk_level__icontains='HIGH').count()
        
        return Response({
            'total': total,
            'today': today_count,
            'week': week_count,
            'high_risk': high_risk,
            'devices_active': AuthorizedDevice.objects.filter(is_active=True).count()
        })
