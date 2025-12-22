"""
Automated Tests for CardioDetect Predictions App.
Tests manual prediction, history, statistics, and dashboard endpoints.
"""

from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from django.contrib.auth import get_user_model
from predictions.models import Prediction
import uuid

User = get_user_model()


class ManualPredictionTests(APITestCase):
    """Tests for manual prediction endpoint."""
    
    def setUp(self):
        self.predict_url = reverse('predictions:predict-manual')
        self.user = User.objects.create_user(
            email='predtest@example.com',
            password='TestPass123!',
            first_name='Prediction',
            last_name='Test',
            email_verified=True
        )
        self.client.force_authenticate(user=self.user)
        
        self.valid_payload = {
            'age': 55,
            'sex': 1,
            'systolic_bp': 140,
            'diastolic_bp': 90,
            'cholesterol': 220,
            'heart_rate': 75,
            'glucose': 100,
            'bmi': 27.5,
            'smoking': False,
            'diabetes': False,
        }
    
    def test_predict_valid_data(self):
        """Test prediction with valid clinical data returns success."""
        response = self.client.post(self.predict_url, self.valid_payload, format='json')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_201_CREATED])
    
    def test_predict_missing_required_field(self):
        """Test prediction without required field fails."""
        payload = self.valid_payload.copy()
        del payload['age']
        response = self.client.post(self.predict_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class PredictionHistoryTests(APITestCase):
    """Tests for prediction history endpoint."""
    
    def setUp(self):
        self.history_url = reverse('predictions:prediction-history')
        self.user = User.objects.create_user(
            email='histtest@example.com',
            password='TestPass123!',
            first_name='History',
            last_name='Test',
            email_verified=True
        )
        self.client.force_authenticate(user=self.user)
        
        for i in range(3):
            Prediction.objects.create(
                user=self.user,
                risk_score=0.3 + (i * 0.1),
                risk_category='MODERATE',
                input_data={'age': 50 + i, 'sex': 1},
            )
    
    def test_get_history_authenticated(self):
        """Test getting prediction history when authenticated."""
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_get_history_unauthenticated(self):
        """Test getting history when not authenticated fails."""
        self.client.force_authenticate(user=None)
        response = self.client.get(self.history_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class PredictionDetailTests(APITestCase):
    """Tests for prediction detail endpoint."""
    
    def setUp(self):
        self.user = User.objects.create_user(
            email='detailtest@example.com',
            password='TestPass123!',
            first_name='Detail',
            last_name='Test',
            email_verified=True
        )
        self.client.force_authenticate(user=self.user)
        
        self.prediction = Prediction.objects.create(
            user=self.user,
            risk_score=0.5,
            risk_category='MODERATE',
            input_data={'age': 55, 'sex': 1},
        )
        self.detail_url = reverse('predictions:prediction-detail', kwargs={'id': self.prediction.id})
    
    def test_get_prediction_detail(self):
        """Test getting single prediction detail."""
        response = self.client.get(self.detail_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_delete_prediction(self):
        """Test deleting a prediction."""
        response = self.client.delete(self.detail_url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
    
    def test_get_nonexistent_prediction(self):
        """Test getting non-existent prediction returns 404."""
        fake_url = reverse('predictions:prediction-detail', kwargs={'id': uuid.uuid4()})
        response = self.client.get(fake_url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)


class StatisticsTests(APITestCase):
    """Tests for prediction statistics endpoint."""
    
    def setUp(self):
        self.stats_url = reverse('predictions:statistics')
        self.user = User.objects.create_user(
            email='statstest@example.com',
            password='TestPass123!',
            first_name='Stats',
            last_name='Test',
            email_verified=True
        )
        self.client.force_authenticate(user=self.user)
    
    def test_get_statistics(self):
        """Test getting prediction statistics."""
        response = self.client.get(self.stats_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class DashboardTests(APITestCase):
    """Tests for dashboard endpoint."""
    
    def setUp(self):
        self.dashboard_url = reverse('predictions:dashboard')
        self.user = User.objects.create_user(
            email='dashtest@example.com',
            password='TestPass123!',
            first_name='Dashboard',
            last_name='Test',
            email_verified=True
        )
        self.client.force_authenticate(user=self.user)
    
    def test_get_dashboard(self):
        """Test getting dashboard data."""
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
    
    def test_dashboard_unauthenticated(self):
        """Test dashboard when not authenticated fails."""
        self.client.force_authenticate(user=None)
        response = self.client.get(self.dashboard_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class HealthCheckTests(APITestCase):
    """Tests for public health check endpoint."""
    
    def test_health_check_public(self):
        """Test health check endpoint is publicly accessible."""
        url = reverse('predictions:health-check')
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
