"""
Automated Tests for CardioDetect Accounts App.
Tests authentication, registration, profile, and password management.
"""

from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from django.contrib.auth import get_user_model

User = get_user_model()


class RegistrationTests(APITestCase):
    """Tests for user registration endpoint."""
    
    def setUp(self):
        self.register_url = reverse('accounts:register')
        self.valid_payload = {
            'email': 'testuser@example.com',
            'password': 'SecurePass123!',
            'password_confirm': 'SecurePass123!',
            'first_name': 'Test',
            'last_name': 'User',
            'role': 'patient',
        }
    
    def test_register_missing_email(self):
        """Test registration without email fails."""
        payload = self.valid_payload.copy()
        del payload['email']
        response = self.client.post(self.register_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_register_weak_password(self):
        """Test registration with weak password fails."""
        payload = self.valid_payload.copy()
        payload['password'] = '123'
        payload['password_confirm'] = '123'
        response = self.client.post(self.register_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_register_password_mismatch(self):
        """Test registration with mismatched passwords fails."""
        payload = self.valid_payload.copy()
        payload['password_confirm'] = 'DifferentPass123!'
        response = self.client.post(self.register_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
    
    def test_register_duplicate_email(self):
        """Test registration with existing email fails."""
        User.objects.create_user(
            email='testuser@example.com',
            password='ExistingPass123!',
            first_name='Existing',
            last_name='User'
        )
        response = self.client.post(self.register_url, self.valid_payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class LoginTests(APITestCase):
    """Tests for user login endpoint."""
    
    def setUp(self):
        self.login_url = reverse('accounts:login')
        self.user = User.objects.create_user(
            email='logintest@example.com',
            password='TestPass123!',
            first_name='Login',
            last_name='Test',
            email_verified=True
        )
    
    def test_login_valid_credentials(self):
        """Test login with valid credentials returns tokens."""
        payload = {'email': 'logintest@example.com', 'password': 'TestPass123!'}
        response = self.client.post(self.login_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
        self.assertIn('refresh', response.data)
    
    def test_login_missing_email(self):
        """Test login without email fails."""
        payload = {'password': 'TestPass123!'}
        response = self.client.post(self.login_url, payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class ProfileTests(APITestCase):
    """Tests for user profile endpoint."""
    
    def setUp(self):
        self.profile_url = reverse('accounts:profile')
        self.user = User.objects.create_user(
            email='profiletest@example.com',
            password='TestPass123!',
            first_name='Profile',
            last_name='Test',
            email_verified=True
        )
        self.client.force_authenticate(user=self.user)

    def test_get_profile_unauthenticated(self):
        """Test getting profile when not authenticated fails."""
        self.client.force_authenticate(user=None)
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class TokenRefreshTests(APITestCase):
    """Tests for JWT token refresh."""
    
    def setUp(self):
        self.login_url = reverse('accounts:login')
        self.refresh_url = reverse('accounts:token-refresh')
        self.user = User.objects.create_user(
            email='tokentest@example.com',
            password='TestPass123!',
            first_name='Token',
            last_name='Test',
            email_verified=True
        )
    
    def test_refresh_token_valid(self):
        """Test refreshing token with valid refresh token."""
        login_response = self.client.post(
            self.login_url, 
            {'email': 'tokentest@example.com', 'password': 'TestPass123!'}, 
            format='json'
        )
        refresh_token = login_response.data.get('refresh')
        
        response = self.client.post(self.refresh_url, {'refresh': refresh_token}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('access', response.data)
