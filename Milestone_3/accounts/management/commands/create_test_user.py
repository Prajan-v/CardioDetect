"""
Django management command to create the test user.
Creates user: prajan@cardiodetect.com with password: CardioDetect@
"""

from django.core.management.base import BaseCommand
from accounts.models import User


class Command(BaseCommand):
    help = 'Create the test user (prajan@cardiodetect.com / CardioDetect@)'

    def handle(self, *args, **options):
        email = 'prajan@cardiodetect.com'
        password = 'CardioDetect@'
        
        # Check if user already exists
        if User.objects.filter(email=email).exists():
            user = User.objects.get(email=email)
            self.stdout.write(
                self.style.WARNING(f'User {email} already exists. Updating password...')
            )
            user.set_password(password)
            user.save()
            self.stdout.write(self.style.SUCCESS(f'Password updated for {email}'))
        else:
            # Create new user
            user = User.objects.create_user(
                email=email,
                username='prajan',
                password=password,
                first_name='Prajan',
                last_name='V',
                role=User.Role.DOCTOR,
                email_verified=True,
                phone='+91 9876543210',
                city='Chennai',
                country='India',
                specialization='Cardiology',
                hospital='Apollo Hospitals',
                license_number='MCI-123456',
                bio='Cardiologist with expertise in AI-powered heart disease detection.',
            )
            user.accept_terms()
            user.accept_privacy()
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully created user: {email}')
            )
        
        self.stdout.write(self.style.SUCCESS('---'))
        self.stdout.write(self.style.SUCCESS(f'Email: {email}'))
        self.stdout.write(self.style.SUCCESS(f'Password: {password}'))
        self.stdout.write(self.style.SUCCESS(f'Role: Doctor'))
        self.stdout.write(self.style.SUCCESS('---'))
