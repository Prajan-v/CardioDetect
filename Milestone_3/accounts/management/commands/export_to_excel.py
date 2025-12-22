"""
Django management command to export database to Excel.
Usage: python manage.py export_to_excel
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
import os


class Command(BaseCommand):
    help = 'Export database tables to Excel files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default='exports',
            help='Output directory for Excel files (default: exports)'
        )
        parser.add_argument(
            '--tables',
            type=str,
            default='all',
            help='Tables to export: all, users, predictions, documents (comma-separated)'
        )

    def handle(self, *args, **options):
        try:
            import pandas as pd
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        except ImportError:
            self.stderr.write(self.style.ERROR(
                'Please install pandas and openpyxl: pip install pandas openpyxl'
            ))
            return

        from accounts.models import User
        from predictions.models import Prediction, MedicalDocument, DoctorPatient

        output_dir = options['output']
        os.makedirs(output_dir, exist_ok=True)

        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        tables = options['tables'].split(',')

        self.stdout.write(self.style.SUCCESS('📊 Starting database export to Excel...'))

        # Export Users
        if 'all' in tables or 'users' in tables:
            self.export_users(pd, output_dir, timestamp)

        # Export Predictions
        if 'all' in tables or 'predictions' in tables:
            self.export_predictions(pd, output_dir, timestamp)

        # Export Documents
        if 'all' in tables or 'documents' in tables:
            self.export_documents(pd, output_dir, timestamp)

        # Export Doctor-Patient Relationships
        if 'all' in tables or 'relationships' in tables:
            self.export_relationships(pd, output_dir, timestamp)

        # Create combined workbook
        self.create_combined_workbook(pd, output_dir, timestamp)

        self.stdout.write(self.style.SUCCESS(f'\n✅ Export complete! Files saved to: {output_dir}/'))

    def export_users(self, pd, output_dir, timestamp):
        from accounts.models import User

        users = User.objects.all().values(
            'id', 'email', 'first_name', 'last_name', 'role',
            'email_verified', 'is_active', 'date_joined',
            'specialization', 'hospital', 'license_number'
        )

        df = pd.DataFrame(list(users))
        if not df.empty:
            df['date_joined'] = pd.to_datetime(df['date_joined']).dt.strftime('%Y-%m-%d %H:%M')

        filename = f'{output_dir}/users_{timestamp}.xlsx'
        df.to_excel(filename, index=False, sheet_name='Users')
        self.stdout.write(f'  ✓ Exported {len(df)} users to users_{timestamp}.xlsx')
        return df

    def export_predictions(self, pd, output_dir, timestamp):
        from predictions.models import Prediction

        predictions = Prediction.objects.all().values(
            'id', 'user__email', 'input_method', 'model_used',
            'age', 'sex', 'systolic_bp', 'cholesterol', 'hdl',
            'smoking', 'diabetes', 'bp_medication',
            'risk_score', 'risk_percentage', 'risk_category',
            'detection_result', 'detection_probability',
            'clinical_override_applied', 'created_at'
        )

        df = pd.DataFrame(list(predictions))
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            df = df.rename(columns={'user__email': 'user_email'})

        filename = f'{output_dir}/predictions_{timestamp}.xlsx'
        df.to_excel(filename, index=False, sheet_name='Predictions')
        self.stdout.write(f'  ✓ Exported {len(df)} predictions to predictions_{timestamp}.xlsx')
        return df

    def export_documents(self, pd, output_dir, timestamp):
        from predictions.models import MedicalDocument

        documents = MedicalDocument.objects.all().values(
            'id', 'user__email', 'filename', 'file_type', 'file_size',
            'ocr_status', 'ocr_confidence', 'ocr_method',
            'uploaded_at', 'processed_at', 'processing_time_ms'
        )

        df = pd.DataFrame(list(documents))
        if not df.empty:
            df['uploaded_at'] = pd.to_datetime(df['uploaded_at']).dt.strftime('%Y-%m-%d %H:%M')
            df = df.rename(columns={'user__email': 'user_email'})

        filename = f'{output_dir}/documents_{timestamp}.xlsx'
        df.to_excel(filename, index=False, sheet_name='Documents')
        self.stdout.write(f'  ✓ Exported {len(df)} documents to documents_{timestamp}.xlsx')
        return df

    def export_relationships(self, pd, output_dir, timestamp):
        from predictions.models import DoctorPatient

        relationships = DoctorPatient.objects.all().values(
            'id', 'doctor__email', 'patient__email', 'status',
            'can_view_history', 'can_add_notes', 'can_modify_predictions',
            'created_at'
        )

        df = pd.DataFrame(list(relationships))
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            df = df.rename(columns={
                'doctor__email': 'doctor_email',
                'patient__email': 'patient_email'
            })

        filename = f'{output_dir}/doctor_patient_{timestamp}.xlsx'
        df.to_excel(filename, index=False, sheet_name='Relationships')
        self.stdout.write(f'  ✓ Exported {len(df)} relationships to doctor_patient_{timestamp}.xlsx')
        return df

    def create_combined_workbook(self, pd, output_dir, timestamp):
        """Create a single Excel file with all tables as separate sheets."""
        from accounts.models import User
        from predictions.models import Prediction, MedicalDocument, DoctorPatient

        filename = f'{output_dir}/cardiodetect_database_{timestamp}.xlsx'

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Users sheet
            users = User.objects.all().values(
                'email', 'first_name', 'last_name', 'role',
                'email_verified', 'is_active', 'date_joined'
            )
            df_users = pd.DataFrame(list(users))
            if not df_users.empty:
                # Remove timezone for Excel compatibility
                if 'date_joined' in df_users.columns:
                    df_users['date_joined'] = pd.to_datetime(df_users['date_joined']).dt.tz_localize(None)
                df_users.to_excel(writer, sheet_name='Users', index=False)

            # Predictions sheet
            predictions = Prediction.objects.all().values(
                'user__email', 'age', 'sex', 'risk_category',
                'risk_percentage', 'detection_result', 'created_at'
            )
            df_pred = pd.DataFrame(list(predictions))
            if not df_pred.empty:
                df_pred = df_pred.rename(columns={'user__email': 'user_email'})
                # Remove timezone for Excel compatibility
                if 'created_at' in df_pred.columns:
                    df_pred['created_at'] = pd.to_datetime(df_pred['created_at']).dt.tz_localize(None)
                df_pred.to_excel(writer, sheet_name='Predictions', index=False)


            # Summary sheet
            summary_data = {
                'Metric': ['Total Users', 'Doctors', 'Patients', 'Total Predictions',
                           'High Risk', 'Moderate Risk', 'Low Risk'],
                'Value': [
                    User.objects.count(),
                    User.objects.filter(role='doctor').count(),
                    User.objects.filter(role='patient').count(),
                    Prediction.objects.count(),
                    Prediction.objects.filter(risk_category='HIGH').count(),
                    Prediction.objects.filter(risk_category='MODERATE').count(),
                    Prediction.objects.filter(risk_category='LOW').count(),
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)

        self.stdout.write(f'  ✓ Created combined workbook: cardiodetect_database_{timestamp}.xlsx')
