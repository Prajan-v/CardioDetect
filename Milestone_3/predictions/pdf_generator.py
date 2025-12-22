"""
PDF Report Generator for CardioDetect.
Generates professional clinical reports for predictions.
"""

import io
from datetime import datetime
from django.http import HttpResponse
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from rest_framework.response import Response

# Try to import reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .models import Prediction


class GeneratePredictionPDFView(APIView):
    """
    Generate PDF report for a prediction.
    
    GET /api/predictions/{id}/pdf/
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, id):
        if not REPORTLAB_AVAILABLE:
            return Response({
                'error': 'PDF generation not available. Install reportlab: pip install reportlab'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        try:
            prediction = Prediction.objects.get(id=id, user=request.user)
        except Prediction.DoesNotExist:
            return Response({
                'error': 'Prediction not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Generate PDF
        buffer = io.BytesIO()
        pdf = self._generate_report(prediction, request.user, buffer)
        buffer.seek(0)
        
        filename = f"CardioDetect_Report_{prediction.created_at.strftime('%Y%m%d_%H%M%S')}.pdf"
        
        response = HttpResponse(buffer.read(), content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
    
    def _generate_report(self, prediction, user, buffer):
        """Generate the PDF report."""
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=1*cm,
            leftMargin=1*cm,
            topMargin=1*cm,
            bottomMargin=1*cm
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=HexColor('#DC2626'),
            spaceAfter=20,
            alignment=TA_CENTER,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#1E293B'),
            spaceBefore=8,
            spaceAfter=5,
            fontName='Helvetica-Bold',
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            textColor=HexColor('#374151'),
            spaceAfter=8,
        )
        
        risk_high = ParagraphStyle(
            'RiskHigh',
            parent=styles['Normal'],
            fontSize=18,
            textColor=HexColor('#DC2626'),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
        )
        
        risk_moderate = ParagraphStyle(
            'RiskModerate',
            parent=styles['Normal'],
            fontSize=18,
            textColor=HexColor('#D97706'),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
        )
        
        risk_low = ParagraphStyle(
            'RiskLow',
            parent=styles['Normal'],
            fontSize=18,
            textColor=HexColor('#059669'),
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
        )
        
        elements = []
        
        # Header
        elements.append(Paragraph("❤️ CardioDetect", title_style))
        elements.append(Paragraph("AI-Powered Cardiovascular Risk Assessment Report", 
                                  ParagraphStyle('Subtitle', parent=styles['Normal'], 
                                                fontSize=12, alignment=TA_CENTER, 
                                                textColor=HexColor('#6B7280'))))
        elements.append(Spacer(1, 10))
        
        # Horizontal line
        elements.append(HRFlowable(width="100%", thickness=2, color=HexColor('#DC2626')))
        elements.append(Spacer(1, 8))
        
        # Patient Information
        elements.append(Paragraph("📋 Patient Information", heading_style))
        
        patient_data = [
            ['Name:', user.get_full_name() or 'N/A', 'Email:', user.email],
            ['Age:', str(prediction.age or 'N/A'), 'Gender:', 'Male' if prediction.sex == 1 else 'Female' if prediction.sex == 0 else 'N/A'],
            ['Assessment Date:', prediction.created_at.strftime('%B %d, %Y'), 'Time:', prediction.created_at.strftime('%I:%M %p')],
            ['Input Method:', prediction.input_method.title() if prediction.input_method else 'Manual', 'Report ID:', str(prediction.id)[:8]],
        ]
        
        patient_table = Table(patient_data, colWidths=[2.5*cm, 5*cm, 2.5*cm, 5*cm])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#6B7280')),
            ('TEXTCOLOR', (2, 0), (2, -1), HexColor('#6B7280')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 10))
        
        # Risk Assessment Result
        elements.append(Paragraph("🩺 Risk Assessment Result", heading_style))
        
        # Risk category box
        risk_category = prediction.risk_category or 'N/A'
        risk_percentage = prediction.risk_percentage
        
        if risk_category == 'HIGH':
            risk_style = risk_high
            risk_bg = '#FEE2E2'
        elif risk_category == 'MODERATE':
            risk_style = risk_moderate
            risk_bg = '#FEF3C7'
        else:
            risk_style = risk_low
            risk_bg = '#D1FAE5'
        
        risk_data = [
            [Paragraph(f"{risk_category}", risk_style)],
            [Paragraph(f"{risk_percentage:.1f}% 10-Year Risk" if risk_percentage else "Risk percentage not available", 
                      ParagraphStyle('RiskPct', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12))],
        ]
        
        risk_table = Table(risk_data, colWidths=[15*cm])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor(risk_bg)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (0, 0), (-1, -1), 20),
            ('RIGHTPADDING', (0, 0), (-1, -1), 20),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
        ]))
        elements.append(risk_table)
        elements.append(Spacer(1, 10))
        
        # Clinical Data
        elements.append(Paragraph("📊 Clinical Measurements", heading_style))
        
        clinical_data = [
            ['Parameter', 'Value', 'Parameter', 'Value'],
            ['Systolic BP', f"{prediction.systolic_bp} mmHg" if prediction.systolic_bp else 'N/A',
             'Diastolic BP', f"{prediction.diastolic_bp} mmHg" if prediction.diastolic_bp else 'N/A'],
            ['Total Cholesterol', f"{prediction.cholesterol} mg/dL" if prediction.cholesterol else 'N/A',
             'HDL Cholesterol', f"{prediction.hdl} mg/dL" if prediction.hdl else 'N/A'],
            ['Blood Glucose', f"{prediction.glucose} mg/dL" if prediction.glucose else 'N/A',
             'BMI', f"{prediction.bmi:.1f} kg/m²" if prediction.bmi else 'N/A'],
            ['Heart Rate', f"{prediction.heart_rate} bpm" if prediction.heart_rate else 'N/A',
             'Smoking Status', 'Yes' if prediction.smoking else 'No'],
            ['Diabetes', 'Yes' if prediction.diabetes else 'No',
             'BP Medication', 'Yes' if prediction.bp_medication else 'No'],
        ]
        
        clinical_table = Table(clinical_data, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        clinical_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1E293B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#E5E7EB')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(clinical_table)
        elements.append(Spacer(1, 10))
        
        # Risk Factors
        if prediction.risk_factors:
            elements.append(Paragraph("⚠️ Identified Risk Factors", heading_style))
            for factor in prediction.risk_factors:
                elements.append(Paragraph(f"• {factor}", normal_style))
            elements.append(Spacer(1, 15))
        
        # Recommendations
        if prediction.recommendations:
            elements.append(Paragraph("💡 Recommendations", heading_style))
            elements.append(Paragraph(prediction.recommendations, normal_style))
            elements.append(Spacer(1, 15))
        
        # Disclaimer
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1, color=HexColor('#E5E7EB')))
        elements.append(Spacer(1, 10))
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=HexColor('#6B7280'),
            alignment=TA_JUSTIFY,
        )
        
        elements.append(Paragraph(
            "<b>Disclaimer:</b> This report is generated by CardioDetect AI and is intended for informational "
            "purposes only. It is NOT a medical diagnosis. Please consult with a qualified healthcare "
            "professional for proper medical advice and treatment decisions.",
            disclaimer_style
        ))
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | "
            f"CardioDetect v1.0 | Detection Accuracy: 91.45% | Prediction Accuracy: 91.63%",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                          textColor=HexColor('#9CA3AF'), alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(elements)
        
        return doc
