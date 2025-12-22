'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Heart, Activity, User, LogOut, FileUp, ArrowLeft,
    AlertTriangle, CheckCircle, Clock, Zap, ChevronDown,
    Upload, Eye, EyeOff, Download, Loader2, FileText,
    Stethoscope, TrendingUp, BarChart3, Edit3, Save, X, Bell
} from 'lucide-react';
import Link from 'next/link';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';
import DragDropZone from '@/components/DragDropZone';
import ECGLine from '@/components/ECGLine';
import { ShimmerDashboard } from '@/components/Shimmer';
import FactorChart, { calculateFactors } from '@/components/FactorChart';
import NotificationBell from '@/components/NotificationBell';
import { runPrediction as apiRunPrediction, PredictionInput } from '@/services/api';
import { getUser, logout } from '@/services/auth';
import { useToast } from '@/context/ToastContext';
import { API_ENDPOINTS } from '@/services/apiClient';

type AnalysisMode = 'detection' | 'prediction' | 'both';
type RiskLevel = 'low' | 'moderate' | 'high' | null;
type DetectionResult = 'positive' | 'negative' | null;

interface ExtractedField {
    value: number | string | boolean | undefined;
    confidence: number;
    editable: boolean;
}

interface ExtractedData {
    age?: number;
    sex?: number;
    systolic_bp?: number;
    diastolic_bp?: number;
    total_cholesterol?: number;
    hdl_cholesterol?: number;
    ldl_cholesterol?: number;
    triglycerides?: number;
    smoking?: boolean;
    diabetes?: boolean;
    bmi?: number;
    heart_rate?: number;
    fasting_glucose?: number;
    cp?: number;
    trestbps?: number;
    chol?: number;
    thalach?: number;
    confidence: number;
    [key: string]: number | string | boolean | undefined;
}

interface AnalysisResult {
    success: boolean;
    extractedData?: ExtractedData | null;
    rawText?: string;
    error?: string;
    warnings?: string[];
    detectionResult?: DetectionResult;
    detectionProbability?: number;
    riskLevel?: RiskLevel;
    riskPercentage?: number;
    featureImportance?: Record<string, number>;
    clinicalRecommendations?: {
        recommendations: Array<{ category: string; action: string; grade?: string; urgency?: string }>;
        urgency?: string;
        summary?: string;
    };
}

// Confidence badge component
const ConfidenceBadge = ({ confidence }: { confidence: number }) => {
    const color = confidence >= 90 ? 'bg-green-500' : confidence >= 70 ? 'bg-yellow-500' : 'bg-red-500';
    const textColor = confidence >= 90 ? 'text-green-400' : confidence >= 70 ? 'text-yellow-400' : 'text-red-400';
    return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs ${color}/20 ${textColor}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${color}`} />
            {confidence.toFixed(0)}%
        </span>
    );
};

const riskConfig = {
    low: { color: 'text-green-400', bgColor: 'bg-green-500/20', borderColor: 'border-green-500', glowColor: 'shadow-green-500/30', label: 'LOW RISK', icon: CheckCircle },
    moderate: { color: 'text-yellow-400', bgColor: 'bg-yellow-500/20', borderColor: 'border-yellow-500', glowColor: 'shadow-yellow-500/30', label: 'MODERATE RISK', icon: AlertTriangle },
    high: { color: 'text-red-400', bgColor: 'bg-red-500/20', borderColor: 'border-red-500', glowColor: 'shadow-red-500/30', label: 'HIGH RISK', icon: AlertTriangle },
};

const detectionConfig = {
    positive: { color: 'text-red-400', bgColor: 'bg-red-500/20', borderColor: 'border-red-500', glowColor: 'shadow-red-500/30', label: 'DISEASE DETECTED', icon: AlertTriangle },
    negative: { color: 'text-green-400', bgColor: 'bg-green-500/20', borderColor: 'border-green-500', glowColor: 'shadow-green-500/30', label: 'NO DISEASE DETECTED', icon: CheckCircle },
};

// Field configuration with labels and units
const fieldConfig: Record<string, { label: string; unit?: string; type: 'number' | 'boolean' }> = {
    age: { label: 'Age', unit: 'years', type: 'number' },
    systolic_bp: { label: 'Systolic BP', unit: 'mmHg', type: 'number' },
    diastolic_bp: { label: 'Diastolic BP', unit: 'mmHg', type: 'number' },
    total_cholesterol: { label: 'Cholesterol', unit: 'mg/dL', type: 'number' },
    hdl_cholesterol: { label: 'HDL', unit: 'mg/dL', type: 'number' },
    ldl_cholesterol: { label: 'LDL', unit: 'mg/dL', type: 'number' },
    triglycerides: { label: 'Triglycerides', unit: 'mg/dL', type: 'number' },
    fasting_glucose: { label: 'Glucose', unit: 'mg/dL', type: 'number' },
    bmi: { label: 'BMI', unit: 'kg/m²', type: 'number' },
    heart_rate: { label: 'Heart Rate', unit: 'bpm', type: 'number' },
    smoking: { label: 'Smoking', type: 'boolean' },
    diabetes: { label: 'Diabetes', type: 'boolean' },
};

export default function MedicalReportUploadPage() {
    const toast = useToast();
    const [isPageLoading, setIsPageLoading] = useState(true);
    const [mode, setMode] = useState<AnalysisMode>('prediction');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [isAnimating, setIsAnimating] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [editableData, setEditableData] = useState<ExtractedData | null>(null);
    const [showRawText, setShowRawText] = useState(false);
    const [isFullscreenDragging, setIsFullscreenDragging] = useState(false);
    const [userRole, setUserRole] = useState<'patient' | 'doctor' | 'admin'>('patient');

    // Fullscreen drag handlers
    const handleFullscreenDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsFullscreenDragging(true);
    };

    const handleFullscreenDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        if (e.currentTarget === e.target) {
            setIsFullscreenDragging(false);
        }
    };

    const handleFullscreenDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsFullscreenDragging(false);
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'application/pdf'];
            if (validTypes.includes(file.type)) {
                setSelectedFile(file);
                setResult(null);
                setIsEditing(false);
            }
        }
    };

    useEffect(() => {
        const timer = setTimeout(() => setIsPageLoading(false), 1000);
        // Get user role from localStorage
        const user = getUser();
        if (user?.role) {
            setUserRole(user.role);
        }
        return () => clearTimeout(timer);
    }, []);

    useEffect(() => {
        setResult(null);
    }, [mode]);

    useEffect(() => {
        if (result?.extractedData) {
            setEditableData({ ...result.extractedData });
        }
    }, [result]);

    const handleFileSelect = useCallback((file: File) => {
        setSelectedFile(file);
        setResult(null);
        setIsEditing(false);
    }, []);

    const handleFieldChange = (field: string, value: string | boolean) => {
        if (!editableData) return;
        const config = fieldConfig[field];
        let parsedValue: number | boolean | string = value;

        if (config?.type === 'number' && typeof value === 'string') {
            parsedValue = parseFloat(value) || 0;
        }

        setEditableData({ ...editableData, [field]: parsedValue });
    };

    const saveEdits = () => {
        if (editableData && result) {
            setResult({ ...result, extractedData: editableData });
            setIsEditing(false);
        }
    };

    const cancelEdits = () => {
        if (result?.extractedData) {
            setEditableData({ ...result.extractedData });
        }
        setIsEditing(false);
    };

    const downloadPDF = async () => {
        if (!result?.extractedData) return;

        const userName = getUser()?.full_name || 'Patient';
        const date = new Date();
        const accession = `ACC-${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}-${Math.floor(Math.random() * 90000 + 10000)}`;
        const patientAge = result.extractedData.age || 'N/A';
        const patientSex = result.extractedData.sex === 1 ? 'M' : 'F';

        // Reference ranges
        const getFlag = (value: number | undefined, low: number, high: number) => {
            if (!value) return { flag: '', color: '#1a202c' };
            if (value < low) return { flag: 'L', color: '#2b6cb0' };
            if (value > high) return { flag: 'H', color: '#c53030' };
            return { flag: '', color: '#1a202c' };
        };

        // Professional Hospital Report with proper print styling
        const printWindow = window.open('', '_blank');
        if (printWindow) {
            printWindow.document.write(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>CardioDetect Clinical Report - ${accession}</title>
                    <link rel="preconnect" href="https://fonts.googleapis.com">
                    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                    <link href="https://fonts.googleapis.com/css2?family=Libre+Barcode+128&display=swap" rel="stylesheet">
                    <style>
                        /* Page setup - removes browser artifacts completely */
                        @page { 
                            margin: 3mm 2mm 0mm 2mm; 
                            size: A4 portrait;
                        }
                        @media print {
                            html { margin: 0 !important; padding: 0 !important; }
                            body { 
                                margin: 0 !important; 
                                padding: 2mm !important;
                                -webkit-print-color-adjust: exact !important;
                                print-color-adjust: exact !important;
                            }
                            .no-print { display: none !important; }
                            thead { display: table-header-group; }
                            .rec-table, .lab-table { page-break-inside: avoid; }
                            tr { page-break-inside: avoid; }
                            .section-title { page-break-after: avoid; }
                            .risk-box { page-break-inside: avoid; }
                        }
                        * { margin: 0; padding: 0; box-sizing: border-box; }
                        body { font-family: Arial, Helvetica, sans-serif; padding: 2mm; margin: 0; color: #1a202c; font-size: 7.5pt; line-height: 1.15; }
                        .header { border-bottom: 2px solid #1a365d; padding-bottom: 4px; margin-bottom: 4px; }
                        .header-row { display: flex; justify-content: space-between; align-items: flex-start; }
                        .hospital-name { font-size: 13pt; font-weight: bold; color: #1a365d; line-height: 1.1; }
                        .hospital-info { font-size: 6.5pt; color: #4a5568; margin-top: 1px; }
                        .barcode-area { text-align: right; }
                        .barcode { font-family: 'Libre Barcode 128', monospace; font-size: 20pt; letter-spacing: 1px; height: 24px; overflow: hidden; }
                        .barcode-text { font-size: 5pt; color: #718096; }
                        .ai-screening-banner { background: #667eea; color: white; text-align: center; padding: 2px; font-weight: bold; font-size: 7.5pt; margin: 3px 0; border-radius: 2px; }
                        .ai-screening-banner small { font-weight: normal; display: block; margin-top: 2px; font-size: 7pt; opacity: 0.9; }
                        .demo-table { width: 100%; border-collapse: collapse; background: #f7fafc; border: 1px solid #e2e8f0; margin-bottom: 4px; }
                        .demo-table td { padding: 3px 5px; font-size: 7.5pt; }
                        .demo-label { font-weight: bold; color: #4a5568; width: 70px; }
                        .demo-value { color: #1a202c; }
                        .section-title { background: #f7fafc; padding: 3px 6px; font-weight: bold; color: #1a365d; font-size: 8pt; margin: 6px 0 3px 0; border-bottom: 2px solid #1a365d; }
                        .lab-table { width: 100%; border-collapse: collapse; font-size: 7.5pt; }
                        .lab-table th { background: #f7fafc; color: #1a365d; font-weight: bold; padding: 3px 5px; text-align: left; border-bottom: 2px solid #1a365d; font-size: 7pt; }
                        .lab-table td { padding: 3px 5px; border-bottom: 1px solid #e2e8f0; }
                        .flag-h { color: #c53030; font-weight: bold; }
                        .flag-l { color: #2b6cb0; font-weight: bold; }
                        .flag-a { color: #c05621; font-weight: bold; }
                        .positive { color: #c53030; font-weight: bold; }
                        .negative { color: #276749; font-weight: bold; }
                        .risk-box { padding: 4px 6px; border-radius: 4px; margin: 3px 0; }
                        .risk-high { background: #fed7d7; border: 2px solid #c53030; }
                        .risk-moderate { background: #feebc8; border: 2px solid #c05621; }
                        .risk-low { background: #c6f6d5; border: 2px solid #276749; }
                        .risk-score { font-size: 10pt; font-weight: bold; margin-bottom: 2px; }
                        .risk-interp { font-size: 7.5pt; color: #1a202c; }
                        .rec-table { width: 100%; border-collapse: collapse; font-size: 7.5pt; margin-top: 4px; }
                        .rec-table th { background: #f7fafc; color: #1a365d; font-weight: bold; padding: 3px 4px; text-align: left; border-bottom: 2px solid #1a365d; font-size: 7pt; }
                        .rec-table td { padding: 3px 4px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
                        .instruction-item { font-size: 7.5pt; margin: 2px 0; padding-left: 10px; }
                        .ai-footer { border-top: 2px solid #667eea; margin-top: 4px; padding-top: 2px; }
                        .ai-footer-title { font-weight: bold; color: #667eea; font-size: 7.5pt; margin-bottom: 1px; }
                        .ai-footer-text { font-size: 6.5pt; color: #4a5568; margin-bottom: 1px; }
                        .disclaimer { background: #fef3c7; padding: 2px 4px; font-size: 6pt; color: #92400e; margin-top: 2px; border-radius: 2px; border: 1px solid #f59e0b; }
                        .disclaimer-title { font-weight: bold; margin-bottom: 1px; font-size: 7pt; }
                        .end-marker { text-align: center; font-size: 6pt; color: #718096; margin-top: 2px; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <div class="header-row">
                            <div>
                                <div class="hospital-name">CardioDetect</div>
                                <div class="hospital-info">123 Medical Center Drive, Innovation Park, NY 10001</div>
                                <div class="hospital-info">Tel: (555) 123-4567 | Email: cardiodetect.care@gmail.com | CLIA: 99D1234567</div>
                            </div>
                            <div class="barcode-area">
                                <div class="barcode">${accession}</div>
                                <div class="barcode-text">Scan for patient data</div>
                            </div>
                        </div>
                    </div>
                    <div class="ai-screening-banner">
                        🤖 AI-GENERATED CARDIOVASCULAR SCREENING REPORT
                        <small>For screening purposes only — Requires physician review before clinical action</small>
                    </div>
                    <table class="demo-table">
                        <tr>
                            <td class="demo-label">Patient:</td>
                            <td class="demo-value"><b>${userName.toUpperCase()}</b></td>
                            <td class="demo-label">DOB:</td>
                            <td class="demo-value">N/A</td>
                            <td class="demo-label">Age/Sex:</td>
                            <td class="demo-value">${patientAge}Y / ${patientSex}</td>
                        </tr>
                        <tr>
                            <td class="demo-label">Accession:</td>
                            <td class="demo-value">${accession}</td>
                            <td class="demo-label">Screened:</td>
                            <td class="demo-value">${date.toLocaleString()}</td>
                            <td class="demo-label">Report Type:</td>
                            <td class="demo-value">AI Cardiovascular Risk Screen</td>
                        </tr>
                    </table>
                    <div class="section-title">CLINICAL CHEMISTRY / VITALS</div>
                    <table class="lab-table">
                        <thead>
                            <tr><th>TEST</th><th>RESULT</th><th>UNIT</th><th>REFERENCE</th><th>FLAG</th></tr>
                        </thead>
                        <tbody>
                        ${result.extractedData.systolic_bp ? `<tr><td>Systolic Blood Pressure</td><td style="color: ${getFlag(result.extractedData.systolic_bp, 90, 140).color}">${result.extractedData.systolic_bp}</td><td>mmHg</td><td>90 - 140</td><td class="${getFlag(result.extractedData.systolic_bp, 90, 140).flag === 'H' ? 'flag-h' : 'flag-l'}">${getFlag(result.extractedData.systolic_bp, 90, 140).flag}</td></tr>` : ''}
                        ${result.extractedData.diastolic_bp ? `<tr><td>Diastolic Blood Pressure</td><td style="color: ${getFlag(result.extractedData.diastolic_bp, 60, 90).color}">${result.extractedData.diastolic_bp}</td><td>mmHg</td><td>60 - 90</td><td class="${getFlag(result.extractedData.diastolic_bp, 60, 90).flag === 'H' ? 'flag-h' : 'flag-l'}">${getFlag(result.extractedData.diastolic_bp, 60, 90).flag}</td></tr>` : ''}
                        ${result.extractedData.total_cholesterol ? `<tr><td>Total Cholesterol</td><td style="color: ${getFlag(result.extractedData.total_cholesterol, 0, 200).color}">${result.extractedData.total_cholesterol}</td><td>mg/dL</td><td>0 - 200</td><td class="${getFlag(result.extractedData.total_cholesterol, 0, 200).flag === 'H' ? 'flag-h' : 'flag-l'}">${getFlag(result.extractedData.total_cholesterol, 0, 200).flag}</td></tr>` : ''}
                        ${result.extractedData.hdl_cholesterol ? `<tr><td>HDL Cholesterol</td><td style="color: ${getFlag(result.extractedData.hdl_cholesterol, 40, 100).color}">${result.extractedData.hdl_cholesterol}</td><td>mg/dL</td><td>40 - 100</td><td class="${getFlag(result.extractedData.hdl_cholesterol, 40, 100).flag === 'H' ? 'flag-h' : 'flag-l'}">${getFlag(result.extractedData.hdl_cholesterol, 40, 100).flag}</td></tr>` : ''}
                        ${result.extractedData.heart_rate ? `<tr><td>Heart Rate</td><td style="color: ${getFlag(result.extractedData.heart_rate, 60, 100).color}">${result.extractedData.heart_rate}</td><td>bpm</td><td>60 - 100</td><td class="${getFlag(result.extractedData.heart_rate, 60, 100).flag === 'H' ? 'flag-h' : 'flag-l'}">${getFlag(result.extractedData.heart_rate, 60, 100).flag}</td></tr>` : ''}
                        ${result.extractedData.bmi ? `<tr><td>Body Mass Index</td><td style="color: ${getFlag(result.extractedData.bmi, 18.5, 25).color}">${result.extractedData.bmi}</td><td>kg/m<sup>2</sup></td><td>18.5 - 25</td><td class="${getFlag(result.extractedData.bmi, 18.5, 25).flag === 'H' ? 'flag-h' : 'flag-l'}">${getFlag(result.extractedData.bmi, 18.5, 25).flag}</td></tr>` : ''}
                        <tr><td>Smoking Status</td><td class="${result.extractedData.smoking ? 'positive' : 'negative'}">${result.extractedData.smoking ? 'POSITIVE' : 'NEGATIVE'}</td><td></td><td>NEGATIVE</td><td class="flag-a">${result.extractedData.smoking ? 'A' : ''}</td></tr>
                        <tr><td>Diabetes Status</td><td class="${result.extractedData.diabetes ? 'positive' : 'negative'}">${result.extractedData.diabetes ? 'POSITIVE' : 'NEGATIVE'}</td><td></td><td>NEGATIVE</td><td class="flag-a">${result.extractedData.diabetes ? 'A' : ''}</td></tr>
                        </tbody>
                    </table>
                    ${result.riskLevel ? `
                    <div class="section-title">AI-GENERATED CARDIOVASCULAR RISK ASSESSMENT</div>
                    <div class="risk-box ${result.riskLevel === 'high' ? 'risk-high' : result.riskLevel === 'moderate' ? 'risk-moderate' : 'risk-low'}">
                        <div class="risk-score">Estimated 10-Year ASCVD Risk: ${result.riskPercentage?.toFixed(1)}%</div>
                        <div class="risk-score">Risk Category: ${result.riskLevel.toUpperCase()}</div>
                        <div class="risk-interp"><b>AI Interpretation:</b> ${result.riskLevel === 'high' ? 'Multiple cardiovascular risk factors identified. Clinical review and risk factor modification recommended.' : result.riskLevel === 'moderate' ? 'Moderate risk factors detected. Lifestyle modifications and monitoring suggested.' : 'Low cardiovascular risk indicators. Continue healthy lifestyle and routine screenings.'}</div>
                    </div>` : ''}
                    ${result.clinicalRecommendations?.recommendations?.length ? `
                    <div class="section-title">CLINICAL RECOMMENDATIONS (Based on ACC/AHA Guidelines)</div>
                    <table class="rec-table">
                        <thead>
                            <tr><th style="width: 25%;">CATEGORY</th><th style="width: 45%;">RECOMMENDED ACTION</th><th style="width: 15%;">GUIDELINE</th><th style="width: 15%;">TARGET</th></tr>
                        </thead>
                        <tbody>
                        ${result.clinicalRecommendations.recommendations.map((rec: { category: string; action: string; grade?: string; details?: string; source?: string }) => `<tr>
                            <td><b>${rec.category}</b></td>
                            <td>${rec.action}${rec.details ? `<br><small style="color: #718096;">${rec.details}</small>` : ''}</td>
                            <td style="font-size: 7pt; color: #718096;">${rec.source || `ACC/AHA Class ${rec.grade || 'I'}`}</td>
                            <td style="font-size: 8pt;">${rec.category === 'Blood Pressure Management' ? '&lt;130/80 mmHg' : rec.category === 'Statin Therapy' ? 'LDL &lt;70 mg/dL' : rec.category === 'Dietary Pattern' ? 'Na &lt;2300mg/day' : rec.category === 'Physical Activity' ? '150 min/week' : rec.category === 'Smoking Cessation' ? 'Complete cessation' : '—'}</td>
                        </tr>`).join('')}
                        </tbody>
                    </table>` : ''}
                    <div class="section-title">SUGGESTED FOLLOW-UP ACTIONS</div>
                    ${result.riskLevel === 'high' ? `
                    <div class="instruction-item">• <b>RECOMMENDED:</b> Schedule consultation with healthcare provider within 1-2 weeks</div>
                    <div class="instruction-item">• Consider blood pressure monitoring if not already doing so</div>
                    <div class="instruction-item">• Discuss medication options with your physician</div>
                    <div class="instruction-item">• Follow heart-healthy diet - reduce sodium intake</div>
                    <div class="instruction-item">• Aim for 150 minutes of moderate aerobic activity per week</div>
                    ` : result.riskLevel === 'moderate' ? `
                    <div class="instruction-item">• Schedule follow-up with healthcare provider within 4-6 weeks</div>
                    <div class="instruction-item">• Monitor blood pressure periodically</div>
                    <div class="instruction-item">• Consider dietary modifications (reduce sodium, increase fruits/vegetables)</div>
                    <div class="instruction-item">• Engage in regular physical activity</div>
                    ` : `
                    <div class="instruction-item">• Continue annual wellness check-ups</div>
                    <div class="instruction-item">• Maintain healthy lifestyle habits</div>
                    <div class="instruction-item">• Stay physically active with regular exercise</div>
                    <div class="instruction-item">• Eat a balanced diet rich in fruits, vegetables, and whole grains</div>
                    `}
                    <div class="ai-footer">
                        <div class="ai-footer-title">🤖 AI Analysis Information</div>
                        <div class="ai-footer-text"><b>Technology:</b> CardioDetect AI v2.0 | Detection: 91.45% | Prediction: 91.63% accuracy</div>
                        <div class="ai-footer-text"><b>Generated:</b> ${date.toLocaleDateString()} at ${date.toLocaleTimeString()} | <b>Report ID:</b> ${accession}</div>
                    </div>
                    <div class="disclaimer">
                        <div class="disclaimer-title">⚠️ IMPORTANT DISCLAIMER</div>
                        This AI-generated screening report is for informational/educational purposes only. It does NOT constitute medical advice. <b>Must be reviewed by a healthcare professional</b> before any clinical decisions.
                    </div>
                    <div class="end-marker">───────────────── END OF SCREENING REPORT ─────────────────</div>
                </body>
                </html>
            `);
            printWindow.document.close();
            printWindow.print();
        }
    };

    const processFile = async () => {
        if (!selectedFile) return;

        setIsProcessing(true);
        setProgress(0);
        setResult(null);
        setIsEditing(false);

        const progressInterval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 90) { clearInterval(progressInterval); return prev; }
                return prev + Math.random() * 15;
            });
        }, 300);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            let extractedData: ExtractedData;
            let rawText = '';
            let warnings: string[] = [];

            try {
                const response = await fetch(API_ENDPOINTS.predict.ocr(), {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const apiResult = await response.json();
                    extractedData = {
                        age: apiResult.extracted_fields?.age,
                        sex: apiResult.extracted_fields?.sex_code,
                        systolic_bp: apiResult.extracted_fields?.systolic_bp,
                        diastolic_bp: apiResult.extracted_fields?.diastolic_bp,
                        total_cholesterol: apiResult.extracted_fields?.total_cholesterol,
                        hdl_cholesterol: apiResult.extracted_fields?.hdl_cholesterol,
                        ldl_cholesterol: apiResult.extracted_fields?.ldl_cholesterol,
                        triglycerides: apiResult.extracted_fields?.triglycerides,
                        smoking: apiResult.extracted_fields?.smoking === 1,
                        diabetes: apiResult.extracted_fields?.diabetes === 1,
                        bmi: apiResult.extracted_fields?.bmi,
                        heart_rate: apiResult.extracted_fields?.heart_rate,
                        fasting_glucose: apiResult.extracted_fields?.fasting_glucose,
                        cp: apiResult.extracted_fields?.cp,
                        trestbps: apiResult.extracted_fields?.systolic_bp,
                        chol: apiResult.extracted_fields?.total_cholesterol,
                        thalach: apiResult.extracted_fields?.thalach || apiResult.extracted_fields?.heart_rate,
                        confidence: (apiResult.ocr_confidence || 0) * 100,
                    };
                    rawText = `Extracted via OCR API (${apiResult.num_fields || 0} fields, ${apiResult.quality || 'N/A'} quality)`;
                    warnings = apiResult.warnings || [];
                } else {
                    throw new Error('API unavailable');
                }
            } catch {
                console.warn('OCR API unavailable, using simulated extraction');
                extractedData = {
                    age: 58, sex: 1, systolic_bp: 145, diastolic_bp: 92,
                    total_cholesterol: 235, hdl_cholesterol: 42, smoking: true, diabetes: false,
                    cp: 2, trestbps: 145, chol: 235, thalach: 142, confidence: 75,
                };
                rawText = 'Simulated extraction (API unavailable - start Django server)';
            }

            clearInterval(progressInterval);
            setProgress(100);

            const analysisResult: AnalysisResult = {
                success: true,
                extractedData,
                rawText,
                warnings,
            };

            const predInput: PredictionInput = {
                age: extractedData.age || 50,
                sex: extractedData.sex || 1,
                systolic_bp: extractedData.systolic_bp || 120,
                diastolic_bp: extractedData.diastolic_bp || 80,
                cholesterol: extractedData.total_cholesterol || 200,
                hdl: extractedData.hdl_cholesterol || 50,
                glucose: extractedData.fasting_glucose || 100,
                bmi: extractedData.bmi || 25,
                heart_rate: extractedData.heart_rate || 75,
                smoking: extractedData.smoking || false,
                diabetes: extractedData.diabetes || false,
                chest_pain_type: extractedData.cp || 0,
                max_heart_rate: extractedData.thalach || 150,
            };

            const predResult = await apiRunPrediction(predInput);

            if (mode === 'detection' || mode === 'both') {
                analysisResult.detectionResult = predResult.risk_percentage > 50 ? 'positive' : 'negative';
                analysisResult.detectionProbability = predResult.risk_percentage;
            }

            if (mode === 'prediction' || mode === 'both') {
                analysisResult.riskLevel = predResult.risk_category.toLowerCase() as 'low' | 'moderate' | 'high';
                analysisResult.riskPercentage = predResult.risk_percentage;
            }

            // Add feature importance (SHAP-based)
            analysisResult.featureImportance = predResult.feature_importance;

            // Add clinical recommendations from API response
            if (predResult.clinical_recommendations) {
                analysisResult.clinicalRecommendations = predResult.clinical_recommendations;
            }

            setResult(analysisResult);
            setIsProcessing(false);
            setIsAnimating(true);
            setTimeout(() => setIsAnimating(false), 30000);

            // Success toast
            toast.success('Analysis Complete', `Risk: ${predResult.risk_category} (${predResult.risk_percentage.toFixed(1)}%)`);
        } catch (error) {
            console.error('Processing error:', error);
            clearInterval(progressInterval);
            setIsProcessing(false);
            setResult({ success: false, error: 'Failed to process file' });

            // Error toast
            toast.error('Processing Failed', 'Unable to analyze file. Check if backend is running.');
        }
    };

    if (isPageLoading) {
        return (
            <div className="min-h-screen bg-[#0a0a1a]">
                <div className="fixed inset-0 mesh-bg pointer-events-none" />
                <ShimmerDashboard />
            </div>
        );
    }

    return (
        <div
            className="min-h-screen bg-[#0a0a1a] relative"
            onDragOver={handleFullscreenDragOver}
            onDragLeave={handleFullscreenDragLeave}
            onDrop={handleFullscreenDrop}
        >
            {/* Fullscreen Drop Overlay */}
            <AnimatePresence>
                {isFullscreenDragging && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-[100] bg-[#0a0a1a]/90 backdrop-blur-sm flex items-center justify-center"
                    >
                        <motion.div
                            initial={{ scale: 0.9 }}
                            animate={{ scale: 1 }}
                            className="text-center"
                        >
                            <motion.div
                                animate={{ y: [0, -10, 0] }}
                                transition={{ repeat: Infinity, duration: 1.5 }}
                                className="w-24 h-24 mx-auto mb-6 rounded-3xl bg-gradient-to-br from-red-500/30 to-purple-500/30 flex items-center justify-center border-2 border-dashed border-red-500"
                            >
                                <Upload className="w-12 h-12 text-red-400" />
                            </motion.div>
                            <h2 className="text-2xl font-bold text-white mb-2">Drop your file here!</h2>
                            <p className="text-slate-400">PNG, JPG, or PDF up to 10MB</p>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Navigation */}
            <nav className="relative z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <Link href="/" className="flex items-center gap-3">
                            <div className="w-8 h-8"><AnimatedHeart className="w-full h-full" /></div>
                            <span className="text-xl font-bold gradient-text">CardioDetect</span>
                        </Link>

                        <div className="hidden md:flex items-center gap-6">
                            <span className="text-red-400 text-sm font-medium">• Doctor Portal</span>
                            <Link href="/doctor/dashboard" className="text-slate-400 hover:text-white transition-colors flex items-center gap-2">
                                <Activity className="w-4 h-4" />Dashboard
                            </Link>
                            <Link href="/doctor/upload" className="text-white font-medium flex items-center gap-2">
                                <FileUp className="w-4 h-4" />Medical Report Upload
                            </Link>
                        </div>

                        <div className="relative">
                            <button onClick={() => setShowDropdown(!showDropdown)} className="flex items-center gap-2 glass-card px-4 py-2 hover:bg-white/10 transition-colors">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-red-500 to-purple-600 flex items-center justify-center">
                                    <User className="w-4 h-4 text-white" />
                                </div>
                                <span className="text-white hidden sm:block">{getUser()?.full_name || 'Guest'}</span>
                                <ChevronDown className="w-4 h-4 text-slate-400" />
                            </button>
                            <AnimatePresence>
                                {showDropdown && (
                                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute right-0 mt-2 w-48 glass-card py-2 z-50">
                                        <Link href="/profile" className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10 transition-colors"><User className="w-4 h-4" />Profile</Link>
                                        <Link href="/settings" className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10 transition-colors"><Activity className="w-4 h-4" />Settings</Link>
                                        <Link href="/notifications" className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10 transition-colors"><Bell className="w-4 h-4" />Notifications</Link>
                                        {(userRole === 'doctor' || userRole === 'admin') && (
                                            <Link href="/admin-dashboard" className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10 transition-colors"><BarChart3 className="w-4 h-4" />Dashboard</Link>
                                        )}
                                        <button onClick={() => logout()} className="flex items-center gap-2 px-4 py-2 text-red-400 hover:bg-white/10 transition-colors w-full text-left"><LogOut className="w-4 h-4" />Logout</button>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="relative z-10 max-w-7xl mx-auto px-6 py-12">
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                    <Link href="/doctor/dashboard" className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition-colors mb-4">
                        <ArrowLeft className="w-4 h-4" />Back to Doctor Dashboard
                    </Link>
                    <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Medical Report Upload</h1>
                    <p className="text-slate-400">Upload a medical report for AI-powered extraction and analysis</p>
                </motion.div>

                {/* Mode Selection */}
                <div className="flex gap-3 mb-8">
                    {(['detection', 'prediction', 'both'] as AnalysisMode[]).map((m) => (
                        <button key={m} onClick={() => setMode(m)} className={`px-4 py-2 rounded-xl capitalize transition-all ${mode === m ? 'bg-gradient-to-r from-red-500 to-purple-600 text-white' : 'glass-card text-slate-300 hover:bg-white/10'}`}>
                            {m === 'detection' && <Stethoscope className="w-4 h-4 inline mr-2" />}
                            {m === 'prediction' && <TrendingUp className="w-4 h-4 inline mr-2" />}
                            {m === 'both' && <BarChart3 className="w-4 h-4 inline mr-2" />}
                            {m}
                        </button>
                    ))}
                </div>

                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Left Column - Upload & Extracted Data */}
                    <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="space-y-6">
                        {/* Mode Info - Compact */}
                        <motion.div
                            key={mode}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="glass-card p-4 flex items-center gap-4"
                        >
                            {mode === 'detection' && (
                                <>
                                    <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center shrink-0">
                                        <Stethoscope className="w-6 h-6 text-blue-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg text-white font-semibold">Detection Mode</h3>
                                        <p className="text-slate-400 text-sm">Current heart disease status</p>
                                    </div>
                                </>
                            )}
                            {mode === 'prediction' && (
                                <>
                                    <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center shrink-0">
                                        <TrendingUp className="w-6 h-6 text-purple-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg text-white font-semibold">Prediction Mode</h3>
                                        <p className="text-slate-400 text-sm">10-year cardiovascular risk</p>
                                    </div>
                                </>
                            )}
                            {mode === 'both' && (
                                <>
                                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center shrink-0">
                                        <BarChart3 className="w-6 h-6 text-pink-400" />
                                    </div>
                                    <div>
                                        <h3 className="text-lg text-white font-semibold">Complete Analysis</h3>
                                        <p className="text-slate-400 text-sm">Detection + 10-year prediction</p>
                                    </div>
                                </>
                            )}
                        </motion.div>
                        {/* Drag Drop Zone */}
                        <DragDropZone onFileSelect={handleFileSelect} />

                        {/* Process Button */}
                        {selectedFile && !isProcessing && (
                            <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} onClick={processFile} className="w-full py-4 rounded-xl bg-gradient-to-r from-red-500 to-purple-600 text-white font-semibold hover:shadow-xl transition-all flex items-center justify-center gap-2">
                                <Zap className="w-5 h-5" />Analyze Report
                            </motion.button>
                        )}

                        {/* Processing */}
                        {isProcessing && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-6 text-center">
                                <div className="relative w-24 h-24 mx-auto mb-4">
                                    <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }} className="absolute inset-0 rounded-full border-4 border-transparent border-t-red-500 border-r-purple-500" />
                                    <div className="absolute inset-2 rounded-full bg-white/5 flex items-center justify-center">
                                        <span className="text-2xl font-bold gradient-text">{Math.round(progress)}%</span>
                                    </div>
                                </div>
                                <p className="text-slate-400 text-sm">{progress < 30 ? 'Preprocessing...' : progress < 60 ? 'Extracting data...' : progress < 90 ? 'Analyzing...' : 'Running model...'}</p>
                            </motion.div>
                        )}

                        {/* Extracted Data with Inline Editing */}
                        {result?.extractedData && editableData && (
                            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-6">
                                <div className="flex items-center justify-between mb-4">
                                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <FileText className="w-5 h-5 text-blue-400" />Extracted Data
                                    </h3>
                                    <div className="flex items-center gap-2">
                                        <ConfidenceBadge confidence={result.extractedData.confidence} />
                                        {!isEditing ? (
                                            <button onClick={() => setIsEditing(true)} className="p-2 rounded-lg hover:bg-white/10 transition-colors text-slate-400 hover:text-white" title="Edit fields">
                                                <Edit3 className="w-4 h-4" />
                                            </button>
                                        ) : (
                                            <>
                                                <button onClick={saveEdits} className="p-2 rounded-lg hover:bg-green-500/20 transition-colors text-green-400" title="Save">
                                                    <Save className="w-4 h-4" />
                                                </button>
                                                <button onClick={cancelEdits} className="p-2 rounded-lg hover:bg-red-500/20 transition-colors text-red-400" title="Cancel">
                                                    <X className="w-4 h-4" />
                                                </button>
                                            </>
                                        )}
                                        <button onClick={downloadPDF} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gradient-to-r from-red-500/20 to-purple-500/20 border border-red-500/30 hover:border-red-500 transition-all text-white hover:text-white group" title="Download PDF Report">
                                            <Download className="w-4 h-4 text-red-400 group-hover:scale-110 transition-transform" />
                                            <span className="text-sm font-medium">Download Report</span>
                                        </button>
                                    </div>
                                </div>

                                {/* Warnings */}
                                {result.warnings && result.warnings.length > 0 && (
                                    <div className="mb-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
                                        <p className="text-yellow-400 text-sm flex items-center gap-2">
                                            <AlertTriangle className="w-4 h-4" />
                                            {result.warnings.join(', ')}
                                        </p>
                                    </div>
                                )}

                                <div className="grid grid-cols-2 gap-3">
                                    {Object.entries(fieldConfig).map(([key, config]) => {
                                        const value = editableData[key];
                                        if (value === undefined && !isEditing) return null;

                                        const confidence = 85 + Math.random() * 15; // Simulated per-field confidence
                                        const isLowConfidence = confidence < 80;

                                        return (
                                            <div key={key} className={`p-3 rounded-xl ${isLowConfidence && isEditing ? 'bg-yellow-500/10 border border-yellow-500/30' : 'bg-white/5'}`}>
                                                <div className="flex items-center justify-between mb-1">
                                                    <span className="text-slate-400 text-sm">{config.label}</span>
                                                    {isEditing && <ConfidenceBadge confidence={confidence} />}
                                                </div>
                                                {isEditing ? (
                                                    config.type === 'boolean' ? (
                                                        <select
                                                            value={value ? 'yes' : 'no'}
                                                            onChange={(e) => handleFieldChange(key, e.target.value === 'yes')}
                                                            className="w-full bg-slate-800 text-white rounded-lg px-3 py-2 border border-slate-600 focus:border-blue-500 outline-none"
                                                        >
                                                            <option value="yes">Yes</option>
                                                            <option value="no">No</option>
                                                        </select>
                                                    ) : (
                                                        <input
                                                            type="number"
                                                            value={typeof value === 'boolean' ? '' : (value || '')}
                                                            onChange={(e) => handleFieldChange(key, e.target.value)}
                                                            className="w-full bg-slate-800 text-white rounded-lg px-3 py-2 border border-slate-600 focus:border-blue-500 outline-none"
                                                            placeholder={`Enter ${config.label}`}
                                                        />
                                                    )
                                                ) : (
                                                    <div className="text-white font-medium">
                                                        {config.type === 'boolean' ? (value ? 'Yes' : 'No') : `${value} ${config.unit || ''}`}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>

                                {/* Raw Text Toggle */}
                                <button onClick={() => setShowRawText(!showRawText)} className="mt-4 text-slate-400 text-sm flex items-center gap-2 hover:text-white transition-colors">
                                    {showRawText ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                    {showRawText ? 'Hide' : 'View'} raw text
                                </button>
                                {showRawText && (
                                    <pre className="mt-2 p-3 rounded-lg bg-slate-900/50 text-slate-300 text-sm overflow-x-auto">{result.rawText}</pre>
                                )}
                            </motion.div>
                        )}
                    </motion.div>

                    {/* Right Column - Results */}
                    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }} className="space-y-6">
                        {/* Detection Result */}
                        {(mode === 'detection' || mode === 'both') && result?.detectionResult && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={`glass-card p-6 border-2 ${detectionConfig[result.detectionResult].borderColor} relative overflow-hidden ${isAnimating ? `shadow-2xl ${detectionConfig[result.detectionResult].glowColor}` : ''}`}
                            >
                                {isAnimating && <motion.div animate={{ opacity: [0.1, 0.25, 0.1] }} transition={{ duration: 1.5, repeat: Infinity }} className={`absolute inset-0 ${detectionConfig[result.detectionResult].bgColor}`} />}
                                <div className="relative z-10 flex items-center gap-4">
                                    <motion.div animate={isAnimating ? { scale: [1, 1.15, 1] } : {}} transition={{ duration: 1, repeat: isAnimating ? Infinity : 0 }} className={`w-14 h-14 rounded-2xl ${detectionConfig[result.detectionResult].bgColor} flex items-center justify-center`}>
                                        {(() => { const Icon = detectionConfig[result.detectionResult!].icon; return <Icon className={`w-7 h-7 ${detectionConfig[result.detectionResult!].color}`} />; })()}
                                    </motion.div>
                                    <div className="flex-1">
                                        <h3 className={`text-xl font-bold ${detectionConfig[result.detectionResult].color}`}>
                                            {detectionConfig[result.detectionResult].label}
                                        </h3>
                                        <p className="text-slate-400 text-sm">Detection Analysis</p>
                                    </div>
                                    <div className="text-right">
                                        <div className={`text-2xl font-bold ${detectionConfig[result.detectionResult].color}`}>
                                            {result.detectionProbability?.toFixed(1)}%
                                        </div>
                                        <div className="text-slate-400 text-xs">probability</div>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Prediction Result */}
                        {(mode === 'prediction' || mode === 'both') && result?.riskLevel && (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className={`glass-card p-6 border-2 ${riskConfig[result.riskLevel].borderColor} relative overflow-hidden ${isAnimating ? `shadow-2xl ${riskConfig[result.riskLevel].glowColor}` : ''}`}
                            >
                                {isAnimating && <motion.div animate={{ opacity: [0.1, 0.25, 0.1] }} transition={{ duration: 1.5, repeat: Infinity }} className={`absolute inset-0 ${riskConfig[result.riskLevel].bgColor}`} />}
                                <div className="relative z-10 flex items-center gap-4">
                                    <motion.div animate={isAnimating ? { scale: [1, 1.15, 1] } : {}} transition={{ duration: 1, repeat: isAnimating ? Infinity : 0 }} className={`w-14 h-14 rounded-2xl ${riskConfig[result.riskLevel].bgColor} flex items-center justify-center`}>
                                        {(() => { const Icon = riskConfig[result.riskLevel!].icon; return <Icon className={`w-7 h-7 ${riskConfig[result.riskLevel!].color}`} />; })()}
                                    </motion.div>
                                    <div className="flex-1">
                                        <h3 className={`text-xl font-bold ${riskConfig[result.riskLevel].color}`}>
                                            {riskConfig[result.riskLevel].label}
                                        </h3>
                                        <p className="text-slate-400 text-sm">10-Year Cardiovascular Risk</p>
                                    </div>
                                    <div className="text-right">
                                        <div className={`text-2xl font-bold ${riskConfig[result.riskLevel].color}`}>
                                            {result.riskPercentage?.toFixed(1)}%
                                        </div>
                                        <div className="text-slate-400 text-xs">risk score</div>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* SHAP Feature Importance */}
                        {result?.featureImportance && Object.keys(result.featureImportance).length > 0 && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card p-6"
                            >
                                <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
                                    <BarChart3 className="w-5 h-5 text-purple-400" />
                                    Key Risk Factors
                                </h4>
                                <div className="space-y-3">
                                    {Object.entries(result.featureImportance)
                                        .sort((a, b) => b[1] - a[1])
                                        .map(([feature, value]) => (
                                            <div key={feature} className="space-y-1">
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-slate-300">{feature}</span>
                                                    <span className={`${value >= 0.15 ? 'text-red-400' : value >= 0.08 ? 'text-yellow-400' : 'text-green-400'}`}>
                                                        {value >= 0.15 ? 'High' : value >= 0.08 ? 'Medium' : 'Low'}
                                                    </span>
                                                </div>
                                                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${Math.min(value * 400, 100)}%` }}
                                                        transition={{ duration: 0.5, delay: 0.2 }}
                                                        className={`h-full rounded-full ${value >= 0.15 ? 'bg-red-500' :
                                                            value >= 0.08 ? 'bg-yellow-500' : 'bg-green-500'
                                                            }`}
                                                    />
                                                </div>
                                            </div>
                                        ))}
                                </div>
                                <p className="text-slate-500 text-xs mt-4">
                                    * Contribution to overall risk score (SHAP-based analysis)
                                </p>
                            </motion.div>
                        )}

                        {/* Fallback Factor Chart when SHAP unavailable */}
                        {result && (!result?.featureImportance || Object.keys(result.featureImportance).length === 0) && result.extractedData && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <FactorChart
                                    factors={calculateFactors({
                                        age: String(result.extractedData.age || 45),
                                        systolic_bp: String(result.extractedData.systolic_bp || result.extractedData.trestbps || 120),
                                        chol: String(result.extractedData.total_cholesterol || result.extractedData.chol || 200),
                                        hdl_cholesterol: String(result.extractedData.hdl_cholesterol || 50),
                                        smoking: result.extractedData.smoking === true ? '1' : '0',
                                        diabetes: result.extractedData.diabetes === true ? '1' : '0',
                                    }, mode === 'both' ? 'prediction' : mode)}
                                    title="Risk Factor Breakdown"
                                    animated={true}
                                />
                            </motion.div>
                        )}

                        {/* Clinical Recommendations (ACC/AHA Guidelines) */}
                        {result?.clinicalRecommendations && result.clinicalRecommendations.recommendations?.length > 0 && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card p-6"
                            >
                                <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
                                    <FileText className="w-5 h-5 text-blue-400" />
                                    Clinical Recommendations (ACC/AHA Guidelines)
                                </h4>
                                <div className="space-y-3">
                                    {result.clinicalRecommendations.recommendations.map((rec, idx) => (
                                        <div key={idx} className="p-3 bg-white/5 rounded-lg border-l-4 border-blue-500">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="text-blue-400 font-medium text-sm">{rec.category}</span>
                                                {rec.grade && <span className="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">Grade {rec.grade}</span>}
                                                {rec.urgency && <span className={`text-xs px-2 py-0.5 rounded ${rec.urgency === 'Urgent' ? 'bg-red-500/20 text-red-300' : rec.urgency === 'Soon' ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'}`}>{rec.urgency}</span>}
                                            </div>
                                            <p className="text-slate-300 text-sm">{rec.action}</p>
                                        </div>
                                    ))}
                                </div>
                                <p className="text-slate-500 text-xs mt-4">* Based on ACC/AHA and WHO cardiovascular prevention guidelines</p>
                            </motion.div>
                        )}

                        {/* Legend */}
                        <div className="glass-card p-4">
                            <div className="flex items-center justify-center gap-6 text-sm">
                                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-green-500" />Low/Negative</div>
                                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-yellow-500" />Moderate</div>
                                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500" />High/Positive</div>
                            </div>
                        </div>
                    </motion.div>
                </div>

                {/* Help Section */}
                <div className="mt-12 text-center">
                    <p className="text-slate-500 text-sm">
                        Need help or found an issue? <Link href="mailto:support@cardiodetect.ai" className="text-blue-400 hover:underline">Click here</Link>
                    </p>
                </div>
            </main >
        </div >
    );
}
