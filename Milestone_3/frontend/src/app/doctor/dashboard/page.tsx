'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Heart, Activity, User, LogOut, FileUp,
    AlertTriangle, CheckCircle, Clock, Zap,
    ChevronDown, TrendingUp, Shield, Stethoscope, BarChart3, Info,
    Bell, Users, Plus, Search, Settings
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import AnimatedHeart from '@/components/AnimatedHeart';
import ECGLine from '@/components/ECGLine';
import FloatingParticles from '@/components/FloatingParticles';
import NotificationPopup, { showNotification } from '@/components/NotificationPopup';
import { getUser, logout } from '@/services/auth';
import ActivityFeed from '@/components/ActivityFeed';
import { API_ENDPOINTS } from '@/services/apiClient';

type TabType = 'assessment' | 'patients' | 'upload' | 'activity';

type AnalysisMode = 'detection' | 'prediction' | 'both';
type RiskLevel = 'low' | 'moderate' | 'high' | null;

interface DashboardData {
    doctor: { id?: string; name: string; specialization: string; hospital: string; };
    stats: { total_patients: number; total_predictions: number; predictions_this_week: number; high_risk_patients: number; };
    patients: Array<{ id: string; email: string; name: string; risk_category: string | null; predictions_count: number; }>;
    risk_distribution: { low: number; moderate: number; high: number; };
}


interface FormField {
    name: string; label: string; type: string; placeholder?: string;
    min?: number; max?: number; step?: number; required?: boolean;
    options?: { value: string; label: string }[]; helpText?: string;
}

// Risk configurations
const riskConfig = {
    low: { color: 'text-green-400', bgColor: 'bg-green-500/20', icon: CheckCircle },
    moderate: { color: 'text-yellow-400', bgColor: 'bg-yellow-500/20', icon: AlertTriangle },
    high: { color: 'text-red-400', bgColor: 'bg-red-500/20', icon: AlertTriangle },
};

// Detection form fields
const detectionFields: FormField[] = [
    { name: 'age', label: 'Age', type: 'number', placeholder: '45', min: 18, max: 120, required: true },
    { name: 'sex', label: 'Sex', type: 'select', options: [{ value: '1', label: 'Male' }, { value: '0', label: 'Female' }], required: true },
    { name: 'cp', label: 'Chest Pain', type: 'select', required: true, options: [{ value: '0', label: 'Typical Angina' }, { value: '1', label: 'Atypical' }, { value: '2', label: 'Non-anginal' }, { value: '3', label: 'Asymptomatic' }] },
    { name: 'trestbps', label: 'Resting BP', type: 'number', placeholder: '120', min: 60, max: 250, required: true },
    { name: 'chol', label: 'Cholesterol', type: 'number', placeholder: '200', min: 80, max: 600, required: true },
    { name: 'fbs', label: 'Fasting BS > 120', type: 'select', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }], required: true },
    { name: 'restecg', label: 'Resting ECG', type: 'select', options: [{ value: '0', label: 'Normal' }, { value: '1', label: 'ST-T Abnormality' }, { value: '2', label: 'LV Hypertrophy' }] },
    { name: 'thalach', label: 'Max Heart Rate', type: 'number', placeholder: '150', min: 60, max: 220, required: true },
    { name: 'exang', label: 'Exercise Angina', type: 'select', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }] },
    { name: 'oldpeak', label: 'ST Depression', type: 'number', placeholder: '1.0', min: 0, max: 10, step: 0.1 },
];

// Prediction form fields
const predictionFields: FormField[] = [
    { name: 'age', label: 'Age', type: 'number', placeholder: '45', min: 18, max: 120, required: true },
    { name: 'sex', label: 'Sex', type: 'select', options: [{ value: '1', label: 'Male' }, { value: '0', label: 'Female' }], required: true },
    { name: 'systolic_bp', label: 'Systolic BP', type: 'number', placeholder: '120', min: 80, max: 250, required: true },
    { name: 'cholesterol', label: 'Cholesterol', type: 'number', placeholder: '200', min: 100, max: 400, required: true },
    { name: 'hdl', label: 'HDL', type: 'number', placeholder: '50', min: 20, max: 100, required: true },
    { name: 'smoking', label: 'Smoker', type: 'select', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }], required: true },
    { name: 'diabetes', label: 'Diabetes', type: 'select', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }], required: true },
    { name: 'bp_medication', label: 'BP Medication', type: 'select', options: [{ value: '0', label: 'No' }, { value: '1', label: 'Yes' }] },
];

export default function DoctorDashboard() {
    const router = useRouter();
    const [activeTab, setActiveTab] = useState<TabType>('assessment');
    const [mode, setMode] = useState<AnalysisMode>('prediction');
    const [formData, setFormData] = useState<Record<string, string>>({});
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [result, setResult] = useState<{ risk: RiskLevel; percentage: number } | null>(null);
    const [showDropdown, setShowDropdown] = useState(false);
    const [selectedPatient, setSelectedPatient] = useState<string>('');

    // Dashboard data
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);
    const [newPatientEmail, setNewPatientEmail] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        fetchDashboard();
    }, []);

    const fetchDashboard = async () => {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) { router.push('/login'); return; }

            const res = await fetch(API_ENDPOINTS.doctor.dashboard(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.status === 403) { router.push('/dashboard'); return; }
            if (res.ok) {
                const dashboardData = await res.json();
                // Normalize: API returns recent_patients but frontend uses patients
                if (dashboardData.recent_patients && !dashboardData.patients) {
                    dashboardData.patients = dashboardData.recent_patients;
                }
                setData(dashboardData);
            }
        } catch (err) {
            console.error('Failed to load dashboard');
        } finally {
            setLoading(false);
        }
    };

    const addPatient = async () => {
        if (!newPatientEmail) return;
        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch(API_ENDPOINTS.doctor.patients(), {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: newPatientEmail })  // Backend expects 'email' not 'patient_email'
            });
            if (res.ok) {
                setNewPatientEmail('');
                fetchDashboard();
                showNotification({
                    title: 'Patient Added',
                    message: 'Patient has been successfully added to your list.',
                    type: 'success',
                    duration: 5000,
                });
            } else {
                const error = await res.json();
                showNotification({
                    title: 'Error',
                    message: error.error || 'Failed to add patient',
                    type: 'error',
                    duration: 5000,
                });
            }
        } catch (err) {
            console.error('Failed to add patient');
            showNotification({
                title: 'Error',
                message: 'Failed to add patient. Please try again.',
                type: 'error',
                duration: 5000,
            });
        }
    };

    // Auto-detect and convert units
    const autoConvertValue = (name: string, value: number): number => {
        // Cholesterol fields: if < 15, assume mmol/L and convert to mg/dL
        if (['chol', 'cholesterol', 'hdl'].includes(name)) {
            if (value < 15 && value > 0) {
                return Math.round(value * 38.67);
            }
        }
        return value;
    };

    // Check if a field value looks like mmol/L for UI feedback
    const isLikelyMmolL = (name: string, value: string): boolean => {
        const num = parseFloat(value);
        if (isNaN(num) || num <= 0) return false;
        if (['chol', 'cholesterol', 'hdl'].includes(name)) {
            return num < 15;
        }
        return false;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);

        try {
            // Process form data with unit conversion
            const processedData: Record<string, string | number | boolean> = { ...formData };

            // Convert numeric fields
            ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'systolic_bp', 'cholesterol', 'hdl', 'bmi'].forEach(field => {
                const fieldValue = processedData[field];
                if (fieldValue && typeof fieldValue === 'string') {
                    let val = parseFloat(fieldValue);
                    val = autoConvertValue(field, val); // Apply conversion
                    processedData[field] = val;
                }
            });

            const token = localStorage.getItem('auth_token');
            const res = await fetch(API_ENDPOINTS.predict.base(), {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...processedData, mode, patient_id: selectedPatient || undefined })
            });

            if (res.ok) {
                const data = await res.json();
                const risk = data.prediction?.risk_category?.toLowerCase() || 'moderate';
                setResult({ risk: risk as RiskLevel, percentage: data.prediction?.risk_percentage || 25 });
            }
        } catch (err) {
            // Fallback result
            setResult({ risk: 'moderate', percentage: 25 });
        } finally {
            setIsSubmitting(false);
        }

        // Show popup notification for results
        if (result) {
            const notifType = result.risk === 'high' ? 'warning' : result.risk === 'low' ? 'success' : 'info';
            showNotification({
                title: `${result.risk?.toUpperCase()} Risk Assessment`,
                message: `Patient risk level: ${result.percentage}%. ${result.risk === 'high' ? 'Urgent attention recommended.' : 'Continue monitoring.'}`,
                type: notifType,
                duration: 8000,
            });
        }
    };

    const currentFields = mode === 'detection' ? detectionFields : predictionFields;
    const filteredPatients = (data?.patients || []).filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.email.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const getRiskStyle = (risk: string | null) => {
        switch (risk?.toUpperCase()) {
            case 'HIGH': return { text: 'text-red-400', bg: 'bg-red-500/20', border: 'border-red-500/30' };
            case 'MODERATE': return { text: 'text-yellow-400', bg: 'bg-yellow-500/20', border: 'border-yellow-500/30' };
            case 'LOW': return { text: 'text-green-400', bg: 'bg-green-500/20', border: 'border-green-500/30' };
            default: return { text: 'text-slate-400', bg: 'bg-slate-500/20', border: 'border-slate-500/30' };
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-[#0a0a1a] flex items-center justify-center">
                <div className="fixed inset-0 mesh-bg pointer-events-none" />
                <div className="w-12 h-12 border-4 border-red-500/30 border-t-red-500 rounded-full animate-spin" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[#0a0a1a] relative">
            <NotificationPopup position="top-right" />
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Navigation */}
            <nav className="relative z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <Link href="/doctor/dashboard" className="flex items-center gap-3">
                            <div className="w-8 h-8"><AnimatedHeart className="w-full h-full" /></div>
                            <span className="text-xl font-bold gradient-text">CardioDetect</span>
                            <span className="text-slate-500 text-sm hidden sm:block">• Doctor Portal</span>
                        </Link>

                        <div className="relative">
                            <button onClick={() => setShowDropdown(!showDropdown)} className="flex items-center gap-2 glass-card px-4 py-2 hover:bg-white/10 transition-colors">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-red-500 to-purple-600 flex items-center justify-center">
                                    <Stethoscope className="w-4 h-4 text-white" />
                                </div>
                                <span className="text-white hidden sm:block">Dr. {data?.doctor.name || getUser()?.full_name || 'Doctor'}</span>
                                <ChevronDown className="w-4 h-4 text-slate-400" />
                            </button>
                            <AnimatePresence>
                                {showDropdown && (
                                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }} className="absolute right-0 mt-2 w-56 glass-card py-2 z-50">
                                        <Link href="/profile" onClick={() => setShowDropdown(false)} className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10">
                                            <User className="w-4 h-4" />Profile
                                        </Link>
                                        <Link href="/settings" onClick={() => setShowDropdown(false)} className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10">
                                            <Settings className="w-4 h-4" />Settings
                                        </Link>
                                        <Link href="/notifications" onClick={() => setShowDropdown(false)} className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10">
                                            <Bell className="w-4 h-4" />Notifications
                                        </Link>
                                        <Link href="/doctor/reports" onClick={() => setShowDropdown(false)} className="flex items-center gap-2 px-4 py-2 text-slate-300 hover:bg-white/10">
                                            <FileUp className="w-4 h-4" />Reports & PDFs
                                        </Link>
                                        <div className="border-t border-white/10 my-2" />
                                        <button onClick={() => { setShowDropdown(false); logout(); router.push('/login'); }} className="flex items-center gap-2 px-4 py-2 text-red-400 hover:bg-white/10 w-full text-left">
                                            <LogOut className="w-4 h-4" />Logout
                                        </button>
                                    </motion.div>

                                )}
                            </AnimatePresence>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
                {/* Welcome */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
                    <h1 className="text-3xl font-bold text-white mb-1">Welcome, Dr. {data?.doctor.name?.split(' ')[0] || 'Doctor'} 👋</h1>
                    <p className="text-slate-400">{data?.doctor.specialization} • {data?.doctor.hospital}</p>
                </motion.div>

                {/* Stats */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    {[
                        { icon: Users, label: 'Patients', value: data?.stats.total_patients || 0, color: 'text-blue-400' },
                        { icon: BarChart3, label: 'Predictions', value: data?.stats.total_predictions || 0, color: 'text-green-400' },
                        { icon: AlertTriangle, label: 'High Risk', value: data?.stats.high_risk_patients || 0, color: 'text-red-400' },
                        { icon: Clock, label: 'This Week', value: data?.stats.predictions_this_week || 0, color: 'text-purple-400' },
                    ].map((stat) => (
                        <div key={stat.label} className="glass-card p-4 text-center">
                            <stat.icon className={`w-6 h-6 ${stat.color} mx-auto mb-2`} />
                            <div className={`text-2xl font-bold ${stat.color}`}>{stat.value}</div>
                            <div className="text-slate-500 text-sm">{stat.label}</div>
                        </div>
                    ))}
                </motion.div>

                {/* Tabs */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="mb-6">
                    <div className="inline-flex p-1 glass-card rounded-2xl">
                        {[
                            { id: 'assessment', label: 'Risk Assessment', icon: Heart },
                            { id: 'activity', label: 'Activity Feed', icon: Activity },
                            { id: 'upload', label: 'Upload Document', icon: FileUp },
                            { id: 'patients', label: 'My Patients', icon: Users },
                        ].map((tab) => (

                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as TabType)}
                                className={`relative px-6 py-3 rounded-xl font-medium transition-all flex items-center gap-2 ${activeTab === tab.id ? 'text-white' : 'text-slate-400 hover:text-white'}`}
                            >
                                {activeTab === tab.id && (
                                    <motion.div layoutId="doctorTab" className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-purple-500/20 rounded-xl border border-red-500/30" />
                                )}
                                <span className="relative z-10 flex items-center gap-2">
                                    <tab.icon className="w-4 h-4" />{tab.label}
                                </span>
                            </button>
                        ))}
                    </div>
                </motion.div>

                {/* Tab Content */}
                <AnimatePresence mode="wait">
                    {activeTab === 'assessment' && (
                        <motion.div key="assessment" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 20 }}>
                            <div className="grid lg:grid-cols-2 gap-8">
                                {/* Form */}
                                <div className="glass-card p-6">
                                    <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                                        <Heart className="w-5 h-5 text-red-400" />
                                        {mode === 'detection' ? 'Heart Disease Detection' : '10-Year Risk Prediction'}
                                    </h2>

                                    {/* Mode Selector */}
                                    <div className="flex gap-2 mb-4">
                                        {(['detection', 'prediction', 'both'] as const).map((m) => (
                                            <button key={m} onClick={() => { setMode(m); setResult(null); }}
                                                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${mode === m ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-white/5 text-slate-400 hover:text-white'}`}>
                                                {m.charAt(0).toUpperCase() + m.slice(1)}
                                            </button>
                                        ))}
                                    </div>

                                    {/* Patient Selector */}
                                    {(data?.patients || []).length > 0 && (
                                        <div className="mb-4">
                                            <label className="block text-sm text-slate-400 mb-2">Run for Patient (optional)</label>
                                            <select value={selectedPatient} onChange={(e) => setSelectedPatient(e.target.value)}
                                                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white">
                                                <option value="">Self / Manual Entry</option>
                                                {(data?.patients || []).map(p => (
                                                    <option key={p.id} value={p.id}>{p.name} ({p.email})</option>
                                                ))}
                                            </select>
                                        </div>
                                    )}

                                    <form onSubmit={handleSubmit} className="space-y-4">
                                        <div className="grid grid-cols-2 gap-4">
                                            {currentFields.map((field) => (
                                                <div key={field.name}>
                                                    <label className="block text-sm text-slate-400 mb-1">
                                                        {field.label} {field.required && <span className="text-red-400">*</span>}
                                                    </label>
                                                    {field.type === 'select' ? (
                                                        <select value={formData[field.name] || ''} onChange={(e) => setFormData({ ...formData, [field.name]: e.target.value })}
                                                            className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white" required={field.required}>
                                                            <option value="">Select...</option>
                                                            {field.options?.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                                        </select>
                                                    ) : (
                                                        <div className="relative">
                                                            <input type={field.type} value={formData[field.name] || ''} onChange={(e) => setFormData({ ...formData, [field.name]: e.target.value })}
                                                                placeholder={field.placeholder} min={field.min} max={field.max} step={field.step}
                                                                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500" required={field.required} />
                                                            {isLikelyMmolL(field.name, formData[field.name] || '') && (
                                                                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400">
                                                                    mmol/L → mg/dL
                                                                </span>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>

                                        <motion.button type="submit" disabled={isSubmitting} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                                            className="w-full glow-button text-white py-4 rounded-xl font-semibold flex items-center justify-center gap-2 disabled:opacity-50">
                                            {isSubmitting ? <><div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />Analyzing...</> : <><Activity className="w-5 h-5" />Run Analysis</>}
                                        </motion.button>
                                    </form>
                                </div>

                                {/* Results */}
                                <div className="space-y-6">
                                    {result ? (
                                        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                                            className={`glass-card p-8 text-center ${riskConfig[result.risk || 'moderate'].bgColor} border ${result.risk === 'high' ? 'border-red-500/30' : result.risk === 'low' ? 'border-green-500/30' : 'border-yellow-500/30'}`}>
                                            <div className={`text-6xl font-bold ${riskConfig[result.risk || 'moderate'].color} mb-2`}>{result.percentage}%</div>
                                            <div className={`text-xl font-semibold ${riskConfig[result.risk || 'moderate'].color}`}>{result.risk?.toUpperCase()} RISK</div>
                                            <p className="text-slate-400 mt-2">10-year cardiovascular risk</p>
                                        </motion.div>
                                    ) : (
                                        <div className="glass-card p-8 text-center">
                                            <TrendingUp className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                                            <h3 className="text-xl font-semibold text-white mb-2">Ready to Analyze</h3>
                                            <p className="text-slate-400">Fill in patient data and click Run Analysis</p>
                                        </div>
                                    )}

                                    {/* Risk Legend */}
                                    <div className="glass-card p-4">
                                        <div className="flex justify-center gap-6 text-sm">
                                            <span className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-green-500" />Low Risk</span>
                                            <span className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-yellow-500" />Moderate</span>
                                            <span className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-red-500" />High Risk</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {activeTab === 'upload' && (
                        <motion.div key="upload" initial={{ opacity: 0, x: 0 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                            <div className="glass-card p-8">
                                <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-3">
                                    <FileUp className="w-6 h-6 text-red-400" />
                                    Upload Medical Document
                                </h2>
                                <p className="text-slate-400 mb-6">Upload a medical report image for automatic data extraction using OCR</p>

                                <Link href="/doctor/upload">
                                    <motion.div
                                        whileHover={{ scale: 1.01, borderColor: 'rgba(239, 68, 68, 0.5)' }}
                                        whileTap={{ scale: 0.99 }}
                                        className="border-2 border-dashed border-white/20 rounded-2xl p-12 text-center cursor-pointer hover:border-red-500/50 transition-all"
                                    >
                                        <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center mx-auto mb-4">
                                            <FileUp className="w-8 h-8 text-red-400" />
                                        </div>
                                        <h3 className="text-lg font-semibold text-white mb-2">Drag & Drop Medical Report</h3>
                                        <p className="text-slate-400 mb-4">or click to browse files</p>
                                        <div className="flex justify-center gap-4 text-sm text-slate-500">
                                            <span>📄 PDF</span>
                                            <span>🖼️ PNG</span>
                                            <span>🖼️ JPG</span>
                                        </div>
                                    </motion.div>
                                </Link>

                                <div className="mt-6 grid grid-cols-2 gap-4">
                                    <div className="glass-card p-4">
                                        <h4 className="text-white font-medium mb-2">Supported Reports</h4>
                                        <ul className="text-slate-400 text-sm space-y-1">
                                            <li>• Lab test results</li>
                                            <li>• Blood pressure reports</li>
                                            <li>• Cholesterol panels</li>
                                            <li>• ECG reports</li>
                                        </ul>
                                    </div>
                                    <div className="glass-card p-4">
                                        <h4 className="text-white font-medium mb-2">How It Works</h4>
                                        <ul className="text-slate-400 text-sm space-y-1">
                                            <li>1. Upload document</li>
                                            <li>2. AI extracts data</li>
                                            <li>3. Review & confirm</li>
                                            <li>4. Get risk assessment</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {activeTab === 'patients' && (

                        <motion.div key="patients" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                            {/* Add Patient */}
                            <div className="glass-card p-6 mb-6">
                                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <Plus className="w-5 h-5 text-green-400" />Add New Patient
                                </h2>
                                <div className="flex gap-3">
                                    <input type="email" placeholder="Enter patient email..." value={newPatientEmail} onChange={(e) => setNewPatientEmail(e.target.value)}
                                        className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500" />
                                    <motion.button onClick={addPatient} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} disabled={!newPatientEmail}
                                        className="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl font-semibold disabled:opacity-50">
                                        Add Patient
                                    </motion.button>
                                </div>
                            </div>

                            {/* Risk Distribution */}
                            <div className="grid grid-cols-3 gap-4 mb-6">
                                {[
                                    { label: 'Low Risk', value: data?.risk_distribution.low || 0, color: 'from-green-500/20 to-green-600/20', text: 'text-green-400', border: 'border-green-500/30' },
                                    { label: 'Moderate', value: data?.risk_distribution.moderate || 0, color: 'from-yellow-500/20 to-orange-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30' },
                                    { label: 'High Risk', value: data?.risk_distribution.high || 0, color: 'from-red-500/20 to-pink-500/20', text: 'text-red-400', border: 'border-red-500/30' },
                                ].map((item) => (
                                    <div key={item.label} className={`glass-card p-5 bg-gradient-to-br ${item.color} border ${item.border}`}>
                                        <div className={`text-3xl font-bold ${item.text}`}>{item.value}</div>
                                        <div className="text-slate-400 text-sm">{item.label}</div>
                                    </div>
                                ))}
                            </div>

                            {/* Patient List */}
                            <div className="glass-card p-6">
                                <div className="flex items-center justify-between mb-6">
                                    <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                                        <Users className="w-5 h-5 text-blue-400" />My Patients
                                    </h2>
                                    <div className="relative">
                                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                                        <input type="text" placeholder="Search..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
                                            className="bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-2 text-white placeholder-slate-500 w-48" />
                                    </div>
                                </div>

                                {filteredPatients.length === 0 ? (
                                    <div className="text-center py-12">
                                        <Users className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                                        <p className="text-slate-400">No patients yet. Add one above.</p>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        {filteredPatients.map((patient) => {
                                            const style = getRiskStyle(patient.risk_category);
                                            return (
                                                <motion.div key={patient.id} whileHover={{ scale: 1.01 }}
                                                    className={`flex items-center justify-between p-4 rounded-xl bg-white/5 border ${style.border} cursor-pointer hover:bg-white/10`}>
                                                    <div className="flex items-center gap-4">
                                                        <div className={`w-10 h-10 rounded-full ${style.bg} flex items-center justify-center`}>
                                                            <User className={`w-5 h-5 ${style.text}`} />
                                                        </div>
                                                        <div>
                                                            <div className="text-white font-medium">{patient.name}</div>
                                                            <div className="text-slate-500 text-sm">{patient.email}</div>
                                                        </div>
                                                    </div>
                                                    <div className="text-right">
                                                        <div className={`font-medium ${style.text}`}>{patient.risk_category || 'No Assessment'}</div>
                                                        <div className="text-slate-500 text-xs">{patient.predictions_count} predictions</div>
                                                    </div>
                                                </motion.div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {activeTab === 'activity' && (
                        <motion.div key="activity" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }}>
                            <div className="flex items-center gap-3 mb-6">
                                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                                    <Activity className="w-6 h-6 text-blue-400" />
                                </div>
                                <div>
                                    <h2 className="text-2xl font-bold text-white">Activity Dashboard</h2>
                                    <p className="text-sm text-slate-400">Your activity and patient prediction history</p>
                                </div>
                            </div>
                            <ActivityFeed doctorId={data?.doctor?.id} patients={data?.patients || []} />
                        </motion.div>
                    )}

                    {/* Help Section */}
                    <div className="mt-12 text-center">
                        <p className="text-slate-500 text-sm">
                            Need help or found an issue? <Link href="mailto:support@cardiodetect.ai" className="text-blue-400 hover:underline">Click here</Link>
                        </p>
                    </div>
                </AnimatePresence>
            </main>
        </div>
    );
}
