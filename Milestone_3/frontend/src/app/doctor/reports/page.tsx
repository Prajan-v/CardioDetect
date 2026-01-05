'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { FileText, Download, Calendar, User, TrendingUp, AlertTriangle, Check, RefreshCw, Search } from 'lucide-react';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';
import NotificationPopup, { showNotification } from '@/components/NotificationPopup';
import { getUser, getToken } from '@/services/auth';
import { useRouter } from 'next/navigation';
import { API_ENDPOINTS } from '@/services/apiClient';

interface Patient {
    id: string;
    name: string;
    email: string;
    risk_category: string | null;
    predictions_count: number;
}

interface Report {
    id: string;
    patient_name: string;
    patient_email: string;
    date: string;
    type: string;
    risk_level: string;
    risk_percentage?: number;
}

export default function DoctorReportsPage() {
    const router = useRouter();
    const user = getUser();
    const [reports, setReports] = useState<Report[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        // Check if doctor
        if (!user || user.role !== 'doctor') {
            router.push('/dashboard');
            return;
        }
        fetchDoctorPatients();
    }, [user, router]);

    const fetchDoctorPatients = async () => {
        try {
            const token = getToken();
            if (!token) {
                router.push('/login');
                return;
            }

            // Fetch doctor's dashboard data which includes their patients
            const res = await fetch(API_ENDPOINTS.doctor.dashboard(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                // API returns recent_patients from dashboard endpoint
                const patients: Patient[] = data.patients || data.recent_patients || [];

                // Convert patients to reports format
                const patientReports: Report[] = patients.map((patient, index) => ({
                    id: patient.id,
                    patient_name: patient.name,
                    patient_email: patient.email,
                    date: new Date().toISOString().split('T')[0], // Today's date
                    type: 'Risk Assessment',
                    risk_level: patient.risk_category || 'NO ASSESSMENT',
                    risk_percentage: undefined,
                }));

                setReports(patientReports);
            } else if (res.status === 403) {
                router.push('/dashboard');
            }
        } catch (err) {
            console.error('Failed to fetch patients:', err);
            showNotification({
                title: 'Error',
                message: 'Failed to load patient reports',
                type: 'error',
                duration: 5000,
            });
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    const handleRefresh = () => {
        setRefreshing(true);
        fetchDoctorPatients();
    };

    const getRiskStyle = (risk: string) => {
        switch (risk?.toUpperCase()) {
            case 'HIGH': return { text: 'text-red-400', bg: 'bg-red-500/20' };
            case 'MODERATE': return { text: 'text-yellow-400', bg: 'bg-yellow-500/20' };
            case 'LOW': return { text: 'text-green-400', bg: 'bg-green-500/20' };
            default: return { text: 'text-slate-400', bg: 'bg-slate-500/20' };
        }
    };

    const generatePDF = (report: Report) => {
        showNotification({
            title: 'Generating Report',
            message: `Creating PDF report for ${report.patient_name}...`,
            type: 'info',
            duration: 3000,
        });

        const printWindow = window.open('', '_blank');
        if (printWindow) {
            printWindow.document.write(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Medical Report - ${report.patient_name}</title>
                    <style>
                        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; color: #333; }
                        .header { text-align: center; margin-bottom: 40px; border-bottom: 3px solid #dc2626; padding-bottom: 20px; }
                        .logo { font-size: 24px; font-weight: bold; color: #dc2626; display: flex; align-items: center; justify-content: center; gap: 10px; }
                        h1 { color: #1e293b; margin: 10px 0; font-size: 28px; }
                        .meta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; background: #f8fafc; padding: 20px; border-radius: 8px; }
                        .meta-item { display: flex; flex-direction: column; }
                        .label { font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }
                        .value { font-size: 16px; font-weight: 500; color: #0f172a; }
                        .risk-badge { display: inline-block; padding: 8px 16px; border-radius: 50px; color: white; font-weight: bold; text-transform: uppercase; font-size: 14px; }
                        .risk-high { background-color: #ef4444; }
                        .risk-moderate { background-color: #eab308; }
                        .risk-low { background-color: #22c55e; }
                        .content { line-height: 1.6; }
                        .footer { margin-top: 50px; border-top: 1px solid #e2e8f0; padding-top: 20px; text-align: center; font-size: 12px; color: #94a3b8; }
                    </style>
                </head>
                <body>
                    <div class="header">
                        <div class="logo">❤️ CardioDetect</div>
                        <h1>Medical Report</h1>
                    </div>

                    <div class="meta-grid">
                        <div class="meta-item"><span class="label">Patient Name</span><span class="value">${report.patient_name}</span></div>
                        <div class="meta-item"><span class="label">Patient Email</span><span class="value">${report.patient_email}</span></div>
                        <div class="meta-item"><span class="label">Report Date</span><span class="value">${report.date}</span></div>
                        <div class="meta-item"><span class="label">Report Type</span><span class="value">${report.type}</span></div>
                    </div>

                    <div class="content">
                        <h3>Risk Assessment</h3>
                        <p>
                            Based on the analysis of the patient's vitals and risk factors, the cardiovascular risk level is:
                        </p>
                        <div style="text-align: center; margin: 30px 0;">
                            <span class="risk-badge risk-${report.risk_level === 'High Risk' ? 'high' : report.risk_level === 'Moderate Risk' ? 'moderate' : 'low'}">
                                ${report.risk_level.toUpperCase()}
                            </span>
                            ${report.risk_percentage ? `<div style="font-size: 24px; font-weight: bold; margin-top: 10px;">${report.risk_percentage.toFixed(1)}%</div>` : ''}
                        </div>
                    </div>

                    <div class="footer">
                        <p>Digitally signed by CardioDetect AI System</p>
                        <p>Consulting Physician: Dr. ${user?.full_name || 'Unknown'}</p>
                    </div>
                </body>
                </html>
            `);
            printWindow.document.close();
            printWindow.print();
        }
    };

    // Filter reports based on search
    const filteredReports = reports.filter(r =>
        r.patient_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        r.patient_email.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (loading) {
        return (
            <div className="min-h-screen bg-[#0a0a1a] flex items-center justify-center">
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

            {/* Header */}
            <nav className="relative z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <Link href="/doctor/dashboard" className="flex items-center gap-3">
                            <div className="w-8 h-8"><AnimatedHeart className="w-full h-full" /></div>
                            <span className="text-xl font-bold gradient-text">CardioDetect</span>
                        </Link>
                        <div className="flex items-center gap-4">
                            <motion.button
                                onClick={handleRefresh}
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                className={`p-2 rounded-lg bg-white/10 text-slate-400 hover:text-white ${refreshing ? 'animate-spin' : ''}`}
                            >
                                <RefreshCw className="w-5 h-5" />
                            </motion.button>
                            <Link href="/doctor/dashboard" className="text-slate-400 hover:text-white transition-colors">
                                ← Back to Dashboard
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            <main className="relative z-10 max-w-5xl mx-auto px-6 py-8">
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                    <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                        <FileText className="w-8 h-8 text-red-400" />
                        My Patient Reports
                    </h1>
                    <p className="text-slate-400 mt-1">View and download reports for your assigned patients only</p>
                </motion.div>

                {/* Search */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="mb-6">
                    <div className="relative max-w-md">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search patients..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-4 py-3 text-white placeholder-slate-500 focus:border-red-500/50 focus:outline-none"
                        />
                    </div>
                </motion.div>

                {/* Stats */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="grid grid-cols-3 gap-4 mb-8">
                    <div className="glass-card p-4 text-center">
                        <div className="text-2xl font-bold text-blue-400">{reports.length}</div>
                        <div className="text-slate-500 text-sm">My Patients</div>
                    </div>
                    <div className="glass-card p-4 text-center">
                        <div className="text-2xl font-bold text-green-400">{reports.filter(r => r.risk_level?.toUpperCase() === 'LOW').length}</div>
                        <div className="text-slate-500 text-sm">Low Risk</div>
                    </div>
                    <div className="glass-card p-4 text-center">
                        <div className="text-2xl font-bold text-red-400">{reports.filter(r => r.risk_level?.toUpperCase() === 'HIGH').length}</div>
                        <div className="text-slate-500 text-sm">High Risk</div>
                    </div>
                </motion.div>

                {/* Reports Table */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card overflow-hidden">
                    <div className="p-4 border-b border-white/10 flex items-center justify-between">
                        <h2 className="text-lg font-semibold text-white">Patient Reports</h2>
                        <span className="text-sm text-slate-500">Showing only your assigned patients</span>
                    </div>

                    {filteredReports.length === 0 ? (
                        <div className="p-12 text-center">
                            <User className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                            <p className="text-slate-400">
                                {searchQuery ? 'No patients match your search' : 'No patients assigned yet. Add patients from the dashboard.'}
                            </p>
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead className="bg-white/5">
                                    <tr>
                                        <th className="text-left px-4 py-3 text-slate-400 text-sm font-medium">Patient</th>
                                        <th className="text-left px-4 py-3 text-slate-400 text-sm font-medium">Type</th>
                                        <th className="text-left px-4 py-3 text-slate-400 text-sm font-medium">Risk Level</th>
                                        <th className="text-left px-4 py-3 text-slate-400 text-sm font-medium">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredReports.map((report) => {
                                        const style = getRiskStyle(report.risk_level);
                                        return (
                                            <tr key={report.id} className="border-t border-white/5 hover:bg-white/5 transition-colors">
                                                <td className="px-4 py-4">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center">
                                                            <User className="w-4 h-4 text-slate-400" />
                                                        </div>
                                                        <div>
                                                            <div className="text-white font-medium">{report.patient_name}</div>
                                                            <div className="text-slate-500 text-sm">{report.patient_email}</div>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td className="px-4 py-4 text-slate-300">{report.type}</td>
                                                <td className="px-4 py-4">
                                                    <span className={`px-3 py-1 rounded-full text-sm ${style.bg} ${style.text}`}>
                                                        {report.risk_level || 'No Assessment'}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-4">
                                                    <motion.button
                                                        onClick={() => generatePDF(report)}
                                                        whileHover={{ scale: 1.05 }}
                                                        whileTap={{ scale: 0.95 }}
                                                        className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 flex items-center gap-2"
                                                    >
                                                        <Download className="w-4 h-4" />PDF
                                                    </motion.button>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    )}
                </motion.div>

                {/* Help Section */}
                <div className="mt-12 text-center">
                    <p className="text-slate-500 text-sm">
                        Need help or found an issue? <Link href="mailto:cardiodetect.care@gmail.com" className="text-blue-400 hover:underline">Click here</Link>
                    </p>
                </div>
            </main>
        </div>
    );
}
