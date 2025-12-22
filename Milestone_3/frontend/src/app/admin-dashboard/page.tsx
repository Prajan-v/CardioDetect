'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Users, Activity, TrendingUp, Clock, Heart,
    UserCheck, FileText, AlertTriangle, CheckCircle,
    RefreshCw, ArrowLeft, BarChart3
} from 'lucide-react';
import Link from 'next/link';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';

interface AdminStats {
    system_stats: {
        total_users: number;
        total_doctors: number;
        total_patients: number;
        total_predictions: number;
        predictions_today: number;
        predictions_this_week: number;
    };
    risk_distribution: Record<string, number>;
    doctor_activity: Array<{
        id: string;
        name: string;
        email: string;
        last_login: string | null;
        predictions_this_week: number;
    }>;
    recent_predictions: Array<{
        id: string;
        user_name: string;
        user_role: string;
        risk_category: string;
        risk_percentage: number;
        input_method: string;
        created_at: string;
    }>;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export default function AdminDashboardPage() {
    const [stats, setStats] = useState<AdminStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchStats = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/admin/stats/`);
            if (res.ok) {
                const data = await res.json();
                setStats(data);
                setError(null);
            } else {
                setError('Failed to load admin stats');
            }
        } catch {
            setError('Network error - is backend running?');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStats();
    }, []);

    const riskColors: Record<string, string> = {
        LOW: 'text-green-400',
        MODERATE: 'text-yellow-400',
        HIGH: 'text-red-400',
    };

    return (
        <div className="min-h-screen bg-[#0a0a1a] relative">
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

                        <div className="flex items-center gap-4">
                            <Link href="/dashboard" className="text-slate-400 hover:text-white transition-colors flex items-center gap-2">
                                <ArrowLeft className="w-4 h-4" />
                                Back to Dashboard
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="relative z-10 max-w-6xl mx-auto px-6 py-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                                <BarChart3 className="w-8 h-8 text-red-400" />
                                Admin Dashboard
                            </h1>
                            <p className="text-slate-400 mt-1">System statistics and doctor activity</p>
                        </div>
                        <button
                            onClick={fetchStats}
                            disabled={loading}
                            className="glass-card px-4 py-2 text-slate-400 hover:text-white flex items-center gap-2 transition-colors"
                        >
                            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                            Refresh
                        </button>
                    </div>
                </motion.div>

                {loading ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                        {[1, 2, 3, 4, 5, 6].map(i => (
                            <div key={i} className="glass-card p-6 animate-pulse">
                                <div className="h-4 bg-white/10 rounded w-1/2 mb-4" />
                                <div className="h-8 bg-white/10 rounded w-1/3" />
                            </div>
                        ))}
                    </div>
                ) : error ? (
                    <div className="glass-card p-12 text-center">
                        <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-yellow-400" />
                        <p className="text-slate-400">{error}</p>
                        <button
                            onClick={fetchStats}
                            className="mt-4 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
                        >
                            Try Again
                        </button>
                    </div>
                ) : stats && (
                    <>
                        {/* Stats Grid */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                            className="grid grid-cols-2 md:grid-cols-3 gap-6 mb-8"
                        >
                            <div className="glass-card p-6">
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                                        <Users className="w-6 h-6 text-blue-400" />
                                    </div>
                                    <div>
                                        <p className="text-3xl font-bold text-white">{stats.system_stats.total_users}</p>
                                        <p className="text-slate-400">Total Users</p>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card p-6">
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-purple-500/20 flex items-center justify-center">
                                        <UserCheck className="w-6 h-6 text-purple-400" />
                                    </div>
                                    <div>
                                        <p className="text-3xl font-bold text-white">{stats.system_stats.total_doctors}</p>
                                        <p className="text-slate-400">Doctors</p>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card p-6">
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-green-500/20 flex items-center justify-center">
                                        <FileText className="w-6 h-6 text-green-400" />
                                    </div>
                                    <div>
                                        <p className="text-3xl font-bold text-white">{stats.system_stats.total_predictions}</p>
                                        <p className="text-slate-400">Total Predictions</p>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card p-6">
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-yellow-500/20 flex items-center justify-center">
                                        <TrendingUp className="w-6 h-6 text-yellow-400" />
                                    </div>
                                    <div>
                                        <p className="text-3xl font-bold text-white">{stats.system_stats.predictions_today}</p>
                                        <p className="text-slate-400">Today</p>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card p-6">
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
                                        <Clock className="w-6 h-6 text-red-400" />
                                    </div>
                                    <div>
                                        <p className="text-3xl font-bold text-white">{stats.system_stats.predictions_this_week}</p>
                                        <p className="text-slate-400">This Week</p>
                                    </div>
                                </div>
                            </div>

                            <div className="glass-card p-6">
                                <div className="flex items-center gap-4">
                                    <div className="w-12 h-12 rounded-xl bg-cyan-500/20 flex items-center justify-center">
                                        <Heart className="w-6 h-6 text-cyan-400" />
                                    </div>
                                    <div>
                                        <p className="text-3xl font-bold text-white">{stats.system_stats.total_patients}</p>
                                        <p className="text-slate-400">Patients</p>
                                    </div>
                                </div>
                            </div>
                        </motion.div>

                        {/* Risk Distribution */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="glass-card p-6 mb-8"
                        >
                            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Activity className="w-5 h-5 text-red-400" />
                                Risk Distribution
                            </h2>
                            <div className="flex gap-8">
                                {Object.entries(stats.risk_distribution).map(([risk, count]) => (
                                    <div key={risk} className="flex items-center gap-3">
                                        {risk === 'LOW' ? <CheckCircle className="w-6 h-6 text-green-400" /> :
                                            risk === 'HIGH' ? <AlertTriangle className="w-6 h-6 text-red-400" /> :
                                                <Activity className="w-6 h-6 text-yellow-400" />}
                                        <div>
                                            <span className={`text-2xl font-bold ${riskColors[risk]}`}>{count}</span>
                                            <p className="text-slate-500 text-sm">{risk} Risk</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </motion.div>

                        <div className="grid md:grid-cols-2 gap-8">
                            {/* Doctor Activity */}
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3 }}
                                className="glass-card p-6"
                            >
                                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <UserCheck className="w-5 h-5 text-purple-400" />
                                    Doctor Activity (Last 7 Days)
                                </h2>
                                {stats.doctor_activity.length > 0 ? (
                                    <div className="space-y-3">
                                        {stats.doctor_activity.map(doctor => (
                                            <div key={doctor.id} className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
                                                <div className="flex items-center gap-3">
                                                    <div className="w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center text-lg font-bold text-purple-400">
                                                        {doctor.name[0]}
                                                    </div>
                                                    <div>
                                                        <p className="text-white font-medium">{doctor.name}</p>
                                                        <p className="text-slate-500 text-sm">{doctor.email}</p>
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <p className="text-2xl font-bold text-white">{doctor.predictions_this_week}</p>
                                                    <p className="text-slate-500 text-xs">predictions</p>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-slate-500 text-center py-4">No doctor activity yet</p>
                                )}
                            </motion.div>

                            {/* Recent Predictions */}
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.4 }}
                                className="glass-card p-6"
                            >
                                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <FileText className="w-5 h-5 text-green-400" />
                                    Recent Predictions
                                </h2>
                                {stats.recent_predictions.length > 0 ? (
                                    <div className="space-y-3 max-h-80 overflow-y-auto">
                                        {stats.recent_predictions.map(pred => (
                                            <div key={pred.id} className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
                                                <div>
                                                    <p className="text-white font-medium">{pred.user_name}</p>
                                                    <p className="text-slate-500 text-sm">
                                                        {pred.input_method} • {new Date(pred.created_at).toLocaleDateString()}
                                                    </p>
                                                </div>
                                                <span className={`px-3 py-1 rounded-lg text-sm font-semibold ${pred.risk_category === 'LOW' ? 'bg-green-500/20 text-green-400' :
                                                    pred.risk_category === 'HIGH' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-yellow-500/20 text-yellow-400'
                                                    }`}>
                                                    {pred.risk_category}
                                                </span>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-slate-500 text-center py-4">No predictions yet</p>
                                )}
                            </motion.div>
                        </div>

                        {/* Help Section */}
                        <div className="mt-12 text-center">
                            <p className="text-slate-500 text-sm">
                                Need help or found an issue? <Link href="mailto:support@cardiodetect.ai" className="text-blue-400 hover:underline">Click here</Link>
                            </p>
                        </div>
                    </>
                )}
            </main>
        </div>
    );
}
