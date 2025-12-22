'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Users, Activity, TrendingUp, Clock,
    UserCheck, FileText, AlertTriangle, CheckCircle,
    RefreshCw
} from 'lucide-react';

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

export default function AdminPanel() {
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

    if (loading) {
        return (
            <div className="glass-card p-6 animate-pulse">
                <div className="h-6 bg-white/10 rounded w-1/3 mb-4" />
                <div className="grid grid-cols-3 gap-4">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="h-20 bg-white/10 rounded" />
                    ))}
                </div>
            </div>
        );
    }

    if (error || !stats) {
        return (
            <div className="glass-card p-6 text-center text-slate-400">
                <AlertTriangle className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
                <p>{error || 'No data available'}</p>
                <button
                    onClick={fetchStats}
                    className="mt-3 text-sm text-red-400 hover:text-red-300 flex items-center gap-1 mx-auto"
                >
                    <RefreshCw className="w-4 h-4" /> Retry
                </button>
            </div>
        );
    }

    const riskColors: Record<string, string> = {
        LOW: 'text-green-400',
        MODERATE: 'text-yellow-400',
        HIGH: 'text-red-400',
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
        >
            {/* Header */}
            <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-red-400" />
                    Admin Dashboard
                </h2>
                <button
                    onClick={fetchStats}
                    className="text-slate-400 hover:text-white transition-colors"
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div className="glass-card p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
                            <Users className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">{stats.system_stats.total_users}</p>
                            <p className="text-sm text-slate-400">Total Users</p>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
                            <UserCheck className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">{stats.system_stats.total_doctors}</p>
                            <p className="text-sm text-slate-400">Doctors</p>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-green-500/20 flex items-center justify-center">
                            <FileText className="w-5 h-5 text-green-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">{stats.system_stats.total_predictions}</p>
                            <p className="text-sm text-slate-400">Predictions</p>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-yellow-500/20 flex items-center justify-center">
                            <TrendingUp className="w-5 h-5 text-yellow-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">{stats.system_stats.predictions_today}</p>
                            <p className="text-sm text-slate-400">Today</p>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                            <Clock className="w-5 h-5 text-red-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">{stats.system_stats.predictions_this_week}</p>
                            <p className="text-sm text-slate-400">This Week</p>
                        </div>
                    </div>
                </div>

                <div className="glass-card p-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-cyan-500/20 flex items-center justify-center">
                            <Users className="w-5 h-5 text-cyan-400" />
                        </div>
                        <div>
                            <p className="text-2xl font-bold text-white">{stats.system_stats.total_patients}</p>
                            <p className="text-sm text-slate-400">Patients</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Risk Distribution */}
            <div className="glass-card p-4">
                <h3 className="text-sm font-semibold text-slate-400 mb-3">Risk Distribution</h3>
                <div className="flex gap-4">
                    {Object.entries(stats.risk_distribution).map(([risk, count]) => (
                        <div key={risk} className="flex items-center gap-2">
                            {risk === 'LOW' ? <CheckCircle className="w-4 h-4 text-green-400" /> :
                                risk === 'HIGH' ? <AlertTriangle className="w-4 h-4 text-red-400" /> :
                                    <Activity className="w-4 h-4 text-yellow-400" />}
                            <span className={`font-semibold ${riskColors[risk]}`}>{count}</span>
                            <span className="text-slate-500 text-sm">{risk}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Doctor Activity */}
            {stats.doctor_activity.length > 0 && (
                <div className="glass-card p-4">
                    <h3 className="text-sm font-semibold text-slate-400 mb-3">Doctor Activity (Last 7 Days)</h3>
                    <div className="space-y-2">
                        {stats.doctor_activity.map(doctor => (
                            <div key={doctor.id} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                                <div className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-sm font-bold text-purple-400">
                                        {doctor.name[0]}
                                    </div>
                                    <div>
                                        <p className="text-white text-sm font-medium">{doctor.name}</p>
                                        <p className="text-slate-500 text-xs">{doctor.email}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-white font-semibold">{doctor.predictions_this_week}</p>
                                    <p className="text-slate-500 text-xs">predictions</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Recent Predictions */}
            {stats.recent_predictions.length > 0 && (
                <div className="glass-card p-4">
                    <h3 className="text-sm font-semibold text-slate-400 mb-3">Recent Predictions</h3>
                    <div className="space-y-2 max-h-60 overflow-y-auto">
                        {stats.recent_predictions.slice(0, 5).map(pred => (
                            <div key={pred.id} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                                <div>
                                    <p className="text-white text-sm">{pred.user_name}</p>
                                    <p className="text-slate-500 text-xs">{pred.input_method} • {new Date(pred.created_at).toLocaleString()}</p>
                                </div>
                                <span className={`px-2 py-1 rounded text-xs font-semibold ${pred.risk_category === 'LOW' ? 'bg-green-500/20 text-green-400' :
                                        pred.risk_category === 'HIGH' ? 'bg-red-500/20 text-red-400' :
                                            'bg-yellow-500/20 text-yellow-400'
                                    }`}>
                                    {pred.risk_category}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </motion.div>
    );
}
