'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, User, Clock, AlertTriangle, CheckCircle, RefreshCw, Stethoscope } from 'lucide-react';

interface ActivityEntry {
    id: string;
    type: 'doctor' | 'patient';
    userName: string;
    userEmail?: string;
    riskCategory: string | null;
    riskPercentage: number | null;
    inputMethod: string;
    createdAt: string;
}

interface Patient {
    id: string;
    name: string;
    email: string;
    risk_category?: string | null;
    latest_prediction?: string | null;
}

interface ActivityFeedProps {
    doctorId?: string;
    patients?: Patient[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export default function ActivityFeed({ doctorId, patients = [] }: ActivityFeedProps) {
    const [activities, setActivities] = useState<ActivityEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<'all' | 'doctor' | 'patients'>('all');
    const [showAll, setShowAll] = useState(false);

    // Use JSON.stringify for stable dependency comparison to avoid infinite loops
    const patientsKey = JSON.stringify(patients.map(p => p.id));

    useEffect(() => {
        fetchActivities();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [doctorId, patientsKey]);

    const fetchActivities = async () => {
        setLoading(true);
        const token = localStorage.getItem('auth_token');
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const allActivities: ActivityEntry[] = [];

            // Fetch doctor's own prediction history
            const doctorRes = await fetch(`${API_BASE}/history/`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (doctorRes.ok) {
                const doctorData = await doctorRes.json();
                const doctorPredictions = doctorData.results || doctorData || [];
                doctorPredictions.forEach((p: {
                    id: string;
                    risk_category?: string;
                    risk_percentage?: number;
                    input_method?: string;
                    created_at: string;
                }) => {
                    allActivities.push({
                        id: p.id,
                        type: 'doctor',
                        userName: 'You',
                        riskCategory: p.risk_category || null,
                        riskPercentage: p.risk_percentage || null,
                        inputMethod: p.input_method || 'manual',
                        createdAt: p.created_at,
                    });
                });
            }

            // Fetch each patient's predictions
            for (const patient of patients.slice(0, 10)) { // Limit to first 10 patients
                try {
                    const patientRes = await fetch(`${API_BASE}/doctor/patients/${patient.id}/`, {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    if (patientRes.ok) {
                        const patientData = await patientRes.json();
                        const predictions = patientData.predictions || [];
                        predictions.forEach((p: {
                            id: string;
                            risk_category?: string;
                            risk_percentage?: number;
                            input_method?: string;
                            created_at: string;
                        }) => {
                            allActivities.push({
                                id: p.id,
                                type: 'patient',
                                userName: patient.name,
                                userEmail: patient.email,
                                riskCategory: p.risk_category || null,
                                riskPercentage: p.risk_percentage || null,
                                inputMethod: p.input_method || 'manual',
                                createdAt: p.created_at,
                            });
                        });
                    }
                } catch (e) {
                    console.warn(`Failed to fetch patient ${patient.id} activities:`, e);
                }
            }

            // Sort by date (newest first)
            allActivities.sort((a, b) =>
                new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
            );

            setActivities(allActivities);
        } catch (e) {
            console.error('Failed to fetch activities:', e);
        } finally {
            setLoading(false);
        }
    };

    const formatTime = (dateStr: string) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now.getTime() - date.getTime();
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    };

    const getRiskStyle = (risk: string | null) => {
        switch (risk?.toUpperCase()) {
            case 'HIGH': return { text: 'text-red-400', bg: 'bg-red-500/20', icon: AlertTriangle };
            case 'MODERATE': return { text: 'text-yellow-400', bg: 'bg-yellow-500/20', icon: AlertTriangle };
            case 'LOW': return { text: 'text-green-400', bg: 'bg-green-500/20', icon: CheckCircle };
            default: return { text: 'text-slate-400', bg: 'bg-slate-500/20', icon: Clock };
        }
    };

    const filteredActivities = activities.filter(a => {
        if (filter === 'all') return true;
        if (filter === 'doctor') return a.type === 'doctor';
        if (filter === 'patients') return a.type === 'patient';
        return true;
    });

    const displayedActivities = showAll ? filteredActivities : filteredActivities.slice(0, 5);

    if (loading) {
        return (
            <div className="glass-card p-8 text-center">
                <div className="w-10 h-10 border-4 border-red-500/30 border-t-red-500 rounded-full animate-spin mx-auto" />
                <p className="text-slate-400 mt-4">Loading activity feed...</p>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Filter Buttons */}
            <div className="flex items-center justify-between">
                <div className="flex gap-2">
                    {['all', 'doctor', 'patients'].map((f) => (
                        <button
                            key={f}
                            onClick={() => setFilter(f as typeof filter)}
                            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${filter === f
                                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                                : 'bg-white/5 text-slate-400 hover:text-white'
                                }`}
                        >
                            {f === 'all' ? 'All Activity' : f === 'doctor' ? 'My Activity' : 'Patient Activity'}
                        </button>
                    ))}
                </div>
                <button
                    onClick={fetchActivities}
                    className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                >
                    <RefreshCw className="w-4 h-4" />
                    <span className="text-sm">Refresh</span>
                </button>
            </div>

            {/* Activity List */}
            <div className="glass-card p-6">
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Activity className="w-5 h-5 text-blue-400" />
                    Recent Activity
                    <span className="text-xs text-slate-400 font-normal">({filteredActivities.length})</span>
                </h2>

                {filteredActivities.length === 0 ? (
                    <div className="text-center py-8">
                        <Clock className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                        <p className="text-slate-400">No activity found</p>
                    </div>
                ) : (
                    <>
                        <div className="space-y-3">
                            <AnimatePresence>
                                {displayedActivities.map((activity, index) => {
                                    const style = getRiskStyle(activity.riskCategory);
                                    const Icon = style.icon;

                                    return (
                                        <motion.div
                                            key={activity.id}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            exit={{ opacity: 0, y: -10 }}
                                            transition={{ delay: index * 0.03 }}
                                            className={`flex items-center gap-4 p-4 rounded-xl ${style.bg} border border-white/5`}
                                        >
                                            <div className={`w-10 h-10 rounded-full ${style.bg} flex items-center justify-center`}>
                                                {activity.type === 'doctor' ? (
                                                    <Stethoscope className={`w-5 h-5 ${style.text}`} />
                                                ) : (
                                                    <User className={`w-5 h-5 ${style.text}`} />
                                                )}
                                            </div>
                                            <div className="flex-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-white font-medium">{activity.userName}</span>
                                                    {activity.type === 'patient' && activity.userEmail && (
                                                        <span className="text-xs text-slate-500">({activity.userEmail})</span>
                                                    )}
                                                </div>
                                                <div className="flex items-center gap-2 text-sm">
                                                    <span className={style.text}>
                                                        {activity.riskCategory || 'Pending'}
                                                        {activity.riskPercentage !== null && ` (${activity.riskPercentage.toFixed(1)}%)`}
                                                    </span>
                                                    <span className="text-slate-500">•</span>
                                                    <span className="text-slate-500">{activity.inputMethod}</span>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-slate-400 text-sm">{formatTime(activity.createdAt)}</div>
                                                <div className={`text-xs px-2 py-0.5 rounded-full ${activity.type === 'doctor'
                                                    ? 'bg-blue-500/20 text-blue-400'
                                                    : 'bg-purple-500/20 text-purple-400'
                                                    }`}>
                                                    {activity.type === 'doctor' ? 'You' : 'Patient'}
                                                </div>
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </AnimatePresence>
                        </div>

                        {/* Show More/Less Button */}
                        {filteredActivities.length > 5 && (
                            <div className="mt-4 text-center">
                                <button
                                    onClick={() => setShowAll(!showAll)}
                                    className="px-6 py-2 bg-white/5 hover:bg-white/10 text-slate-300 hover:text-white rounded-xl transition-all flex items-center gap-2 mx-auto"
                                >
                                    {showAll ? 'Show Less' : `Show More (${filteredActivities.length - 5} more)`}
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
