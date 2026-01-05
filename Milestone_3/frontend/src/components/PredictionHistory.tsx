'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { History, Trash2, ChevronRight, AlertTriangle, CheckCircle, Clock, RefreshCw } from 'lucide-react';

export interface HistoryEntry {
    id: string;
    timestamp: number;
    mode: 'detection' | 'prediction' | 'both';
    detectionResult?: 'positive' | 'negative';
    detectionProbability?: number;
    riskLevel?: 'low' | 'moderate' | 'high';
    riskPercentage?: number;
    formData: Record<string, string>;
}

const STORAGE_KEY = 'cardiodetect_history';
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Storage helpers for localStorage fallback
export function saveToHistory(entry: Omit<HistoryEntry, 'id'>): void {
    try {
        const history = getLocalHistory();
        const newEntry: HistoryEntry = {
            ...entry,
            id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        };
        history.unshift(newEntry);
        // Keep only last 20 entries
        const trimmed = history.slice(0, 20);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
    } catch (e) {
        console.warn('Failed to save history:', e);
    }
}

export function getLocalHistory(): HistoryEntry[] {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    } catch (e) {
        return [];
    }
}

// Legacy export for backwards compatibility
export function getHistory(): HistoryEntry[] {
    return getLocalHistory();
}

export function clearHistory(): void {
    localStorage.removeItem(STORAGE_KEY);
}

// Fetch history from backend API (user-scoped)
async function fetchHistoryFromAPI(): Promise<HistoryEntry[]> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;
    if (!token) return [];

    try {
        const res = await fetch(`${API_BASE}/history/`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (!res.ok) return [];

        const data = await res.json();
        const results = data.results || data || [];

        return results.map((p: {
            id: string;
            created_at: string;
            model_used?: string;
            risk_category?: string;
            risk_percentage?: number;
            detection_result?: boolean;
            detection_probability?: number;
            input_data?: Record<string, string>;
        }) => ({
            id: p.id,
            timestamp: new Date(p.created_at).getTime(),
            mode: (p.model_used as HistoryEntry['mode']) || 'prediction',
            riskLevel: p.risk_category?.toLowerCase() as HistoryEntry['riskLevel'],
            riskPercentage: p.risk_percentage,
            detectionResult: p.detection_result !== undefined
                ? (p.detection_result ? 'positive' : 'negative')
                : undefined,
            detectionProbability: p.detection_probability,
            formData: p.input_data || {},
        }));
    } catch (e) {
        console.warn('Failed to fetch history from API:', e);
        return [];
    }
}

interface PredictionHistoryProps {
    onSelectEntry?: (entry: HistoryEntry) => void;
    maxItems?: number;
    useBackend?: boolean;  // Whether to fetch from backend API (default: true for authenticated users)
}

export default function PredictionHistory({
    onSelectEntry,
    maxItems = 5,
    useBackend = true,
}: PredictionHistoryProps) {
    const [history, setHistory] = useState<HistoryEntry[]>([]);
    const [showAll, setShowAll] = useState(false);
    const [loading, setLoading] = useState(false);
    const [dataSource, setDataSource] = useState<'api' | 'local'>('local');

    const loadHistory = useCallback(async () => {
        setLoading(true);

        // Check if user is authenticated
        const token = typeof window !== 'undefined' ? localStorage.getItem('auth_token') : null;

        if (useBackend && token) {
            // Try to fetch from backend API first
            const apiHistory = await fetchHistoryFromAPI();
            if (apiHistory.length > 0) {
                setHistory(apiHistory);
                setDataSource('api');
                setLoading(false);
                return;
            }
        }

        // Fallback to localStorage
        setHistory(getLocalHistory());
        setDataSource('local');
        setLoading(false);
    }, [useBackend]);

    useEffect(() => {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        loadHistory();

        // Listen for storage changes from other tabs/components
        const handleStorage = () => loadHistory();
        window.addEventListener('historyUpdate', handleStorage);
        return () => window.removeEventListener('historyUpdate', handleStorage);
    }, [loadHistory]);

    const handleClear = () => {
        clearHistory();
        setHistory([]);
    };

    const handleRefresh = () => {
        loadHistory();
    };

    const formatTime = (timestamp: number) => {
        const date = new Date(timestamp);
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

    const getRiskConfig = (entry: HistoryEntry) => {
        // For detection mode, prioritize detection result
        if (entry.mode === 'detection' && entry.detectionResult) {
            return entry.detectionResult === 'positive'
                ? { color: 'text-red-400', bg: 'bg-red-500/20', icon: AlertTriangle, label: 'Disease Detected' }
                : { color: 'text-green-400', bg: 'bg-green-500/20', icon: CheckCircle, label: 'No Disease' };
        }

        // For prediction mode, use risk level
        if (entry.mode === 'prediction' && entry.riskLevel) {
            switch (entry.riskLevel) {
                case 'low': return { color: 'text-green-400', bg: 'bg-green-500/20', icon: CheckCircle, label: 'Low Risk' };
                case 'moderate': return { color: 'text-yellow-400', bg: 'bg-yellow-500/20', icon: AlertTriangle, label: 'Moderate Risk' };
                case 'high': return { color: 'text-red-400', bg: 'bg-red-500/20', icon: AlertTriangle, label: 'High Risk' };
            }
        }

        // For 'both' mode - prioritize detection result for color, show combined label
        if (entry.mode === 'both') {
            const isPositive = entry.detectionResult === 'positive' || entry.riskLevel === 'high';
            const isModerate = entry.riskLevel === 'moderate';

            if (isPositive) {
                return { color: 'text-red-400', bg: 'bg-red-500/20', icon: AlertTriangle, label: 'Both Analysis' };
            } else if (isModerate) {
                return { color: 'text-yellow-400', bg: 'bg-yellow-500/20', icon: AlertTriangle, label: 'Both Analysis' };
            } else {
                return { color: 'text-green-400', bg: 'bg-green-500/20', icon: CheckCircle, label: 'Both Analysis' };
            }
        }

        // Fallback: check if riskLevel exists
        if (entry.riskLevel) {
            switch (entry.riskLevel) {
                case 'low': return { color: 'text-green-400', bg: 'bg-green-500/20', icon: CheckCircle, label: 'Low Risk' };
                case 'moderate': return { color: 'text-yellow-400', bg: 'bg-yellow-500/20', icon: AlertTriangle, label: 'Moderate Risk' };
                case 'high': return { color: 'text-red-400', bg: 'bg-red-500/20', icon: AlertTriangle, label: 'High Risk' };
            }
        }

        // Fallback: check detection result
        if (entry.detectionResult) {
            return entry.detectionResult === 'positive'
                ? { color: 'text-red-400', bg: 'bg-red-500/20', icon: AlertTriangle, label: 'Disease Detected' }
                : { color: 'text-green-400', bg: 'bg-green-500/20', icon: CheckCircle, label: 'No Disease' };
        }

        return { color: 'text-slate-400', bg: 'bg-slate-500/20', icon: Clock, label: 'Unknown' };
    };

    const displayedHistory = showAll ? history : history.slice(0, maxItems);

    if (loading) {
        return (
            <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <History className="w-5 h-5 text-blue-400" />
                    Prediction History
                </h3>
                <div className="flex justify-center py-6">
                    <div className="w-6 h-6 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin" />
                </div>
            </div>
        );
    }

    if (history.length === 0) {
        return (
            <div className="glass-card p-6">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <History className="w-5 h-5 text-blue-400" />
                    Prediction History
                </h3>
                <div className="text-center py-6 text-slate-400">
                    <Clock className="w-10 h-10 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No predictions yet</p>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <History className="w-5 h-5 text-blue-400" />
                    History
                    <span className="text-xs text-slate-400 font-normal">({history.length})</span>
                    {dataSource === 'api' && (
                        <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">
                            Synced
                        </span>
                    )}
                </h3>
                <div className="flex items-center gap-2">
                    {useBackend && (
                        <button
                            onClick={handleRefresh}
                            className="text-slate-400 hover:text-blue-400 transition-colors p-1"
                            title="Refresh from server"
                        >
                            <RefreshCw className="w-4 h-4" />
                        </button>
                    )}
                    <button
                        onClick={handleClear}
                        className="text-slate-400 hover:text-red-400 transition-colors p-1"
                        title="Clear history"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            </div>

            <div className="space-y-2">
                <AnimatePresence>
                    {displayedHistory.map((entry, index) => {
                        const config = getRiskConfig(entry);
                        const Icon = config.icon;

                        return (
                            <motion.button
                                key={entry.id}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: 20 }}
                                transition={{ delay: index * 0.05 }}
                                onClick={() => onSelectEntry?.(entry)}
                                className={`w-full flex items-center gap-3 p-3 rounded-xl ${config.bg} hover:bg-opacity-40 transition-all group`}
                            >
                                <div className={`w-8 h-8 rounded-lg ${config.bg} flex items-center justify-center`}>
                                    <Icon className={`w-4 h-4 ${config.color}`} />
                                </div>
                                <div className="flex-1 text-left">
                                    <div className={`text-sm font-medium ${config.color}`}>
                                        {config.label}
                                        {/* Show appropriate percentage based on mode */}
                                        {entry.mode === 'detection' && entry.detectionProbability !== undefined && ` (${entry.detectionProbability.toFixed(1)}%)`}
                                        {entry.mode === 'prediction' && entry.riskPercentage !== undefined && ` (${entry.riskPercentage.toFixed(1)}%)`}
                                        {entry.mode === 'both' && entry.riskPercentage !== undefined && ` (${entry.riskPercentage.toFixed(1)}% risk)`}
                                    </div>
                                    <div className="text-xs text-slate-500">
                                        {formatTime(entry.timestamp)} • {entry.mode}
                                    </div>
                                </div>
                                <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-white transition-colors" />
                            </motion.button>
                        );
                    })}
                </AnimatePresence>
            </div>

            {history.length > maxItems && (
                <button
                    onClick={() => setShowAll(!showAll)}
                    className="w-full mt-3 text-sm text-slate-400 hover:text-white transition-colors"
                >
                    {showAll ? 'Show less' : `Show all (${history.length})`}
                </button>
            )}
        </div>
    );
}
