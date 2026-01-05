'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { API_ENDPOINTS } from '@/services/apiClient';

interface PendingChange {
    id: number;
    user_email: string;
    user_name: string;
    field: string;
    old_value: string;
    new_value: string;
    reason: string;
    created_at: string;
}

interface DeletionRequest {
    id: string;
    user_email: string;
    user_name: string;
    status: 'pending' | 'completed' | 'cancelled' | 'failed';
    reason: string;
    requested_at: string;
    scheduled_deletion_at: string | null;
    grace_period_remaining_hours: number;
    is_cancellable: boolean;
}

export default function AdminApprovalPage() {
    const router = useRouter();
    const [changes, setChanges] = useState<PendingChange[]>([]);
    const [deletionRequests, setDeletionRequests] = useState<DeletionRequest[]>([]);
    const [deletionCounts, setDeletionCounts] = useState({ pending: 0, completed: 0, cancelled: 0, total: 0 });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [processing, setProcessing] = useState<number | null>(null);
    const [activeTab, setActiveTab] = useState<'changes' | 'deletions'>('changes');

    useEffect(() => {
        fetchPendingChanges();
        fetchDeletionRequests();
    }, []);

    const fetchPendingChanges = async () => {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) {
                router.push('/login');
                return;
            }

            const res = await fetch(API_ENDPOINTS.auth.adminPendingChanges(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.status === 403) {
                setError('Admin access required');
                return;
            }

            if (res.ok) {
                const data = await res.json();
                setChanges(data.pending_changes);
            }
        } catch (err) {
            setError('Failed to load pending changes');
        } finally {
            setLoading(false);
        }
    };

    const fetchDeletionRequests = async () => {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) return;

            const res = await fetch(API_ENDPOINTS.auth.adminDeletionRequests(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                setDeletionRequests(data.deletion_requests || []);
                setDeletionCounts(data.counts || { pending: 0, completed: 0, cancelled: 0, total: 0 });
            }
        } catch (err) {
            console.error('Failed to fetch deletion requests:', err);
        }
    };

    const handleApprove = async (changeId: number) => {
        setProcessing(changeId);
        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch(API_ENDPOINTS.auth.adminApprove(changeId), {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ notes: 'Approved' })
            });

            if (res.ok) {
                setChanges(changes.filter(c => c.id !== changeId));
            }
        } catch (err) {
            console.error('Failed to approve:', err);
        } finally {
            setProcessing(null);
        }
    };

    const handleReject = async (changeId: number) => {
        const notes = prompt('Rejection reason:');
        if (!notes) return;

        setProcessing(changeId);
        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch(API_ENDPOINTS.auth.adminReject(changeId), {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ notes })
            });

            if (res.ok) {
                setChanges(changes.filter(c => c.id !== changeId));
            }
        } catch (err) {
            console.error('Failed to reject:', err);
        } finally {
            setProcessing(null);
        }
    };

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'pending':
                return <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs">⏳ Pending</span>;
            case 'completed':
                return <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded text-xs">🗑️ Deleted</span>;
            case 'cancelled':
                return <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs">↩️ Cancelled</span>;
            default:
                return <span className="px-2 py-1 bg-gray-500/20 text-gray-400 rounded text-xs">{status}</span>;
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
                <div className="text-white">Loading...</div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
            {/* Header */}
            <header className="border-b border-white/10 backdrop-blur-xl bg-black/20">
                <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <Link href="/doctor/dashboard" className="flex items-center gap-2">
                        <span className="text-2xl">🛡️</span>
                        <span className="text-xl font-bold text-white">Admin Panel</span>
                    </Link>
                    <nav className="flex items-center gap-4">
                        <Link href="/doctor/dashboard" className="text-gray-300 hover:text-white">Dashboard</Link>
                        <Link href="/settings" className="text-gray-300 hover:text-white">Settings</Link>
                    </nav>
                </div>
            </header>

            <main className="max-w-6xl mx-auto px-4 py-8">
                {/* Tab Navigation */}
                <div className="flex gap-4 mb-8">
                    <button
                        onClick={() => setActiveTab('changes')}
                        className={`px-6 py-3 rounded-xl font-semibold transition-all ${activeTab === 'changes'
                            ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                            : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
                            }`}
                    >
                        📋 Profile Changes ({changes.length})
                    </button>
                    <button
                        onClick={() => setActiveTab('deletions')}
                        className={`px-6 py-3 rounded-xl font-semibold transition-all ${activeTab === 'deletions'
                            ? 'bg-red-500/30 text-red-300 border border-red-500/50'
                            : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
                            }`}
                    >
                        🗑️ Deletion Requests ({deletionCounts.pending})
                    </button>
                </div>

                {error && (
                    <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-lg text-red-300">
                        {error}
                    </div>
                )}

                {/* Profile Changes Tab */}
                {activeTab === 'changes' && (
                    <>
                        <h1 className="text-3xl font-bold text-white mb-2">📋 Pending Profile Changes</h1>
                        <p className="text-gray-400 mb-8">Review and approve/reject user profile change requests</p>

                        {changes.length === 0 ? (
                            <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-12 text-center border border-white/10">
                                <span className="text-4xl mb-4 block">✅</span>
                                <p className="text-gray-400">No pending changes to review</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {changes.map((change) => (
                                    <div key={change.id} className="bg-white/5 backdrop-blur-xl rounded-xl p-6 border border-white/10">
                                        <div className="flex items-start justify-between">
                                            <div>
                                                <div className="flex items-center gap-2 mb-2">
                                                    <span className="text-white font-semibold">{change.user_name}</span>
                                                    <span className="text-gray-500 text-sm">({change.user_email})</span>
                                                </div>
                                                <div className="text-gray-400 text-sm mb-3">
                                                    Requested: {new Date(change.created_at).toLocaleString()}
                                                </div>

                                                <div className="bg-black/30 rounded-lg p-4 mb-3">
                                                    <div className="text-gray-500 text-sm mb-1">Field: <span className="text-purple-400">{change.field}</span></div>
                                                    <div className="grid grid-cols-2 gap-4">
                                                        <div>
                                                            <div className="text-gray-500 text-xs">Old Value</div>
                                                            <div className="text-red-400">{change.old_value || '(empty)'}</div>
                                                        </div>
                                                        <div>
                                                            <div className="text-gray-500 text-xs">New Value</div>
                                                            <div className="text-green-400">{change.new_value}</div>
                                                        </div>
                                                    </div>
                                                </div>

                                                {change.reason && (
                                                    <div className="text-gray-400 text-sm">
                                                        <span className="text-gray-500">Reason:</span> {change.reason}
                                                    </div>
                                                )}
                                            </div>

                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => handleApprove(change.id)}
                                                    disabled={processing === change.id}
                                                    className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 disabled:opacity-50"
                                                >
                                                    {processing === change.id ? '...' : '✓ Approve'}
                                                </button>
                                                <button
                                                    onClick={() => handleReject(change.id)}
                                                    disabled={processing === change.id}
                                                    className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 disabled:opacity-50"
                                                >
                                                    {processing === change.id ? '...' : '✗ Reject'}
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}

                {/* Deletion Requests Tab */}
                {activeTab === 'deletions' && (
                    <>
                        <h1 className="text-3xl font-bold text-white mb-2">🗑️ Data Deletion Requests</h1>
                        <p className="text-gray-400 mb-8">Monitor GDPR data deletion requests (7-day grace period)</p>

                        {/* Stats */}
                        <div className="grid grid-cols-4 gap-4 mb-8">
                            <div className="bg-white/5 backdrop-blur-xl rounded-xl p-4 border border-white/10 text-center">
                                <div className="text-2xl font-bold text-yellow-400">{deletionCounts.pending}</div>
                                <div className="text-gray-400 text-sm">Pending</div>
                            </div>
                            <div className="bg-white/5 backdrop-blur-xl rounded-xl p-4 border border-white/10 text-center">
                                <div className="text-2xl font-bold text-red-400">{deletionCounts.completed}</div>
                                <div className="text-gray-400 text-sm">Completed</div>
                            </div>
                            <div className="bg-white/5 backdrop-blur-xl rounded-xl p-4 border border-white/10 text-center">
                                <div className="text-2xl font-bold text-green-400">{deletionCounts.cancelled}</div>
                                <div className="text-gray-400 text-sm">Cancelled</div>
                            </div>
                            <div className="bg-white/5 backdrop-blur-xl rounded-xl p-4 border border-white/10 text-center">
                                <div className="text-2xl font-bold text-blue-400">{deletionCounts.total}</div>
                                <div className="text-gray-400 text-sm">Total</div>
                            </div>
                        </div>

                        {deletionRequests.length === 0 ? (
                            <div className="bg-white/5 backdrop-blur-xl rounded-2xl p-12 text-center border border-white/10">
                                <span className="text-4xl mb-4 block">✅</span>
                                <p className="text-gray-400">No deletion requests yet</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {deletionRequests.map((req) => (
                                    <div key={req.id} className="bg-white/5 backdrop-blur-xl rounded-xl p-6 border border-white/10">
                                        <div className="flex items-start justify-between">
                                            <div>
                                                <div className="flex items-center gap-3 mb-2">
                                                    <span className="text-white font-semibold">{req.user_name}</span>
                                                    <span className="text-gray-500 text-sm">({req.user_email})</span>
                                                    {getStatusBadge(req.status)}
                                                </div>
                                                <div className="text-gray-400 text-sm mb-2">
                                                    Requested: {new Date(req.requested_at).toLocaleString()}
                                                </div>
                                                {req.scheduled_deletion_at && req.status === 'pending' && (
                                                    <div className="text-yellow-400 text-sm mb-2">
                                                        ⏰ Scheduled deletion: {new Date(req.scheduled_deletion_at).toLocaleString()}
                                                        <span className="text-gray-500 ml-2">
                                                            ({Math.round(req.grace_period_remaining_hours)} hours remaining)
                                                        </span>
                                                    </div>
                                                )}
                                                {req.reason && (
                                                    <div className="text-gray-400 text-sm">
                                                        <span className="text-gray-500">Reason:</span> {req.reason}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="text-right text-gray-500 text-sm">
                                                ID: {req.id.slice(0, 8)}...
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </main>
        </div>
    );
}

