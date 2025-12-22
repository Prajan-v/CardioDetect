'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { User, Lock, Bell, Settings, ChevronRight, Check, X, Clock, AlertCircle, RefreshCw, Shield, Download, Trash2, FileText } from 'lucide-react';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';
import { getUser } from '@/services/auth';
import { API_ENDPOINTS } from '@/services/apiClient';

interface PendingChange {
    id: number;
    field: string;
    old_value: string;
    new_value: string;
    status: 'PENDING' | 'APPROVED' | 'REJECTED';
    admin_notes: string | null;
    created_at: string;
}

export default function SettingsPage() {
    const router = useRouter();
    const [activeTab, setActiveTab] = useState('profile');
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState({ type: '', text: '' });
    const [pendingChanges, setPendingChanges] = useState<PendingChange[]>([]);
    const [refreshing, setRefreshing] = useState(false);
    const user = getUser();



    // Profile state
    const [profile, setProfile] = useState({
        first_name: '',
        last_name: '',
        email: '',
        phone: '',
    });
    const [originalProfile, setOriginalProfile] = useState({
        first_name: '',
        last_name: '',
        email: '',
        phone: '',
    });

    // Password state
    const [passwords, setPasswords] = useState({
        current_password: '',
        new_password: '',
        confirm_password: '',
    });

    // Notification settings
    const [notifications, setNotifications] = useState({
        predictions: true,
        weekly: false,
        security: true,
    });

    // GDPR/Privacy state
    const [deletionRequest, setDeletionRequest] = useState<{
        has_pending_request: boolean;
        request_id?: string;
        scheduled_deletion_at?: string;
        grace_period_remaining_hours?: number;
        is_cancellable?: boolean;
    } | null>(null);
    const [exportLoading, setExportLoading] = useState(false);
    const [deletionLoading, setDeletionLoading] = useState(false);

    const fetchDeletionStatus = useCallback(async () => {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) return;

            const res = await fetch('http://localhost:8000/api/auth/data-deletion/', {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                setDeletionRequest(data);
            }
        } catch (error) {
            console.error('Failed to fetch deletion status:', error);
        }
    }, []);

    const handleDataExport = async () => {
        setExportLoading(true);
        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch('http://localhost:8000/api/auth/data-export/', {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `cardiodetect_data_export_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                setMessage({ type: 'success', text: 'Your data has been downloaded successfully!' });
            } else {
                setMessage({ type: 'error', text: 'Failed to export data. Please try again.' });
            }
        } catch (error) {
            setMessage({ type: 'error', text: 'Network error. Please check your connection.' });
        } finally {
            setExportLoading(false);
        }
    };

    const handleDeletionRequest = async () => {
        if (!confirm('Are you sure you want to request account deletion? You will have 7 days to cancel this request.')) {
            return;
        }
        setDeletionLoading(true);
        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch('http://localhost:8000/api/auth/data-deletion/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ reason: 'User requested account deletion' })
            });

            if (res.ok) {
                setMessage({ type: 'success', text: 'Deletion request created. Your data will be deleted in 7 days.' });
                fetchDeletionStatus();
            } else {
                const data = await res.json();
                setMessage({ type: 'error', text: data.error || 'Failed to create deletion request.' });
            }
        } catch (error) {
            setMessage({ type: 'error', text: 'Network error. Please check your connection.' });
        } finally {
            setDeletionLoading(false);
        }
    };

    const handleCancelDeletion = async () => {
        setDeletionLoading(true);
        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch('http://localhost:8000/api/auth/data-deletion/', {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                setMessage({ type: 'success', text: 'Deletion request cancelled successfully!' });
                fetchDeletionStatus();
            } else {
                setMessage({ type: 'error', text: 'Failed to cancel deletion request.' });
            }
        } catch (error) {
            setMessage({ type: 'error', text: 'Network error. Please check your connection.' });
        } finally {
            setDeletionLoading(false);
        }
    };

    const fetchProfile = useCallback(async () => {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) {
                router.push('/login');
                return;
            }

            const res = await fetch(API_ENDPOINTS.auth.profile(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                const profileData = {
                    first_name: data.first_name || '',
                    last_name: data.last_name || '',
                    email: data.email || '',
                    phone: data.phone || '',
                };
                setProfile(profileData);
                setOriginalProfile(profileData);
            }
        } catch (error) {
            console.error('Failed to fetch profile:', error);
        }
    }, [router]);



    const fetchPendingChanges = useCallback(async () => {
        try {
            const token = localStorage.getItem('auth_token');
            if (!token) return;

            const res = await fetch(API_ENDPOINTS.auth.profileChanges(), {
                headers: { 'Authorization': `Bearer ${token}` }
            });

            if (res.ok) {
                const data = await res.json();
                setPendingChanges(data.changes || []);
            }
        } catch (error) {
            console.error('Failed to fetch pending changes:', error);
        }
    }, []);

    useEffect(() => {
        fetchProfile();
        fetchPendingChanges();
        fetchDeletionStatus();

        // Poll for updates every 10 seconds
        const interval = setInterval(() => {
            fetchPendingChanges();
        }, 10000);

        return () => clearInterval(interval);
    }, [fetchProfile, fetchPendingChanges, fetchDeletionStatus]);

    const refreshData = async () => {
        setRefreshing(true);
        await Promise.all([fetchProfile(), fetchPendingChanges()]);
        setTimeout(() => setRefreshing(false), 500);
    };

    const submitForApproval = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setMessage({ type: '', text: '' });

        // Find changed fields
        const changes: { field: string; old_value: string; new_value: string }[] = [];

        if (profile.first_name !== originalProfile.first_name) {
            changes.push({ field: 'first_name', old_value: originalProfile.first_name, new_value: profile.first_name });
        }
        if (profile.last_name !== originalProfile.last_name) {
            changes.push({ field: 'last_name', old_value: originalProfile.last_name, new_value: profile.last_name });
        }
        if (profile.phone !== originalProfile.phone) {
            changes.push({ field: 'phone', old_value: originalProfile.phone, new_value: profile.phone });
        }

        if (changes.length === 0) {
            setMessage({ type: 'info', text: 'No changes to submit' });
            setLoading(false);
            return;
        }

        try {
            const token = localStorage.getItem('auth_token');

            // Submit each change for approval
            let successCount = 0;
            for (const change of changes) {
                const res = await fetch(API_ENDPOINTS.auth.profileChangesSubmit(), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        field_name: change.field,
                        new_value: change.new_value,
                        reason: 'Profile update request'
                    })
                });
                if (res.ok) {
                    successCount++;
                } else {
                    const errorData = await res.json().catch(() => ({}));
                    console.error('Failed to submit change:', errorData);
                }
            }

            if (successCount > 0) {
                setMessage({ type: 'success', text: `${successCount} change(s) submitted for admin approval!` });
                fetchPendingChanges();
            } else {
                setMessage({ type: 'error', text: 'Failed to submit changes. Please try again.' });
            }
        } catch (error) {
            console.error('Submit error:', error);
            setMessage({ type: 'error', text: 'Network error. Please check your connection and try again.' });
        } finally {
            setLoading(false);
        }
    };


    const changePassword = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setMessage({ type: '', text: '' });

        if (passwords.new_password !== passwords.confirm_password) {
            setMessage({ type: 'error', text: 'Passwords do not match' });
            setLoading(false);
            return;
        }

        if (passwords.new_password.length < 8) {
            setMessage({ type: 'error', text: 'Password must be at least 8 characters' });
            setLoading(false);
            return;
        }

        try {
            const token = localStorage.getItem('auth_token');
            const res = await fetch(API_ENDPOINTS.auth.passwordChange(), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    current_password: passwords.current_password,
                    new_password: passwords.new_password,
                    new_password_confirm: passwords.confirm_password
                })
            });

            if (res.ok) {
                setMessage({ type: 'success', text: 'Password changed successfully!' });
                setPasswords({ current_password: '', new_password: '', confirm_password: '' });
            } else {
                const data = await res.json();
                // Parse various error formats from backend
                let errorText = 'Failed to change password';
                if (data.detail) {
                    errorText = data.detail;
                } else if (data.errors) {
                    // Extract first error message from errors object
                    const firstKey = Object.keys(data.errors)[0];
                    if (firstKey) {
                        const err = data.errors[firstKey];
                        errorText = Array.isArray(err) ? err[0] : err;
                    }
                } else if (data.current_password) {
                    errorText = Array.isArray(data.current_password) ? data.current_password[0] : data.current_password;
                } else if (data.new_password) {
                    errorText = Array.isArray(data.new_password) ? data.new_password.join(', ') : data.new_password;
                }
                setMessage({ type: 'error', text: errorText });
            }
        } catch (error) {
            setMessage({ type: 'error', text: 'Network error. Please check your connection and try again.' });
        } finally {
            setLoading(false);
        }
    };

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'PENDING':
                return <span className="px-2 py-1 rounded-full text-xs bg-yellow-500/20 text-yellow-400 flex items-center gap-1"><Clock className="w-3 h-3" />Pending</span>;
            case 'APPROVED':
                return <span className="px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400 flex items-center gap-1"><Check className="w-3 h-3" />Approved</span>;
            case 'REJECTED':
                return <span className="px-2 py-1 rounded-full text-xs bg-red-500/20 text-red-400 flex items-center gap-1"><X className="w-3 h-3" />Rejected</span>;
            default:
                return null;
        }
    };

    const tabs = [
        { id: 'profile', label: 'Profile', icon: User },
        { id: 'security', label: 'Security', icon: Lock },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'privacy', label: 'Privacy & Data', icon: Shield },
    ];

    const pendingCount = pendingChanges.filter(c => c.status === 'PENDING').length;

    return (
        <div className="min-h-screen bg-[#0a0a1a] relative">
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Header */}
            <nav className="relative z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <Link href={user?.role === 'doctor' ? '/doctor/dashboard' : '/dashboard'} className="flex items-center gap-3">
                            <div className="w-8 h-8"><AnimatedHeart className="w-full h-full" /></div>
                            <span className="text-xl font-bold gradient-text">CardioDetect</span>
                        </Link>
                        <div className="flex items-center gap-4">
                            <motion.button onClick={refreshData} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                                className={`p-2 glass-card text-slate-400 hover:text-white ${refreshing ? 'animate-spin' : ''}`}>
                                <RefreshCw className="w-5 h-5" />
                            </motion.button>
                            <Link href={user?.role === 'doctor' ? '/doctor/dashboard' : '/dashboard'} className="text-slate-400 hover:text-white transition-colors">
                                ← Back to Dashboard
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            <main className="relative z-10 max-w-4xl mx-auto px-6 py-8">
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                    <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                        <Settings className="w-8 h-8 text-red-400" />
                        Settings
                    </h1>
                    <p className="text-slate-400 mb-8">Manage your account preferences</p>
                </motion.div>

                {/* Pending Changes Banner */}
                {pendingCount > 0 && (
                    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                        className="mb-6 p-4 glass-card border-yellow-500/30 flex items-center gap-3">
                        <Clock className="w-5 h-5 text-yellow-400" />
                        <span className="text-yellow-400">You have {pendingCount} pending change(s) waiting for admin approval</span>
                    </motion.div>
                )}

                {/* Message */}
                <AnimatePresence>
                    {message.text && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                            className={`mb-6 p-4 rounded-xl flex items-center gap-3 ${message.type === 'success' ? 'bg-green-500/20 text-green-400 border border-green-500/30' : message.type === 'info' ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                            {message.type === 'success' ? <Check className="w-5 h-5" /> : message.type === 'info' ? <AlertCircle className="w-5 h-5" /> : <X className="w-5 h-5" />}
                            {message.text}
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="grid md:grid-cols-4 gap-6">
                    {/* Sidebar */}
                    <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} className="glass-card p-4">
                        <div className="space-y-1">
                            {tabs.map((tab) => (
                                <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all ${activeTab === tab.id ? 'bg-red-500/20 text-red-400' : 'text-slate-400 hover:bg-white/5 hover:text-white'}`}>
                                    <tab.icon className="w-5 h-5" />
                                    {tab.label}
                                    {activeTab === tab.id && <ChevronRight className="w-4 h-4 ml-auto" />}
                                </button>
                            ))}
                        </div>
                    </motion.div>

                    {/* Content */}
                    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="md:col-span-3 space-y-6">
                        {/* Profile Tab */}
                        {activeTab === 'profile' && (
                            <>
                                <form onSubmit={submitForApproval} className="glass-card p-6">
                                    <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                        <User className="w-5 h-5 text-red-400" />Profile Information
                                    </h2>

                                    <div className="grid md:grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">First Name</label>
                                            <input type="text" value={profile.first_name} onChange={(e) => setProfile({ ...profile, first_name: e.target.value })}
                                                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500/50 focus:outline-none" />
                                        </div>
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Last Name</label>
                                            <input type="text" value={profile.last_name} onChange={(e) => setProfile({ ...profile, last_name: e.target.value })}
                                                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500/50 focus:outline-none" />
                                        </div>
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Email</label>
                                            <input type="email" value={profile.email} disabled
                                                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-slate-500 cursor-not-allowed" />
                                            <p className="text-xs text-slate-500 mt-1">Email cannot be changed</p>
                                        </div>
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Phone</label>
                                            <input type="tel" value={profile.phone} onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                                                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500/50 focus:outline-none" />
                                        </div>
                                    </div>

                                    <div className="mt-6 flex items-center gap-4">
                                        <motion.button type="submit" disabled={loading} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                                            className="px-6 py-3 glow-button text-white rounded-xl font-semibold disabled:opacity-50">
                                            {loading ? 'Submitting...' : 'Submit for Approval'}
                                        </motion.button>
                                        <span className="text-slate-500 text-sm">Changes require admin approval</span>
                                    </div>
                                </form>

                                {/* Pending Changes List */}
                                {pendingChanges.length > 0 && (
                                    <div className="glass-card p-6">
                                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                            <Clock className="w-5 h-5 text-yellow-400" />
                                            Change Requests
                                        </h2>
                                        <div className="space-y-3">
                                            {pendingChanges.map((change) => (
                                                <div key={change.id} className="p-4 bg-white/5 rounded-xl border border-white/10">
                                                    <div className="flex items-center justify-between mb-2">
                                                        <span className="text-white font-medium capitalize">{change.field.replace('_', ' ')}</span>
                                                        {getStatusBadge(change.status)}
                                                    </div>
                                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                                        <div>
                                                            <span className="text-slate-500">From:</span>
                                                            <span className="text-red-400 ml-2">{change.old_value || '(empty)'}</span>
                                                        </div>
                                                        <div>
                                                            <span className="text-slate-500">To:</span>
                                                            <span className="text-green-400 ml-2">{change.new_value}</span>
                                                        </div>
                                                    </div>
                                                    {change.admin_notes && (
                                                        <div className="mt-2 text-sm text-slate-400">
                                                            <span className="text-slate-500">Admin note:</span> {change.admin_notes}
                                                        </div>
                                                    )}
                                                    <div className="mt-2 text-xs text-slate-500">
                                                        Submitted: {new Date(change.created_at).toLocaleString()}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}

                        {/* Security Tab (Password) */}
                        {activeTab === 'security' && (
                            <div className="space-y-6">

                                <form onSubmit={changePassword} className="glass-card p-6">
                                    <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                        <Lock className="w-5 h-5 text-red-400" />Change Password
                                    </h2>

                                    <div className="space-y-4 max-w-md">
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Current Password</label>
                                            <input type="password" value={passwords.current_password} onChange={(e) => setPasswords({ ...passwords, current_password: e.target.value })}
                                                required className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500/50 focus:outline-none" />
                                        </div>
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">New Password</label>
                                            <input type="password" value={passwords.new_password} onChange={(e) => setPasswords({ ...passwords, new_password: e.target.value })}
                                                required minLength={8} className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500/50 focus:outline-none" />
                                            <p className="text-xs text-slate-500 mt-1">Minimum 8 characters</p>
                                        </div>
                                        <div>
                                            <label className="block text-sm text-slate-400 mb-2">Confirm New Password</label>
                                            <input type="password" value={passwords.confirm_password} onChange={(e) => setPasswords({ ...passwords, confirm_password: e.target.value })}
                                                required className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:border-red-500/50 focus:outline-none" />
                                        </div>
                                    </div>

                                    <motion.button type="submit" disabled={loading} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                                        className="mt-6 px-6 py-3 glow-button text-white rounded-xl font-semibold disabled:opacity-50">
                                        {loading ? 'Changing...' : 'Change Password'}
                                    </motion.button>
                                </form>
                            </div>
                        )}

                        {/* Notifications Tab */}
                        {activeTab === 'notifications' && (
                            <div className="glass-card p-6">
                                <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
                                    <Bell className="w-5 h-5 text-red-400" />Notification Preferences
                                </h2>

                                <div className="space-y-4">
                                    {[
                                        { key: 'predictions', label: 'Email notifications for new predictions' },
                                        { key: 'weekly', label: 'Weekly health summary' },
                                        { key: 'security', label: 'Security alerts' },
                                    ].map((item) => (
                                        <label key={item.key} className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 cursor-pointer hover:bg-white/10 transition-all">
                                            <span className="text-slate-300">{item.label}</span>
                                            <div className="relative">
                                                <input type="checkbox" checked={notifications[item.key as keyof typeof notifications]}
                                                    onChange={(e) => setNotifications({ ...notifications, [item.key]: e.target.checked })}
                                                    className="sr-only" />
                                                <div className={`w-12 h-6 rounded-full transition-colors ${notifications[item.key as keyof typeof notifications] ? 'bg-red-500' : 'bg-slate-600'}`}>
                                                    <div className={`w-5 h-5 bg-white rounded-full shadow transform transition-transform mt-0.5 ${notifications[item.key as keyof typeof notifications] ? 'translate-x-6 ml-0.5' : 'translate-x-0.5'}`} />
                                                </div>
                                            </div>
                                        </label>
                                    ))}
                                </div>

                                <motion.button whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                                    className="mt-6 px-6 py-3 glow-button text-white rounded-xl font-semibold">
                                    Save Preferences
                                </motion.button>
                            </div>
                        )}

                        {/* Privacy & Data Tab (GDPR) */}
                        {activeTab === 'privacy' && (
                            <div className="space-y-6">
                                {/* Data Export Section */}
                                <div className="glass-card p-6">
                                    <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                                        <Download className="w-5 h-5 text-blue-400" />Export Your Data
                                    </h2>
                                    <p className="text-slate-400 mb-4">
                                        Download all your personal data including profile information, prediction history,
                                        recommendations, and uploaded documents metadata in JSON format.
                                    </p>
                                    <motion.button
                                        onClick={handleDataExport}
                                        disabled={exportLoading}
                                        whileHover={{ scale: 1.02 }}
                                        whileTap={{ scale: 0.98 }}
                                        className="px-6 py-3 bg-blue-500/20 border border-blue-500/30 text-blue-400 rounded-xl font-semibold hover:bg-blue-500/30 transition-colors disabled:opacity-50 flex items-center gap-2"
                                    >
                                        <Download className="w-4 h-4" />
                                        {exportLoading ? 'Preparing Download...' : 'Download My Data'}
                                    </motion.button>
                                </div>

                                {/* Account Deletion Section */}
                                <div className="glass-card p-6 border-red-500/20">
                                    <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                                        <Trash2 className="w-5 h-5 text-red-400" />Delete Account
                                    </h2>

                                    {deletionRequest?.has_pending_request ? (
                                        <div className="space-y-4">
                                            <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-xl">
                                                <div className="flex items-center gap-2 text-yellow-400 mb-2">
                                                    <Clock className="w-5 h-5" />
                                                    <span className="font-semibold">Deletion Scheduled</span>
                                                </div>
                                                <p className="text-slate-400">
                                                    Your account is scheduled for deletion on{' '}
                                                    <span className="text-white">
                                                        {new Date(deletionRequest.scheduled_deletion_at!).toLocaleDateString()}
                                                    </span>
                                                </p>
                                                <p className="text-slate-500 text-sm mt-1">
                                                    Time remaining: {Math.round(deletionRequest.grace_period_remaining_hours || 0)} hours
                                                </p>
                                            </div>
                                            {deletionRequest.is_cancellable && (
                                                <motion.button
                                                    onClick={handleCancelDeletion}
                                                    disabled={deletionLoading}
                                                    whileHover={{ scale: 1.02 }}
                                                    whileTap={{ scale: 0.98 }}
                                                    className="px-6 py-3 bg-green-500/20 border border-green-500/30 text-green-400 rounded-xl font-semibold hover:bg-green-500/30 transition-colors disabled:opacity-50"
                                                >
                                                    {deletionLoading ? 'Cancelling...' : 'Cancel Deletion Request'}
                                                </motion.button>
                                            )}
                                        </div>
                                    ) : (
                                        <>
                                            <p className="text-slate-400 mb-4">
                                                Request permanent deletion of your account and all associated data.
                                                You will have <span className="text-yellow-400 font-semibold">7 days</span> to cancel this request.
                                            </p>
                                            <motion.button
                                                onClick={handleDeletionRequest}
                                                disabled={deletionLoading}
                                                whileHover={{ scale: 1.02 }}
                                                whileTap={{ scale: 0.98 }}
                                                className="px-6 py-3 bg-red-500/20 border border-red-500/30 text-red-400 rounded-xl font-semibold hover:bg-red-500/30 transition-colors disabled:opacity-50 flex items-center gap-2"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                                {deletionLoading ? 'Processing...' : 'Request Account Deletion'}
                                            </motion.button>
                                        </>
                                    )}
                                </div>

                                {/* GDPR Info */}
                                <div className="glass-card p-6 border-slate-500/20">
                                    <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                                        <FileText className="w-5 h-5 text-slate-400" />Your Privacy Rights
                                    </h2>
                                    <ul className="text-slate-400 space-y-2 text-sm">
                                        <li className="flex items-start gap-2">
                                            <Check className="w-4 h-4 text-green-400 mt-0.5" />
                                            <span><strong>Right to Access:</strong> Download all your personal data at any time.</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <Check className="w-4 h-4 text-green-400 mt-0.5" />
                                            <span><strong>Right to Erasure:</strong> Request complete deletion of your data.</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <Check className="w-4 h-4 text-green-400 mt-0.5" />
                                            <span><strong>7-Day Grace Period:</strong> Cancel deletion requests within 7 days.</span>
                                        </li>
                                        <li className="flex items-start gap-2">
                                            <Check className="w-4 h-4 text-green-400 mt-0.5" />
                                            <span><strong>Data Portability:</strong> Export data in standard JSON format.</span>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        )}
                    </motion.div>
                </div>

                {/* Help Section */}
                <div className="mt-12 text-center">
                    <p className="text-slate-500 text-sm">
                        Need help or found an issue? <Link href="mailto:support@cardiodetect.ai" className="text-blue-400 hover:underline">Click here</Link>
                    </p>
                </div>
            </main>
        </div>
    );
}
