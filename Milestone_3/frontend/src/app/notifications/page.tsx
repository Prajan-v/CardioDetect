'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, Check, AlertTriangle, Info, Trash2, CheckCheck, Loader2 } from 'lucide-react';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';
import { getUser, authFetch, getApiUrl } from '@/services/auth';

interface Notification {
    id: string;
    type: 'info' | 'warning' | 'success' | 'error' | 'prediction';
    title: string;
    message: string;
    time_ago: string;
    read: boolean; // mapped from is_read
}

export default function NotificationsPage() {
    const user = getUser();
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [isLoading, setIsLoading] = useState(true);

    const fetchNotifications = async () => {
        try {
            const res = await authFetch(getApiUrl('auth/notifications/'));
            if (res.ok) {
                const data = await res.json();
                // Map backend fields to frontend interface
                interface BackendNotification {
                    id: string;
                    type: 'info' | 'warning' | 'success' | 'error' | 'prediction';
                    title: string;
                    message: string;
                    time_ago: string;
                    is_read: boolean;
                }
                const mapped = data.map((item: BackendNotification) => ({
                    id: item.id,
                    type: item.type, // Serializer maps notification_type -> type
                    title: item.title,
                    message: item.message,
                    time_ago: item.time_ago,
                    read: item.is_read
                }));
                setNotifications(mapped);
            }
        } catch (error) {
            console.error('Failed to load notifications', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchNotifications();
    }, []);

    const markAllRead = async () => {
        try {
            await authFetch(getApiUrl('auth/notifications/mark_all_read/'), { method: 'POST' });
            setNotifications(notifications.map(n => ({ ...n, read: true })));
        } catch (error) {
            console.error('Failed to mark all read', error);
        }
    };

    const markAsRead = async (id: string) => {
        try {
            await authFetch(getApiUrl(`auth/notifications/${id}/mark_read/`), { method: 'POST' });
            setNotifications(notifications.map(n => n.id === id ? { ...n, read: true } : n));
        } catch (error) {
            console.error('Failed to mark read', error);
        }
    };

    const deleteNotification = async (id: string) => {
        try {
            await authFetch(getApiUrl(`auth/notifications/${id}/`), { method: 'DELETE' });
            setNotifications(notifications.filter(n => n.id !== id));
        } catch (error) {
            console.error('Failed to delete notification', error);
        }
    };

    const getIcon = (type: string) => {
        switch (type) {
            case 'success': return <Check className="w-5 h-5 text-green-400" />;
            case 'warning': return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
            case 'error': return <AlertTriangle className="w-5 h-5 text-red-400" />;
            case 'prediction': return <AnimatedHeart className="w-5 h-5" />;
            default: return <Info className="w-5 h-5 text-blue-400" />;
        }
    };

    const unreadCount = notifications.filter(n => !n.read).length;

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
                        <Link href={user?.role === 'doctor' ? '/doctor/dashboard' : '/dashboard'} className="text-slate-400 hover:text-white transition-colors">
                            ← Back to Dashboard
                        </Link>
                    </div>
                </div>
            </nav>

            <main className="relative z-10 max-w-3xl mx-auto px-6 py-8">
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                            <Bell className="w-8 h-8 text-red-400" />
                            Notifications
                            {unreadCount > 0 && (
                                <span className="px-2 py-1 text-sm bg-red-500 text-white rounded-full">{unreadCount}</span>
                            )}
                        </h1>
                        <p className="text-slate-400 mt-1">Stay updated with your health assessments</p>
                    </div>
                    {notifications.length > 0 && (
                        <motion.button onClick={markAllRead} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                            className="px-4 py-2 glass-card text-slate-300 hover:text-white flex items-center gap-2">
                            <CheckCheck className="w-4 h-4" />Mark all read
                        </motion.button>
                    )}
                </motion.div>

                {isLoading ? (
                    <div className="flex justify-center py-20">
                        <Loader2 className="w-10 h-10 text-red-400 animate-spin" />
                    </div>
                ) : notifications.length === 0 ? (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-card p-12 text-center">
                        <Bell className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                        <h3 className="text-xl font-semibold text-white mb-2">No Notifications</h3>
                        <p className="text-slate-400">You&apos;re all caught up!</p>
                    </motion.div>
                ) : (
                    <div className="space-y-3">
                        <AnimatePresence>
                            {notifications.map((notification) => (
                                <motion.div
                                    key={notification.id}
                                    layout
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    exit={{ opacity: 0, scale: 0.9, height: 0 }}
                                    onClick={() => !notification.read && markAsRead(notification.id)}
                                    className={`glass-card p-4 flex items-start gap-4 transition-all cursor-pointer hover:bg-white/5 ${!notification.read ? 'border-l-4 border-l-red-500 bg-white/5' : 'opacity-75'}`}
                                >
                                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${notification.type === 'success' ? 'bg-green-500/20' : notification.type === 'warning' ? 'bg-yellow-500/20' : 'bg-blue-500/20'}`}>
                                        {getIcon(notification.type)}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center justify-between gap-4">
                                            <h3 className={`text-white font-medium truncate ${!notification.read ? 'font-semibold' : ''}`}>{notification.title}</h3>
                                            <span className="text-slate-500 text-xs shrink-0">{notification.time_ago}</span>
                                        </div>
                                        <p className="text-slate-400 text-sm mt-1 leading-relaxed">{notification.message}</p>
                                    </div>
                                    <button
                                        onClick={(e) => { e.stopPropagation(); deleteNotification(notification.id); }}
                                        className="text-slate-600 hover:text-red-400 transition-colors p-1"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                )}

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
