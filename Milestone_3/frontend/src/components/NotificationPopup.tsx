'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, CheckCircle, Info, Activity } from 'lucide-react';

interface PopupNotification {
    id: string;
    title: string;
    message: string;
    type: 'info' | 'warning' | 'success' | 'error' | 'prediction';
    duration?: number; // Auto-dismiss after ms, 0 = no auto-dismiss
}

interface NotificationPopupProps {
    position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
}

// Global notification state
let notificationListeners: ((notification: PopupNotification) => void)[] = [];
let notificationIdCounter = 0;

// Function to show notifications from anywhere in the app
export const showNotification = (notification: Omit<PopupNotification, 'id'>) => {
    const fullNotification: PopupNotification = {
        ...notification,
        id: `notification-${++notificationIdCounter}`,
        duration: notification.duration ?? 5000, // Default 5 seconds
    };
    notificationListeners.forEach(listener => listener(fullNotification));
};

export default function NotificationPopup({ position = 'top-right' }: NotificationPopupProps) {
    const [notifications, setNotifications] = useState<PopupNotification[]>([]);

    // Define dismissNotification with useCallback BEFORE using it in useEffect
    const dismissNotification = useCallback((id: string) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    }, []);

    useEffect(() => {
        // Register listener
        const listener = (notification: PopupNotification) => {
            setNotifications(prev => [...prev, notification]);

            // Auto-dismiss
            if (notification.duration && notification.duration > 0) {
                setTimeout(() => {
                    dismissNotification(notification.id);
                }, notification.duration);
            }
        };

        notificationListeners.push(listener);

        return () => {
            notificationListeners = notificationListeners.filter(l => l !== listener);
        };
    }, [dismissNotification]);

    const getIcon = (type: string) => {
        switch (type) {
            case 'warning':
                return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
            case 'success':
                return <CheckCircle className="w-5 h-5 text-green-400" />;
            case 'error':
                return <X className="w-5 h-5 text-red-400" />;
            case 'prediction':
                return <Activity className="w-5 h-5 text-purple-400" />;
            default:
                return <Info className="w-5 h-5 text-blue-400" />;
        }
    };

    const getPositionClasses = () => {
        switch (position) {
            case 'top-left':
                return 'top-4 left-4';
            case 'bottom-right':
                return 'bottom-4 right-4';
            case 'bottom-left':
                return 'bottom-4 left-4';
            default:
                return 'top-4 right-4';
        }
    };

    const getBgColor = (type: string) => {
        switch (type) {
            case 'warning':
                return 'border-yellow-500/30 bg-yellow-500/10';
            case 'success':
                return 'border-green-500/30 bg-green-500/10';
            case 'error':
                return 'border-red-500/30 bg-red-500/10';
            case 'prediction':
                return 'border-purple-500/30 bg-purple-500/10';
            default:
                return 'border-blue-500/30 bg-blue-500/10';
        }
    };

    return (
        <div className={`fixed ${getPositionClasses()} z-[100] flex flex-col gap-3 w-80`}>
            <AnimatePresence>
                {notifications.map((notification) => (
                    <motion.div
                        key={notification.id}
                        initial={{ opacity: 0, x: position.includes('right') ? 100 : -100, scale: 0.9 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: position.includes('right') ? 100 : -100, scale: 0.9 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        className={`backdrop-blur-xl rounded-xl p-4 border shadow-2xl ${getBgColor(notification.type)}`}
                    >
                        <div className="flex items-start gap-3">
                            <div className="mt-0.5">
                                {getIcon(notification.type)}
                            </div>
                            <div className="flex-1 min-w-0">
                                <h4 className="text-white font-medium text-sm">
                                    {notification.title}
                                </h4>
                                <p className="text-slate-400 text-xs mt-1 line-clamp-2">
                                    {notification.message}
                                </p>
                            </div>
                            <button
                                onClick={() => dismissNotification(notification.id)}
                                className="text-slate-500 hover:text-white transition-colors p-1"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        {/* Progress bar for auto-dismiss */}
                        {notification.duration && notification.duration > 0 && (
                            <motion.div
                                initial={{ scaleX: 1 }}
                                animate={{ scaleX: 0 }}
                                transition={{ duration: notification.duration / 1000, ease: 'linear' }}
                                className="h-0.5 bg-white/30 rounded-full mt-3 origin-left"
                            />
                        )}
                    </motion.div>
                ))}
            </AnimatePresence>
        </div>
    );
}
