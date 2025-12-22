'use client';

import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, AlertTriangle, AlertCircle, Info } from 'lucide-react';

type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
    id: string;
    type: ToastType;
    title: string;
    message?: string;
    duration?: number;
}

interface ToastContextType {
    toasts: Toast[];
    addToast: (toast: Omit<Toast, 'id'>) => void;
    removeToast: (id: string) => void;
    success: (title: string, message?: string) => void;
    error: (title: string, message?: string) => void;
    warning: (title: string, message?: string) => void;
    info: (title: string, message?: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

const toastConfig = {
    success: {
        icon: CheckCircle,
        bgColor: 'bg-green-500/20',
        borderColor: 'border-green-500/50',
        iconColor: 'text-green-400'
    },
    error: {
        icon: AlertCircle,
        bgColor: 'bg-red-500/20',
        borderColor: 'border-red-500/50',
        iconColor: 'text-red-400'
    },
    warning: {
        icon: AlertTriangle,
        bgColor: 'bg-yellow-500/20',
        borderColor: 'border-yellow-500/50',
        iconColor: 'text-yellow-400'
    },
    info: {
        icon: Info,
        bgColor: 'bg-blue-500/20',
        borderColor: 'border-blue-500/50',
        iconColor: 'text-blue-400'
    },
};

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);

    const removeToast = useCallback((id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    }, []);

    const addToast = useCallback((toast: Omit<Toast, 'id'>) => {
        const id = Math.random().toString(36).substr(2, 9);
        const newToast = { ...toast, id };
        setToasts(prev => [...prev, newToast]);

        // Auto-remove after duration
        setTimeout(() => {
            removeToast(id);
        }, toast.duration || 5000);
    }, [removeToast]);

    const success = useCallback((title: string, message?: string) => {
        addToast({ type: 'success', title, message });
    }, [addToast]);

    const error = useCallback((title: string, message?: string) => {
        addToast({ type: 'error', title, message, duration: 8000 });
    }, [addToast]);

    const warning = useCallback((title: string, message?: string) => {
        addToast({ type: 'warning', title, message });
    }, [addToast]);

    const info = useCallback((title: string, message?: string) => {
        addToast({ type: 'info', title, message });
    }, [addToast]);

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast, success, error, warning, info }}>
            {children}

            {/* Toast Container */}
            <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-3 max-w-sm">
                <AnimatePresence>
                    {toasts.map(toast => {
                        const config = toastConfig[toast.type];
                        const Icon = config.icon;

                        return (
                            <motion.div
                                key={toast.id}
                                initial={{ opacity: 0, x: 100, scale: 0.9 }}
                                animate={{ opacity: 1, x: 0, scale: 1 }}
                                exit={{ opacity: 0, x: 100, scale: 0.9 }}
                                className={`${config.bgColor} ${config.borderColor} border backdrop-blur-xl rounded-xl p-4 shadow-lg`}
                            >
                                <div className="flex items-start gap-3">
                                    <Icon className={`w-5 h-5 ${config.iconColor} shrink-0 mt-0.5`} />
                                    <div className="flex-1 min-w-0">
                                        <p className="text-white font-medium text-sm">{toast.title}</p>
                                        {toast.message && (
                                            <p className="text-slate-400 text-xs mt-1">{toast.message}</p>
                                        )}
                                    </div>
                                    <button
                                        onClick={() => removeToast(toast.id)}
                                        className="text-slate-500 hover:text-white transition-colors shrink-0"
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>
        </ToastContext.Provider>
    );
}

export function useToast() {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error('useToast must be used within a ToastProvider');
    }
    return context;
}
