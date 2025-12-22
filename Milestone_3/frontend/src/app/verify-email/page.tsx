'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Heart, Mail, CheckCircle, XCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import FloatingParticles from '@/components/FloatingParticles';
import { API_ENDPOINTS } from '@/services/apiClient';

function VerifyEmailContent() {
    const searchParams = useSearchParams();
    const router = useRouter();

    const [status, setStatus] = useState<'verifying' | 'success' | 'error' | 'invalid'>('verifying');
    const [message, setMessage] = useState('');
    const [resending, setResending] = useState(false);

    useEffect(() => {
        const email = searchParams.get('email');
        const token = searchParams.get('token');

        if (!email || !token) {
            setStatus('invalid');
            return;
        }

        verifyEmail(email, token);
    }, [searchParams]);

    const verifyEmail = async (email: string, token: string) => {
        try {
            const res = await fetch(API_ENDPOINTS.auth.verifyEmail(email, token));

            if (res.ok) {
                setStatus('success');
                setTimeout(() => router.push('/login'), 3000);
            } else {
                const data = await res.json();
                setStatus('error');
                setMessage(data.detail || 'Verification failed. The link may have expired.');
            }
        } catch (err) {
            setStatus('error');
            setMessage('Network error. Please try again.');
        }
    };

    const resendVerification = async () => {
        const email = searchParams.get('email');
        if (!email) return;

        setResending(true);
        try {
            const res = await fetch(API_ENDPOINTS.auth.resendVerification(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email })
            });

            if (res.ok) {
                setMessage('Verification email sent! Check your inbox.');
            } else {
                setMessage('Failed to resend. Please try again later.');
            }
        } catch (err) {
            setMessage('Network error. Please try again.');
        } finally {
            setResending(false);
        }
    };

    return (
        <>
            {status === 'verifying' && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-center"
                >
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-red-500/20 to-pink-500/20 flex items-center justify-center">
                        <Mail className="w-8 h-8 text-red-400 animate-pulse" />
                    </div>
                    <h2 className="text-xl font-bold text-white mb-4">Verifying Email...</h2>
                    <div className="flex justify-center">
                        <div className="w-8 h-8 border-2 border-red-500/30 border-t-red-500 rounded-full animate-spin" />
                    </div>
                </motion.div>
            )}

            {status === 'success' && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-center"
                >
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-green-500/20 to-emerald-500/20 flex items-center justify-center">
                        <CheckCircle className="w-8 h-8 text-green-400" />
                    </div>
                    <h2 className="text-xl font-bold text-white mb-2">Email Verified!</h2>
                    <p className="text-slate-400 mb-6">
                        Your email has been verified successfully.
                    </p>
                    <p className="text-sm text-slate-500 mb-4">
                        Redirecting to login...
                    </p>
                    <Link
                        href="/login"
                        className="inline-block glow-button text-white px-6 py-3 rounded-xl font-medium"
                    >
                        Sign In Now
                    </Link>
                </motion.div>
            )}

            {status === 'error' && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-center"
                >
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-red-500/20 to-pink-500/20 flex items-center justify-center">
                        <XCircle className="w-8 h-8 text-red-400" />
                    </div>
                    <h2 className="text-xl font-bold text-white mb-2">Verification Failed</h2>
                    <p className="text-slate-400 mb-6">{message}</p>
                    <button
                        onClick={resendVerification}
                        disabled={resending}
                        className="glow-button text-white px-6 py-3 rounded-xl font-medium disabled:opacity-50 flex items-center gap-2 mx-auto"
                    >
                        {resending ? (
                            <>
                                <RefreshCw className="w-5 h-5 animate-spin" />
                                Sending...
                            </>
                        ) : (
                            <>
                                <RefreshCw className="w-5 h-5" />
                                Resend Verification Email
                            </>
                        )}
                    </button>
                </motion.div>
            )}

            {status === 'invalid' && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="text-center"
                >
                    <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-amber-500/20 to-orange-500/20 flex items-center justify-center">
                        <AlertTriangle className="w-8 h-8 text-amber-400" />
                    </div>
                    <h2 className="text-xl font-bold text-white mb-2">Invalid Link</h2>
                    <p className="text-slate-400 mb-6">
                        This verification link is invalid or incomplete.
                    </p>
                    <Link
                        href="/login"
                        className="inline-block glow-button text-white px-6 py-3 rounded-xl font-medium"
                    >
                        Go to Login
                    </Link>
                </motion.div>
            )}
        </>
    );
}

export default function VerifyEmailPage() {
    return (
        <div className="min-h-screen bg-[#0a0a1a] flex items-center justify-center px-4">
            {/* Background Effects */}
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="w-full max-w-md relative z-10"
            >
                {/* Logo */}
                <div className="text-center mb-8">
                    <Link href="/" className="inline-flex items-center gap-3">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-red-500 to-pink-500 flex items-center justify-center">
                            <Heart className="w-6 h-6 text-white" />
                        </div>
                        <span className="text-2xl font-bold gradient-text">CardioDetect</span>
                    </Link>
                </div>

                <div className="glass-card p-8 md:p-10">
                    <Suspense fallback={
                        <div className="text-center py-8">
                            <div className="w-8 h-8 border-2 border-red-500/30 border-t-red-500 rounded-full animate-spin mx-auto" />
                        </div>
                    }>
                        <VerifyEmailContent />
                    </Suspense>

                    <div className="mt-8 pt-6 border-t border-white/10 text-center">
                        <Link
                            href="/login"
                            className="text-slate-400 hover:text-white transition-colors text-sm"
                        >
                            Back to Sign In
                        </Link>
                    </div>
                </div>

                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="text-center mt-6"
                >
                    <Link
                        href="/"
                        className="text-slate-500 hover:text-white text-sm transition-colors flex items-center justify-center gap-2"
                    >
                        <Heart className="w-4 h-4" />
                        Back to CardioDetect
                    </Link>
                </motion.div>
            </motion.div>
        </div>
    );
}
