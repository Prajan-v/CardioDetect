'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Eye, EyeOff, Mail, Lock, ArrowRight, Heart } from 'lucide-react';
import Link from 'next/link';
import AnimatedHeart from '@/components/AnimatedHeart';
import ECGLine from '@/components/ECGLine';
import FloatingParticles from '@/components/FloatingParticles';
import { login } from '@/services/auth';

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            const result = await login({ email, password });

            if (result.success) {
                // Normal login success
                if (result.user) {
                    const destination = result.user.role === 'doctor' ? '/doctor/dashboard' : '/dashboard';
                    window.location.href = destination;
                } else {
                    window.location.href = '/dashboard';
                }
            } else {
                setError(result.error || 'Invalid email or password');
                setIsLoading(false);
            }
        } catch (err) {
            console.error('Login exception:', err);
            setError('Server not available. Please try again later.');
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-[#0a0a1a] flex">
            {/* Background Effects */}
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Left Side - Branding */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden">
                {/* ECG Background */}
                <div className="absolute top-1/2 left-0 right-0 h-32 opacity-20">
                    <ECGLine className="w-full h-full" />
                </div>

                <div className="relative z-10 flex flex-col items-center justify-center w-full p-12">
                    {/* Animated Heart */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.8 }}
                        className="w-40 h-40 mb-8"
                    >
                        <AnimatedHeart className="w-full h-full" />
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="text-4xl font-bold gradient-text mb-4"
                    >
                        CardioDetect
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="text-slate-400 text-center max-w-md text-lg"
                    >
                        AI-Powered Heart Disease Detection & 10-Year Risk Prediction
                    </motion.p>
                </div>
            </div>

            {/* Right Side - Login Form */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-6 md:p-12">
                <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6 }}
                    className="w-full max-w-md"
                >
                    <div className="glass-card p-8 md:p-10">
                        <div className="text-center mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">
                                Welcome Back
                            </h2>
                            <p className="text-slate-400">
                                Sign in to continue to CardioDetect
                            </p>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-6">
                            {error && (
                                <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-xl text-sm">
                                    {error}
                                </div>
                            )}

                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">Email Address</label>
                                <div className="relative">
                                    <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                    <input
                                        type="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="you@example.com"
                                        className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-4 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-red-500/50"
                                        required
                                    />
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
                                <div className="relative">
                                    <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="Enter your password"
                                        className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-12 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-red-500/50"
                                        required
                                    />
                                    <button type="button" onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white">
                                        {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                                    </button>
                                </div>
                            </div>

                            <div className="flex items-center justify-between text-sm">
                                <label className="flex items-center gap-2 text-slate-400 cursor-pointer">
                                    <input type="checkbox" className="w-4 h-4 rounded border-slate-600 bg-white/5 text-red-500 focus:ring-red-500/50" />
                                    Remember me
                                </label>
                                <Link href="/forgot-password" className="text-red-400 hover:text-red-300 transition-colors">
                                    Forgot password?
                                </Link>
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading}
                                className="w-full glow-button text-white py-4 rounded-xl font-semibold flex items-center justify-center gap-2 disabled:opacity-50"
                            >
                                {isLoading ? <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <>Sign In <ArrowRight className="w-5 h-5" /></>}
                            </button>

                            <div className="flex items-center gap-4 my-8">
                                <div className="flex-1 h-px bg-white/10" />
                                <span className="text-slate-500 text-sm">or</span>
                                <div className="flex-1 h-px bg-white/10" />
                            </div>

                            <p className="text-center text-slate-400">
                                Don&apos;t have an account?{' '}
                                <Link href="/register" className="text-red-400 hover:text-red-300 font-medium transition-colors">
                                    Create account
                                </Link>
                            </p>
                        </form>
                    </div>

                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="text-center mt-6">
                        <Link href="/" className="text-slate-500 hover:text-white text-sm transition-colors flex items-center justify-center gap-2">
                            <Heart className="w-4 h-4" />Back to CardioDetect
                        </Link>
                    </motion.div>
                </motion.div>
            </div>
        </div>
    );
}
