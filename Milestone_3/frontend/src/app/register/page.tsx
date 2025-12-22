'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Eye, EyeOff, Mail, Lock, User, Stethoscope, UserCircle, ArrowRight, Heart, Check, X } from 'lucide-react';
import Link from 'next/link';
import AnimatedHeart from '@/components/AnimatedHeart';
import ECGLine from '@/components/ECGLine';
import FloatingParticles from '@/components/FloatingParticles';
import Shimmer from '@/components/Shimmer';

import { register } from '@/services/auth';

type UserRole = 'patient' | 'doctor';

interface PasswordStrength {
    score: number;
    label: string;
    color: string;
}

const getPasswordStrength = (password: string): PasswordStrength => {
    let score = 0;
    if (password.length >= 8) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[a-z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;

    if (score <= 2) return { score, label: 'Weak', color: 'bg-red-500' };
    if (score <= 3) return { score, label: 'Fair', color: 'bg-yellow-500' };
    if (score <= 4) return { score, label: 'Good', color: 'bg-blue-500' };
    return { score, label: 'Strong', color: 'bg-green-500' };
};

// Skeleton loader for the form
function RegisterFormSkeleton() {
    return (
        <div className="glass-card p-8 md:p-10 animate-pulse">
            <div className="text-center mb-8">
                <Shimmer height="h-8" width="w-48" className="mx-auto mb-3" />
                <Shimmer height="h-4" width="w-64" className="mx-auto" />
            </div>

            {/* Role selector skeleton */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <Shimmer height="h-20" className="rounded-xl" />
                <Shimmer height="h-20" className="rounded-xl" />
            </div>

            {/* Form fields skeleton */}
            <div className="space-y-5">
                <div>
                    <Shimmer height="h-4" width="w-24" className="mb-2" />
                    <Shimmer height="h-12" className="rounded-xl" />
                </div>
                <div>
                    <Shimmer height="h-4" width="w-32" className="mb-2" />
                    <Shimmer height="h-12" className="rounded-xl" />
                </div>
                <div>
                    <Shimmer height="h-4" width="w-20" className="mb-2" />
                    <Shimmer height="h-12" className="rounded-xl" />
                </div>
                <div>
                    <Shimmer height="h-4" width="w-36" className="mb-2" />
                    <Shimmer height="h-12" className="rounded-xl" />
                </div>
            </div>

            {/* Terms skeleton */}
            <div className="mt-6 flex items-center gap-3">
                <Shimmer height="h-4" width="w-4" className="rounded" />
                <Shimmer height="h-4" width="w-48" />
            </div>

            {/* Button skeleton */}
            <Shimmer height="h-14" className="rounded-xl mt-6" />
        </div>
    );
}

export default function RegisterPage() {
    const [isPageLoading, setIsPageLoading] = useState(true);
    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [role, setRole] = useState<UserRole>('patient');
    const [licenseNumber, setLicenseNumber] = useState('');
    const [agreedToTerms, setAgreedToTerms] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const passwordStrength = getPasswordStrength(password);
    const passwordsMatch = password === confirmPassword && confirmPassword.length > 0;

    // Simulate page loading
    useEffect(() => {
        const timer = setTimeout(() => setIsPageLoading(false), 1200);
        return () => clearTimeout(timer);
    }, []);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');

        // Validation
        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }
        if (password.length < 8) {
            setError('Password must be at least 8 characters');
            return;
        }
        if (!agreedToTerms) {
            setError('Please agree to the Terms of Service');
            return;
        }
        if (role === 'doctor' && !licenseNumber.trim()) {
            setError('License number is required for doctors');
            return;
        }

        setIsLoading(true);

        // Call real API
        const result = await register({
            email,
            password,
            full_name: fullName,
            role,
            license_number: role === 'doctor' ? licenseNumber : undefined,
        });

        setIsLoading(false);

        if (result.success) {
            // Redirect to login on success
            window.location.href = '/login?registered=true';
        } else {
            setError(result.error || 'Registration failed');
        }
    };

    return (
        <div className="min-h-screen bg-[#0a0a1a] flex">
            {/* Background Effects */}
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Left Side - Form */}
            <div className="w-full lg:w-1/2 flex items-center justify-center p-6 md:p-12 overflow-y-auto">
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6 }}
                    className="w-full max-w-md"
                >
                    {/* Mobile Logo */}
                    <div className="lg:hidden flex items-center justify-center gap-3 mb-8">
                        <div className="w-10 h-10">
                            <AnimatedHeart className="w-full h-full" />
                        </div>
                        <span className="text-2xl font-bold gradient-text">CardioDetect</span>
                    </div>

                    {/* Show skeleton or form */}
                    {isPageLoading ? (
                        <RegisterFormSkeleton />
                    ) : (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4 }}
                            className="glass-card p-8 md:p-10"
                        >
                            <div className="text-center mb-8">
                                <h2 className="text-3xl font-bold text-white mb-2">Create Account</h2>
                                <p className="text-slate-400">Join CardioDetect for AI-powered heart health</p>
                            </div>

                            {/* Role Selector */}
                            <div className="grid grid-cols-2 gap-4 mb-6">
                                <motion.button
                                    type="button"
                                    onClick={() => setRole('patient')}
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${role === 'patient'
                                        ? 'border-red-500 bg-red-500/10'
                                        : 'border-white/10 bg-white/5 hover:border-white/30'
                                        }`}
                                >
                                    <UserCircle className={`w-8 h-8 ${role === 'patient' ? 'text-red-400' : 'text-slate-400'}`} />
                                    <span className={`font-medium ${role === 'patient' ? 'text-white' : 'text-slate-400'}`}>
                                        Patient
                                    </span>
                                </motion.button>

                                <motion.button
                                    type="button"
                                    onClick={() => { setRole('doctor'); setError(''); }}

                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    className={`p-4 rounded-xl border-2 transition-all flex flex-col items-center gap-2 ${role === 'doctor'
                                        ? 'border-blue-500 bg-blue-500/10'
                                        : 'border-white/10 bg-white/5 hover:border-white/30'
                                        }`}
                                >
                                    <Stethoscope className={`w-8 h-8 ${role === 'doctor' ? 'text-blue-400' : 'text-slate-400'}`} />
                                    <span className={`font-medium ${role === 'doctor' ? 'text-white' : 'text-slate-400'}`}>
                                        Doctor
                                    </span>
                                </motion.button>
                            </div>

                            <form onSubmit={handleSubmit} className="space-y-5">
                                {/* Error Message */}
                                {error && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-xl text-sm flex items-center gap-2"
                                    >
                                        <X className="w-4 h-4" />
                                        {error}
                                    </motion.div>
                                )}

                                {/* Full Name */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Full Name
                                    </label>
                                    <div className="relative">
                                        <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                        <input
                                            type="text"
                                            value={fullName}
                                            onChange={(e) => setFullName(e.target.value)}
                                            placeholder="John Doe"
                                            className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-4 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                                            required
                                        />
                                    </div>
                                </div>

                                {/* License Number - Only for Doctors */}
                                {role === 'doctor' && (
                                    <motion.div
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                    >
                                        <label className="block text-sm font-medium text-slate-300 mb-2">
                                            Medical License Number
                                        </label>
                                        <div className="relative">
                                            <Stethoscope className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                            <input
                                                type="text"
                                                value={licenseNumber}
                                                onChange={(e) => { setLicenseNumber(e.target.value); setError(''); }}

                                                placeholder="Enter your medical license number"
                                                className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-4 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/50 transition-all"
                                                required
                                            />
                                        </div>
                                    </motion.div>
                                )}

                                {/* Email */}

                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Email Address
                                    </label>
                                    <div className="relative">
                                        <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                        <input
                                            type="email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            placeholder="you@example.com"
                                            className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-4 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                                            required
                                        />
                                    </div>
                                </div>

                                {/* Password */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Password
                                    </label>
                                    <div className="relative">
                                        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                        <input
                                            type={showPassword ? 'text' : 'password'}
                                            value={password}
                                            onChange={(e) => setPassword(e.target.value)}
                                            placeholder="Create a strong password"
                                            className="w-full bg-white/5 border border-white/10 rounded-xl pl-12 pr-12 py-3.5 text-white placeholder-slate-500 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                                            required
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowPassword(!showPassword)}
                                            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
                                        >
                                            {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                                        </button>
                                    </div>
                                    {/* Password Strength Indicator */}
                                    {password && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            className="mt-2"
                                        >
                                            <div className="flex items-center gap-2">
                                                <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                    <motion.div
                                                        initial={{ width: 0 }}
                                                        animate={{ width: `${(passwordStrength.score / 5) * 100}%` }}
                                                        className={`h-full ${passwordStrength.color} rounded-full`}
                                                    />
                                                </div>
                                                <span className={`text-xs font-medium ${passwordStrength.color === 'bg-red-500' ? 'text-red-400' :
                                                    passwordStrength.color === 'bg-yellow-500' ? 'text-yellow-400' :
                                                        passwordStrength.color === 'bg-blue-500' ? 'text-blue-400' :
                                                            'text-green-400'
                                                    }`}>
                                                    {passwordStrength.label}
                                                </span>
                                            </div>
                                        </motion.div>
                                    )}
                                </div>

                                {/* Confirm Password */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Confirm Password
                                    </label>
                                    <div className="relative">
                                        <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                                        <input
                                            type={showConfirmPassword ? 'text' : 'password'}
                                            value={confirmPassword}
                                            onChange={(e) => setConfirmPassword(e.target.value)}
                                            placeholder="Confirm your password"
                                            className={`w-full bg-white/5 border rounded-xl pl-12 pr-12 py-3.5 text-white placeholder-slate-500 focus:outline-none transition-all ${confirmPassword && (passwordsMatch
                                                ? 'border-green-500/50 focus:border-green-500/50 focus:ring-1 focus:ring-green-500/50'
                                                : 'border-red-500/50 focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50')
                                                } ${!confirmPassword && 'border-white/10 focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50'}`}
                                            required
                                        />
                                        <button
                                            type="button"
                                            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                            className="absolute right-10 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
                                        >
                                            {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                                        </button>
                                        {/* Match indicator */}
                                        {confirmPassword && (
                                            <div className="absolute right-4 top-1/2 -translate-y-1/2">
                                                {passwordsMatch ? (
                                                    <Check className="w-5 h-5 text-green-400" />
                                                ) : (
                                                    <X className="w-5 h-5 text-red-400" />
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>

                                {/* Terms */}
                                <label className="flex items-start gap-3 cursor-pointer group">
                                    <div className="relative mt-0.5">
                                        <input
                                            type="checkbox"
                                            checked={agreedToTerms}
                                            onChange={(e) => setAgreedToTerms(e.target.checked)}
                                            className="w-5 h-5 rounded border-slate-600 bg-white/5 text-red-500 focus:ring-red-500/50 cursor-pointer"
                                        />
                                    </div>
                                    <span className="text-sm text-slate-400 group-hover:text-slate-300 transition-colors">
                                        I agree to the{' '}
                                        <Link href="/terms" className="text-red-400 hover:text-red-300">
                                            Terms of Service
                                        </Link>{' '}
                                        and{' '}
                                        <Link href="/privacy" className="text-red-400 hover:text-red-300">
                                            Privacy Policy
                                        </Link>
                                    </span>
                                </label>

                                {/* Submit Button */}
                                <motion.button
                                    type="submit"
                                    disabled={isLoading || !agreedToTerms}
                                    whileHover={{ scale: agreedToTerms ? 1.02 : 1 }}
                                    whileTap={{ scale: agreedToTerms ? 0.98 : 1 }}
                                    className="w-full glow-button text-white py-4 rounded-xl font-semibold flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed mt-2"
                                >
                                    {isLoading ? (
                                        <div className="flex items-center gap-3">
                                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                            <span>Creating your account...</span>
                                        </div>
                                    ) : (
                                        <>
                                            Create Account
                                            <ArrowRight className="w-5 h-5" />
                                        </>
                                    )}
                                </motion.button>
                            </form>

                            {/* Divider */}
                            <div className="flex items-center gap-4 my-6">
                                <div className="flex-1 h-px bg-white/10" />
                                <span className="text-slate-500 text-sm">or</span>
                                <div className="flex-1 h-px bg-white/10" />
                            </div>

                            {/* Login Link */}
                            <p className="text-center text-slate-400">
                                Already have an account?{' '}
                                <Link href="/login" className="text-red-400 hover:text-red-300 font-medium transition-colors">
                                    Sign in
                                </Link>
                            </p>
                        </motion.div>
                    )}

                    {/* Back to Home */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.6 }}
                        className="text-center mt-6"
                    >
                        <Link href="/" className="text-slate-500 hover:text-white text-sm transition-colors flex items-center justify-center gap-2">
                            <Heart className="w-4 h-4" />
                            Back to CardioDetect
                        </Link>
                    </motion.div>
                </motion.div>
            </div>

            {/* Right Side - Branding */}
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
                        className="text-slate-400 text-center max-w-md text-lg mb-10"
                    >
                        Start your journey to better heart health with AI-powered insights
                    </motion.p>

                    {/* Features List */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                        className="space-y-4"
                    >
                        {[
                            'Instant heart disease detection',
                            '10-year risk prediction',
                            'OCR medical report scanning',
                            'SHAP-powered explanations',
                        ].map((feature, index) => (
                            <motion.div
                                key={feature}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.6 + index * 0.1 }}
                                className="flex items-center gap-3 text-slate-300"
                            >
                                <div className="w-6 h-6 rounded-full bg-green-500/20 flex items-center justify-center">
                                    <Check className="w-4 h-4 text-green-400" />
                                </div>
                                {feature}
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </div>
        </div>
    );
}
