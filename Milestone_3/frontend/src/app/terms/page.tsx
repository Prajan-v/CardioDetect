'use client';

import { motion } from 'framer-motion';
import { FileText, ArrowLeft, Heart } from 'lucide-react';
import Link from 'next/link';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';

export default function TermsPage() {
    return (
        <div className="min-h-screen bg-[#0a0a1a] relative">
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            <div className="relative z-10 max-w-4xl mx-auto px-6 py-12">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-4 mb-12"
                >
                    <Link href="/" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        Back
                    </Link>
                    <div className="flex-1" />
                    <div className="w-8 h-8">
                        <AnimatedHeart className="w-full h-full" />
                    </div>
                    <span className="text-xl font-bold gradient-text">CardioDetect</span>
                </motion.div>

                {/* Content */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-card p-8 md:p-12"
                >
                    <div className="flex items-center gap-4 mb-8">
                        <div className="w-14 h-14 rounded-2xl bg-red-500/20 flex items-center justify-center">
                            <FileText className="w-7 h-7 text-red-400" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-white">Terms of Service</h1>
                            <p className="text-slate-400">Last updated: December 2025</p>
                        </div>
                    </div>

                    <div className="prose prose-invert max-w-none">
                        <div className="space-y-6 text-slate-300">
                            <section>
                                <h2 className="text-xl font-semibold text-white mb-3">1. Medical Disclaimer</h2>
                                <p className="leading-relaxed">
                                    CardioDetect is an AI-powered tool designed to assist in cardiovascular risk assessment.
                                    It is <strong className="text-red-400">NOT a substitute for professional medical advice</strong>,
                                    diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.
                                </p>
                            </section>

                            <section>
                                <h2 className="text-xl font-semibold text-white mb-3">2. AI Accuracy</h2>
                                <p className="leading-relaxed">
                                    Our models achieve 91.45% detection accuracy and 91.63% prediction accuracy.
                                    However, no AI system is 100% accurate. Users should use results as one of many
                                    factors in health decision-making.
                                </p>
                            </section>

                            <section>
                                <h2 className="text-xl font-semibold text-white mb-3">3. Service Usage</h2>
                                <p className="leading-relaxed">
                                    By using CardioDetect, you agree to provide accurate health information and
                                    understand that results are for informational purposes only.
                                </p>
                            </section>

                            <div className="mt-8 p-6 rounded-xl bg-yellow-500/10 border border-yellow-500/30">
                                <p className="text-yellow-400 text-sm flex items-center gap-2">
                                    <Heart className="w-5 h-5" />
                                    Full terms will be available soon. This is a placeholder.
                                </p>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
