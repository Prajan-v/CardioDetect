'use client';

import { motion } from 'framer-motion';
import { Shield, ArrowLeft, Heart, Lock, Eye, Database } from 'lucide-react';
import Link from 'next/link';
import AnimatedHeart from '@/components/AnimatedHeart';
import FloatingParticles from '@/components/FloatingParticles';

export default function PrivacyPage() {
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
                        <div className="w-14 h-14 rounded-2xl bg-green-500/20 flex items-center justify-center">
                            <Shield className="w-7 h-7 text-green-400" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-white">Privacy Policy</h1>
                            <p className="text-slate-400">Last updated: December 2025</p>
                        </div>
                    </div>

                    {/* Privacy Highlights */}
                    <div className="grid md:grid-cols-3 gap-4 mb-8">
                        {[
                            { icon: Lock, title: 'Encrypted', desc: 'End-to-end encryption', color: 'text-blue-400' },
                            { icon: Eye, title: 'Private', desc: 'No data sharing', color: 'text-green-400' },
                            { icon: Database, title: 'Secure', desc: 'HIPAA-compliant', color: 'text-purple-400' },
                        ].map((item) => (
                            <div key={item.title} className="p-4 rounded-xl bg-white/5 border border-white/10">
                                <item.icon className={`w-6 h-6 ${item.color} mb-2`} />
                                <h3 className="text-white font-medium">{item.title}</h3>
                                <p className="text-slate-500 text-sm">{item.desc}</p>
                            </div>
                        ))}
                    </div>

                    <div className="prose prose-invert max-w-none">
                        <div className="space-y-6 text-slate-300">
                            <section>
                                <h2 className="text-xl font-semibold text-white mb-3">Data Collection</h2>
                                <p className="leading-relaxed">
                                    We collect health data you provide (age, blood pressure, cholesterol, etc.)
                                    solely for cardiovascular risk assessment. We do not sell or share your
                                    personal health information with third parties.
                                </p>
                            </section>

                            <section>
                                <h2 className="text-xl font-semibold text-white mb-3">Data Security</h2>
                                <p className="leading-relaxed">
                                    All data is encrypted in transit and at rest. We follow industry-standard
                                    security practices to protect your sensitive health information.
                                </p>
                            </section>

                            <section>
                                <h2 className="text-xl font-semibold text-white mb-3">Your Rights</h2>
                                <p className="leading-relaxed">
                                    You can request deletion of your data at any time. Contact us to exercise
                                    your privacy rights.
                                </p>
                            </section>

                            <div className="mt-8 p-6 rounded-xl bg-green-500/10 border border-green-500/30">
                                <p className="text-green-400 text-sm flex items-center gap-2">
                                    <Heart className="w-5 h-5" />
                                    Full privacy policy will be available soon. This is a placeholder.
                                </p>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
