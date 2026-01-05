'use client';

import { motion } from 'framer-motion';
import { Heart, Shield, Zap, FileSearch, Brain, Activity, ChevronRight, ArrowRight } from 'lucide-react';
import AnimatedHeart from '@/components/AnimatedHeart';
import HeartWithStethoscope from '@/components/HeartWithStethoscope';
import ECGLine from '@/components/ECGLine';
import FeatureCard from '@/components/FeatureCard';
import FloatingParticles from '@/components/FloatingParticles';
import AnimatedCounter from '@/components/AnimatedCounter';
import Link from 'next/link';

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Detection',
    description: 'Advanced machine learning models trained on thousands of cardiac cases for accurate heart disease detection.',
  },
  {
    icon: FileSearch,
    title: 'OCR Document Scan',
    description: 'Upload medical reports and let our AI extract vital information automatically using OCR technology.',
  },
  {
    icon: Activity,
    title: 'Real-Time Analysis',
    description: 'Get instant risk assessment with detailed breakdown of contributing factors and recommendations.',
  },
  {
    icon: Shield,
    title: 'HIPAA Compliant',
    description: 'Your medical data is encrypted and protected with enterprise-grade security standards.',
  },
  {
    icon: Zap,
    title: '91%+ Accuracy',
    description: 'Our models achieve over 91% accuracy in heart disease prediction, validated on clinical datasets.',
  },
  {
    icon: Heart,
    title: 'Comprehensive Care',
    description: 'From detection to prediction, get a complete cardiovascular risk profile in minutes.',
  },
];

const stats = [
  { value: 91.45, suffix: '%', label: 'Detection Accuracy' },
  { value: 91.63, suffix: '%', label: 'Prediction Accuracy' },
  { value: 5000, suffix: '+', label: 'Training Samples' },
  { value: 3, prefix: '<', suffix: 's', label: 'Analysis Time' },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0a0a1a] overflow-hidden relative">
      {/* Background Effects */}
      <div className="fixed inset-0 mesh-bg pointer-events-none" />
      <div className="gradient-orb orb-1" />
      <div className="gradient-orb orb-2" />
      <div className="gradient-orb orb-3" />

      {/* Floating Medical Particles */}
      <FloatingParticles />

      {/* ECG Background Animation */}
      <div className="fixed top-1/4 left-0 right-0 h-20 opacity-20 pointer-events-none">
        <ECGLine className="w-full h-full" />
      </div>

      {/* Navigation */}
      <nav className="relative z-50">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="w-10 h-10 relative">
                <AnimatedHeart className="w-full h-full" />
              </div>
              <span className="text-2xl font-bold gradient-text">CardioDetect</span>
            </motion.div>

            {/* Nav Links */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="hidden md:flex items-center gap-8"
            >
              <a href="#features" className="text-slate-400 hover:text-white transition-colors">Features</a>
              <a href="#stats" className="text-slate-400 hover:text-white transition-colors">Statistics</a>
              <a href="#about" className="text-slate-400 hover:text-white transition-colors">About</a>
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="flex items-center gap-4"
            >
              <Link href="/login" className="text-slate-400 hover:text-white transition-colors px-4 py-2">
                Login
              </Link>
              <Link
                href="/register"
                className="glow-button text-white px-6 py-2.5 rounded-full font-medium flex items-center gap-2"
              >
                Get Started <ChevronRight className="w-4 h-4" />
              </Link>
            </motion.div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 pt-12 md:pt-20 pb-16 md:pb-32 px-4 md:px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8 md:gap-16 items-center">
            {/* Left Content */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center lg:text-left"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
                className="inline-flex items-center gap-2 glass-card px-4 py-2 mb-6 md:mb-8"
              >
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-xs md:text-sm text-slate-300">AI-Powered Heart Disease Detection</span>
              </motion.div>

              <h1 className="text-4xl md:text-5xl lg:text-7xl font-bold leading-tight mb-6 md:mb-8">
                <span className="text-white">Protect Your</span>
                <br />
                <span className="gradient-text">Heart Health</span>
              </h1>

              <p className="text-lg md:text-xl text-slate-400 leading-relaxed mb-8 md:mb-10 max-w-lg mx-auto lg:mx-0">
                Advanced AI technology for early detection and prediction of cardiovascular diseases.
                Get your risk assessment in seconds with 91%+ accuracy.
              </p>

              <div className="flex flex-wrap gap-4">
                <Link
                  href="/register"
                  className="glow-button text-white px-8 py-4 rounded-full font-semibold text-lg flex items-center gap-3"
                >
                  Start Free Analysis
                  <ArrowRight className="w-5 h-5" />
                </Link>
                <button className="glass-card px-8 py-4 rounded-full font-medium text-white hover:bg-white/10 transition-colors flex items-center gap-3">
                  <Activity className="w-5 h-5 text-red-400" />
                  Watch Demo
                </button>
              </div>
            </motion.div>

            {/* Right - Animated Heart */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.4 }}
              className="relative flex justify-center items-center"
            >
              {/* Outer Ring */}
              <div className="absolute w-[400px] h-[400px] border border-red-500/20 rounded-full animate-pulse" />
              <div className="absolute w-[500px] h-[500px] border border-purple-500/10 rounded-full" />

              {/* ECG Circle */}
              <div className="absolute w-[350px] h-[350px] rounded-full overflow-hidden opacity-30">
                <ECGLine className="w-full h-full" />
              </div>

              {/* Main Heart */}
              <div className="w-48 h-48 floating">
                <AnimatedHeart className="w-full h-full" />
              </div>

            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section id="stats" className="relative z-10 py-12 md:py-20 px-4 md:px-6">
        <div className="max-w-7xl mx-auto">
          <div className="glass-card p-6 md:p-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-8">
              {stats.map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="text-center"
                >
                  <div className="text-3xl md:text-5xl font-bold gradient-text mb-2">
                    <AnimatedCounter
                      value={stat.value}
                      prefix={stat.prefix || ''}
                      suffix={stat.suffix}
                      duration={2}
                    />
                  </div>
                  <div className="text-sm md:text-base text-slate-400">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Two Models Section */}
      <section id="models" className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Two Powerful <span className="gradient-text">AI Models</span>
            </h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Comprehensive cardiac analysis combining instant detection with long-term risk prediction
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Detection Model */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass-card p-8 relative overflow-hidden group"
            >
              <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/10 rounded-full blur-3xl group-hover:bg-red-500/20 transition-colors" />
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-red-500/20 flex items-center justify-center">
                  <Zap className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white">Instant Detection</h3>
                  <span className="text-sm text-red-400">Real-time Analysis</span>
                </div>
              </div>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Immediate heart disease detection using clinical indicators. Get results in seconds
                with our ensemble machine learning model trained on the UCI Heart Disease dataset.
              </p>
              <ul className="space-y-3 text-slate-300">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                  13 clinical features analyzed
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                  91.45% accuracy on test data
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                  Includes stress test parameters
                </li>
              </ul>
            </motion.div>

            {/* Prediction Model */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass-card p-8 relative overflow-hidden group"
            >
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl group-hover:bg-blue-500/20 transition-colors" />
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 rounded-xl bg-blue-500/20 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white">10-Year Prediction</h3>
                  <span className="text-sm text-blue-400">Long-term Risk</span>
                </div>
              </div>
              <p className="text-slate-400 mb-6 leading-relaxed">
                Predict your 10-year cardiovascular disease risk using the Framingham Heart Study
                methodology. Plan ahead with evidence-based risk stratification.
              </p>
              <ul className="space-y-3 text-slate-300">
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                  Based on Framingham Risk Score
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                  91.63% prediction accuracy
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-blue-400 rounded-full" />
                  Low / Moderate / High risk levels
                </li>
              </ul>
            </motion.div>
          </div>

          {/* Clinical Guidelines - Enhanced */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="mt-16 relative"
          >
            {/* Background Glow */}
            <div className="absolute inset-0 bg-gradient-to-r from-red-500/10 via-purple-500/10 to-blue-500/10 blur-3xl" />

            <div className="glass-card p-10 relative overflow-hidden">
              {/* Animated border */}
              <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-red-500/20 via-purple-500/20 to-blue-500/20 animate-pulse" style={{ padding: '1px' }}>
                <div className="w-full h-full bg-[#0a0a1a] rounded-3xl" />
              </div>

              <div className="relative z-10">
                <motion.div
                  initial={{ scale: 0.9 }}
                  whileInView={{ scale: 1 }}
                  viewport={{ once: true }}
                  className="flex items-center justify-center gap-3 mb-6"
                >
                  <Shield className="w-8 h-8 text-green-400" />
                  <h3 className="text-2xl font-bold text-white">
                    Built on <span className="gradient-text">Clinical Guidelines</span>
                  </h3>
                </motion.div>

                <p className="text-slate-400 mb-10 max-w-2xl mx-auto text-center text-lg">
                  Our models follow established clinical frameworks trusted by cardiologists worldwide
                </p>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { name: 'UCI Heart Disease', icon: '🏥', desc: '303 clinical cases' },
                    { name: 'Framingham Study', icon: '📊', desc: '50+ years research' },
                    { name: 'ACC/AHA Guidelines', icon: '📋', desc: 'Gold standard' },
                    { name: 'Cleveland Clinic', icon: '🔬', desc: 'Leading cardiac care' },
                  ].map((item, index) => (
                    <motion.div
                      key={item.name}
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.05, y: -5 }}
                      className="glass-card p-5 border border-white/10 hover:border-white/30 transition-all cursor-pointer group"
                    >
                      <span className="text-3xl mb-3 block group-hover:scale-110 transition-transform">{item.icon}</span>
                      <h4 className="text-white font-semibold mb-1">{item.name}</h4>
                      <p className="text-slate-500 text-sm">{item.desc}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Powerful <span className="gradient-text">Features</span>
            </h2>
            <p className="text-xl text-slate-400 max-w-2xl mx-auto">
              Everything you need for comprehensive cardiovascular health monitoring
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                delay={index * 0.1}
              />
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 py-32 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-6xl font-bold text-white mb-8">
              Ready to Check Your <span className="gradient-text">Heart Health?</span>
            </h2>
            <p className="text-xl text-slate-400 mb-10 max-w-2xl mx-auto">
              Take control of your cardiovascular health with AI-powered insights.
              Get accurate risk predictions in seconds, not hours.
            </p>
            <Link
              href="/register"
              className="glow-button inline-flex items-center gap-3 text-white px-10 py-5 rounded-full font-semibold text-xl"
            >
              Get Started Free
              <ArrowRight className="w-6 h-6" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 py-12 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8">
                <AnimatedHeart className="w-full h-full" />
              </div>
              <span className="text-xl font-bold gradient-text">CardioDetect</span>
            </div>

            {/* Developer Credit */}
            <div className="flex flex-col items-center">
              <p className="text-slate-500 text-sm">
                © 2025 CardioDetect. AI-Powered Heart Disease Detection.
              </p>

            </div>

            <div className="flex gap-6">
              <a href="#" className="text-slate-400 hover:text-white transition-colors text-sm">Privacy</a>
              <a href="#" className="text-slate-400 hover:text-white transition-colors text-sm">Terms</a>
              <a href="mailto:cardiodetect.care@gmail.com" className="text-slate-400 hover:text-white transition-colors text-sm">Contact</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
