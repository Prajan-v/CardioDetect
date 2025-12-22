'use client';

import { motion } from 'framer-motion';

export default function AnatomicalHeart({ className = '' }: { className?: string }) {
    return (
        <motion.div
            className={`relative ${className}`}
            initial={{ scale: 1 }}
            animate={{
                scale: [1, 1.08, 1, 1.04, 1],
            }}
            transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: "easeInOut"
            }}
        >
            {/* Glow Effect */}
            <div className="absolute inset-0 blur-2xl bg-red-500/40 rounded-full" />

            {/* Anatomical Heart SVG */}
            <svg
                viewBox="0 0 100 100"
                className="w-full h-full relative z-10 drop-shadow-[0_0_25px_rgba(239,68,68,0.6)]"
            >
                <defs>
                    {/* Gradient for main heart */}
                    <linearGradient id="heartGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#ef4444" />
                        <stop offset="50%" stopColor="#dc2626" />
                        <stop offset="100%" stopColor="#991b1b" />
                    </linearGradient>

                    {/* Gradient for aorta */}
                    <linearGradient id="aortaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#f87171" />
                        <stop offset="100%" stopColor="#dc2626" />
                    </linearGradient>

                    {/* Glow filter */}
                    <filter id="heartGlow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Aorta - top vessels */}
                <g filter="url(#heartGlow)">
                    {/* Main Aorta */}
                    <path
                        d="M50 15 Q50 5, 55 5 Q65 5, 65 15 L65 25 Q65 28, 62 30"
                        fill="none"
                        stroke="url(#aortaGradient)"
                        strokeWidth="6"
                        strokeLinecap="round"
                    />

                    {/* Pulmonary Artery */}
                    <path
                        d="M40 20 Q35 10, 30 12 Q22 14, 25 25"
                        fill="none"
                        stroke="url(#aortaGradient)"
                        strokeWidth="5"
                        strokeLinecap="round"
                    />

                    {/* Superior Vena Cava */}
                    <path
                        d="M65 22 Q72 15, 78 18 Q82 22, 78 30"
                        fill="none"
                        stroke="#f87171"
                        strokeWidth="4"
                        strokeLinecap="round"
                    />
                </g>

                {/* Main Heart Body */}
                <g filter="url(#heartGlow)">
                    {/* Left Ventricle (larger, left side) */}
                    <path
                        d="M30 30 
               Q15 35, 12 50 
               Q10 70, 30 85 
               Q45 95, 50 90"
                        fill="url(#heartGradient)"
                        stroke="#b91c1c"
                        strokeWidth="1"
                    />

                    {/* Right Ventricle (smaller, right side) */}
                    <path
                        d="M70 30 
               Q85 35, 88 48 
               Q90 65, 75 80 
               Q60 92, 50 90"
                        fill="url(#heartGradient)"
                        stroke="#b91c1c"
                        strokeWidth="1"
                    />

                    {/* Left Atrium */}
                    <ellipse
                        cx="35"
                        cy="32"
                        rx="12"
                        ry="10"
                        fill="#ef4444"
                        stroke="#b91c1c"
                        strokeWidth="1"
                    />

                    {/* Right Atrium */}
                    <ellipse
                        cx="65"
                        cy="32"
                        rx="12"
                        ry="10"
                        fill="#f87171"
                        stroke="#dc2626"
                        strokeWidth="1"
                    />

                    {/* Septum line */}
                    <path
                        d="M50 35 Q48 55, 50 75 Q51 85, 50 90"
                        fill="none"
                        stroke="#991b1b"
                        strokeWidth="1.5"
                        opacity="0.6"
                    />

                    {/* Highlight on left ventricle */}
                    <ellipse
                        cx="28"
                        cy="55"
                        rx="6"
                        ry="12"
                        fill="rgba(255,255,255,0.15)"
                    />
                </g>
            </svg>

            {/* Pulse Rings */}
            <motion.div
                className="absolute inset-0 border border-red-500/20 rounded-full"
                initial={{ scale: 1, opacity: 0.5 }}
                animate={{ scale: 2, opacity: 0 }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeOut" }}
            />
            <motion.div
                className="absolute inset-0 border border-red-500/15 rounded-full"
                initial={{ scale: 1, opacity: 0.3 }}
                animate={{ scale: 2.5, opacity: 0 }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeOut", delay: 1.5 }}
            />
        </motion.div>
    );
}
