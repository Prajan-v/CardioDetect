'use client';

import { motion } from 'framer-motion';

export default function ECGLine({ className = '' }: { className?: string }) {
    // ECG path - realistic P-QRS-T wave pattern
    const ecgPath = "M0,50 L20,50 L25,50 L30,45 L35,50 L40,50 L45,50 L50,50 L55,30 L60,80 L65,10 L70,60 L75,50 L80,50 L85,50 L90,45 L100,50 L110,50 L120,50";

    return (
        <div className={`overflow-hidden ${className}`}>
            <motion.svg
                viewBox="0 0 240 100"
                className="w-full h-full"
                preserveAspectRatio="none"
            >
                <defs>
                    <linearGradient id="ecg-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="rgba(239, 68, 68, 0)" />
                        <stop offset="50%" stopColor="rgba(239, 68, 68, 1)" />
                        <stop offset="100%" stopColor="rgba(239, 68, 68, 0)" />
                    </linearGradient>
                    <filter id="ecg-glow">
                        <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Animated ECG Line */}
                <g className="ecg-line">
                    <path
                        d={ecgPath}
                        fill="none"
                        stroke="url(#ecg-gradient)"
                        strokeWidth="2"
                        filter="url(#ecg-glow)"
                    />
                    <path
                        d={ecgPath}
                        fill="none"
                        stroke="url(#ecg-gradient)"
                        strokeWidth="2"
                        filter="url(#ecg-glow)"
                        transform="translate(120, 0)"
                    />
                </g>

                {/* Glowing dot that follows */}
                <motion.circle
                    r="4"
                    fill="#ef4444"
                    filter="url(#ecg-glow)"
                    initial={{ cx: 0, cy: 50 }}
                    animate={{ cx: 240 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                />
            </motion.svg>
        </div>
    );
}
