'use client';

import { motion } from 'framer-motion';

interface StethoscopeHeartLogoProps {
    className?: string;
    size?: number;
    animated?: boolean;
}

export default function StethoscopeHeartLogo({
    className = '',
    size = 40,
    animated = true
}: StethoscopeHeartLogoProps) {
    return (
        <motion.svg
            viewBox="0 0 100 120"
            className={className}
            style={{ width: size, height: size * 1.2 }}
            initial={animated ? { opacity: 0, scale: 0.8 } : {}}
            animate={animated ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.5 }}
        >
            <defs>
                {/* Tube gradient - dark with shine */}
                <linearGradient id="tubeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#374151" />
                    <stop offset="30%" stopColor="#4b5563" />
                    <stop offset="50%" stopColor="#6b7280" />
                    <stop offset="70%" stopColor="#4b5563" />
                    <stop offset="100%" stopColor="#374151" />
                </linearGradient>

                {/* Red gradient for heart glow effect */}
                <linearGradient id="heartGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#ef4444" />
                    <stop offset="100%" stopColor="#dc2626" />
                </linearGradient>

                {/* Metallic gradient for chest piece */}
                <radialGradient id="chestPiece" cx="50%" cy="30%" r="60%">
                    <stop offset="0%" stopColor="#9ca3af" />
                    <stop offset="50%" stopColor="#6b7280" />
                    <stop offset="100%" stopColor="#374151" />
                </radialGradient>

                {/* Drop shadow */}
                <filter id="logoShadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="1" dy="2" stdDeviation="2" floodOpacity="0.3" />
                </filter>
            </defs>

            {/* Heart shape formed by stethoscope tube */}
            <motion.path
                d="M50 20 
           C30 0, 5 10, 10 35
           C15 55, 35 70, 50 85
           C65 70, 85 55, 90 35
           C95 10, 70 0, 50 20"
                fill="none"
                stroke="url(#tubeGrad)"
                strokeWidth="8"
                strokeLinecap="round"
                strokeLinejoin="round"
                filter="url(#logoShadow)"
                animate={animated ? {
                    strokeDashoffset: [0, -10, 0],
                } : {}}
                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            />

            {/* Tube crossing at bottom of heart - going down */}
            <path
                d="M50 85 L50 95"
                fill="none"
                stroke="url(#tubeGrad)"
                strokeWidth="6"
                strokeLinecap="round"
            />

            {/* Y-split for earpieces */}
            <path
                d="M50 95 Q50 100, 42 105"
                fill="none"
                stroke="url(#tubeGrad)"
                strokeWidth="5"
                strokeLinecap="round"
            />
            <path
                d="M50 95 Q50 100, 58 105"
                fill="none"
                stroke="url(#tubeGrad)"
                strokeWidth="5"
                strokeLinecap="round"
            />

            {/* Metallic Y-junction */}
            <circle cx="50" cy="95" r="4" fill="#9ca3af" stroke="#6b7280" strokeWidth="1" />

            {/* Earpiece tubes */}
            <path
                d="M42 105 Q35 108, 32 115"
                fill="none"
                stroke="#9ca3af"
                strokeWidth="3"
                strokeLinecap="round"
            />
            <path
                d="M58 105 Q65 108, 68 115"
                fill="none"
                stroke="#9ca3af"
                strokeWidth="3"
                strokeLinecap="round"
            />

            {/* Earpieces */}
            <ellipse cx="30" cy="117" rx="5" ry="4" fill="#6b7280" />
            <ellipse cx="70" cy="117" rx="5" ry="4" fill="#6b7280" />

            {/* Chest piece (diaphragm) at top right */}
            <motion.g
                animate={animated ? { rotate: [-5, 5, -5] } : {}}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                style={{ transformOrigin: '75px 50px' }}
            >
                <circle cx="82" cy="55" r="12" fill="url(#chestPiece)" filter="url(#logoShadow)" />
                <circle cx="82" cy="55" r="8" fill="#374151" />
                <circle cx="82" cy="55" r="4" fill="#4b5563" />
                <circle cx="80" cy="53" r="2" fill="rgba(255,255,255,0.2)" />

                {/* Tube connecting to chest piece */}
                <path
                    d="M90 35 Q95 45, 82 55"
                    fill="none"
                    stroke="url(#tubeGrad)"
                    strokeWidth="5"
                    strokeLinecap="round"
                />
            </motion.g>

            {/* Optional: Small red heart in center for color accent */}
            <motion.path
                d="M50 40 
           C45 35, 38 38, 40 45
           C42 50, 50 55, 50 55
           C50 55, 58 50, 60 45
           C62 38, 55 35, 50 40"
                fill="url(#heartGlow)"
                opacity="0.8"
                animate={animated ? { scale: [1, 1.1, 1] } : {}}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                style={{ transformOrigin: '50px 47px' }}
            />
        </motion.svg>
    );
}
