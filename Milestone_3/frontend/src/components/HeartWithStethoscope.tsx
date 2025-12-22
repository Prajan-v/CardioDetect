'use client';

import { motion } from 'framer-motion';
import AnimatedHeart from './AnimatedHeart';

export default function HeartWithStethoscope({ className = '' }: { className?: string }) {
    return (
        <div className={`relative ${className}`}>
            {/* Stethoscope SVG - Draped/Hanging over the heart like a scarf */}
            <motion.svg
                viewBox="0 0 200 200"
                className="absolute inset-0 w-full h-full z-10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 0.8 }}
            >
                <defs>
                    <linearGradient id="tubeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#475569" />
                        <stop offset="50%" stopColor="#64748b" />
                        <stop offset="100%" stopColor="#475569" />
                    </linearGradient>
                    <filter id="tubeShadow" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="2" dy="2" stdDeviation="3" floodOpacity="0.3" />
                    </filter>
                </defs>

                {/* Left side - hanging down from top-left shoulder */}
                <motion.g
                    animate={{ rotate: [-1, 1, -1] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                    style={{ transformOrigin: '50px 30px' }}
                >
                    {/* Left earpiece */}
                    <circle cx="25" cy="15" r="6" fill="#334155" stroke="#475569" strokeWidth="2" />
                    <circle cx="25" cy="15" r="3" fill="#1e293b" />

                    {/* Left tube going down */}
                    <path
                        d="M25 21 
               Q25 40, 35 55 
               Q50 75, 45 100
               Q40 125, 55 145
               Q65 160, 60 175"
                        fill="none"
                        stroke="url(#tubeGradient)"
                        strokeWidth="5"
                        strokeLinecap="round"
                        filter="url(#tubeShadow)"
                    />

                    {/* Left chest piece */}
                    <circle cx="60" cy="180" r="12" fill="#1e293b" stroke="#475569" strokeWidth="3" />
                    <circle cx="60" cy="180" r="7" fill="none" stroke="#64748b" strokeWidth="1.5" />
                    <circle cx="57" cy="177" r="2" fill="rgba(255,255,255,0.15)" />
                </motion.g>

                {/* Right side - hanging down from top-right shoulder */}
                <motion.g
                    animate={{ rotate: [1, -1, 1] }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
                    style={{ transformOrigin: '150px 30px' }}
                >
                    {/* Right earpiece */}
                    <circle cx="175" cy="15" r="6" fill="#334155" stroke="#475569" strokeWidth="2" />
                    <circle cx="175" cy="15" r="3" fill="#1e293b" />

                    {/* Right tube going down */}
                    <path
                        d="M175 21 
               Q175 40, 165 55 
               Q150 75, 155 100
               Q160 125, 145 145
               Q135 160, 140 175"
                        fill="none"
                        stroke="url(#tubeGradient)"
                        strokeWidth="5"
                        strokeLinecap="round"
                        filter="url(#tubeShadow)"
                    />

                    {/* Right chest piece */}
                    <circle cx="140" cy="180" r="12" fill="#1e293b" stroke="#475569" strokeWidth="3" />
                    <circle cx="140" cy="180" r="7" fill="none" stroke="#64748b" strokeWidth="1.5" />
                    <circle cx="137" cy="177" r="2" fill="rgba(255,255,255,0.15)" />
                </motion.g>

                {/* Connection tube at top - draped across like a scarf */}
                <path
                    d="M31 18 
             Q50 -5, 100 5 
             Q150 -5, 169 18"
                    fill="none"
                    stroke="url(#tubeGradient)"
                    strokeWidth="5"
                    strokeLinecap="round"
                    filter="url(#tubeShadow)"
                />

                {/* Y-junction piece at top center */}
                <circle cx="100" cy="5" r="8" fill="#334155" stroke="#475569" strokeWidth="2" />
                <circle cx="100" cy="5" r="4" fill="#1e293b" />
            </motion.svg>

            {/* Main Heart - Behind the stethoscope */}
            <div className="relative z-0">
                <AnimatedHeart className="w-full h-full" />
            </div>
        </div>
    );
}
