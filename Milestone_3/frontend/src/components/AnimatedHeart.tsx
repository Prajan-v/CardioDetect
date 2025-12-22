'use client';

import { motion } from 'framer-motion';

export default function AnimatedHeart({ className = '' }: { className?: string }) {
    return (
        <motion.div
            className={`relative ${className}`}
            initial={{ scale: 1 }}
            animate={{
                scale: [1, 1.1, 1, 1.05, 1],
            }}
            transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: "easeInOut"
            }}
        >
            {/* Glow Effect */}
            <div className="absolute inset-0 blur-2xl bg-red-500/30 rounded-full" />

            {/* Heart SVG */}
            <svg
                viewBox="0 0 24 24"
                fill="currentColor"
                className="w-full h-full text-red-500 relative z-10 drop-shadow-[0_0_20px_rgba(239,68,68,0.8)]"
            >
                <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
            </svg>

            {/* Pulse Rings - Slower and more subtle */}
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
