'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface ModelPerformanceCardProps {
    title: string;
    value: number;
    suffix?: string;
    icon: React.ReactNode;
    color: string;
    description?: string;
    animated?: boolean;
}

export default function ModelPerformanceCard({
    title,
    value,
    suffix = '%',
    icon,
    color,
    description,
    animated = true,
}: ModelPerformanceCardProps) {
    const [displayValue, setDisplayValue] = useState(animated ? 0 : value);

    useEffect(() => {
        if (!animated) return;

        const duration = 1500;
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            setDisplayValue(value * eased);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }, [value, animated]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card p-6 relative overflow-hidden group"
        >
            {/* Background glow */}
            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${color} blur-3xl`} />

            <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                    <div className={`w-12 h-12 rounded-xl ${color} flex items-center justify-center`}>
                        {icon}
                    </div>
                </div>

                <div className="text-3xl font-bold text-white mb-1">
                    {displayValue.toFixed(2)}{suffix}
                </div>

                <div className="text-sm font-medium text-slate-300">{title}</div>

                {description && (
                    <div className="text-xs text-slate-500 mt-2">{description}</div>
                )}
            </div>
        </motion.div>
    );
}
