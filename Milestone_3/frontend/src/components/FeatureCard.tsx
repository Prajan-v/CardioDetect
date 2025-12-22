'use client';

import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';

interface FeatureCardProps {
    icon: LucideIcon;
    title: string;
    description: string;
    delay?: number;
}

export default function FeatureCard({ icon: Icon, title, description, delay = 0 }: FeatureCardProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay }}
            whileHover={{ y: -8, scale: 1.02 }}
            className="glass-card glass-card-hover p-8 cursor-pointer group"
        >
            {/* Icon Container */}
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-red-500/20 to-purple-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                <Icon className="w-7 h-7 text-red-400 group-hover:text-red-300 transition-colors" />
            </div>

            {/* Title */}
            <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-red-300 transition-colors">
                {title}
            </h3>

            {/* Description */}
            <p className="text-slate-400 leading-relaxed group-hover:text-slate-300 transition-colors">
                {description}
            </p>

            {/* Bottom Glow Line */}
            <motion.div
                className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-red-500/50 to-transparent"
                initial={{ scaleX: 0 }}
                whileHover={{ scaleX: 1 }}
                transition={{ duration: 0.3 }}
            />
        </motion.div>
    );
}
