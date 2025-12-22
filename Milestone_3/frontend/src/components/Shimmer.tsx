'use client';

import { motion } from 'framer-motion';

interface ShimmerProps {
    className?: string;
    width?: string;
    height?: string;
    rounded?: string;
}

export default function Shimmer({
    className = '',
    width = 'w-full',
    height = 'h-4',
    rounded = 'rounded-lg'
}: ShimmerProps) {
    return (
        <div className={`${width} ${height} ${rounded} ${className} relative overflow-hidden bg-slate-800/50`}>
            <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                animate={{
                    x: ['-100%', '100%'],
                }}
                transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: 'linear',
                }}
            />
        </div>
    );
}

// Card shimmer for loading states
export function ShimmerCard({ className = '' }: { className?: string }) {
    return (
        <div className={`glass-card p-6 ${className}`}>
            <Shimmer height="h-8" width="w-1/3" className="mb-4" />
            <Shimmer height="h-4" className="mb-2" />
            <Shimmer height="h-4" width="w-3/4" className="mb-4" />
            <Shimmer height="h-10" width="w-1/2" />
        </div>
    );
}

// Avatar shimmer
export function ShimmerAvatar({ size = 'w-12 h-12' }: { size?: string }) {
    return <Shimmer width={size.split(' ')[0]} height={size.split(' ')[1]} rounded="rounded-full" />;
}

// Text line shimmer
export function ShimmerText({ lines = 3, className = '' }: { lines?: number; className?: string }) {
    return (
        <div className={`space-y-2 ${className}`}>
            {Array.from({ length: lines }).map((_, i) => (
                <Shimmer
                    key={i}
                    height="h-4"
                    width={i === lines - 1 ? 'w-2/3' : 'w-full'}
                />
            ))}
        </div>
    );
}

// Stats card shimmer
export function ShimmerStats({ className = '' }: { className?: string }) {
    return (
        <div className={`glass-card p-6 text-center ${className}`}>
            <Shimmer height="h-10" width="w-24" className="mx-auto mb-2" />
            <Shimmer height="h-4" width="w-20" className="mx-auto" />
        </div>
    );
}

// Form field shimmer
export function ShimmerFormField({ className = '' }: { className?: string }) {
    return (
        <div className={className}>
            <Shimmer height="h-4" width="w-24" className="mb-2" />
            <Shimmer height="h-12" rounded="rounded-xl" />
        </div>
    );
}

// Button shimmer
export function ShimmerButton({ className = '' }: { className?: string }) {
    return <Shimmer height="h-12" rounded="rounded-xl" className={className} />;
}

// Feature card shimmer
export function ShimmerFeatureCard({ className = '' }: { className?: string }) {
    return (
        <div className={`glass-card p-6 ${className}`}>
            <Shimmer height="h-12" width="w-12" rounded="rounded-xl" className="mb-4" />
            <Shimmer height="h-6" width="w-3/4" className="mb-3" />
            <ShimmerText lines={2} />
        </div>
    );
}

// Page skeleton for dashboard
export function ShimmerDashboard() {
    return (
        <div className="space-y-8 p-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <Shimmer height="h-8" width="w-48" className="mb-2" />
                    <Shimmer height="h-4" width="w-64" />
                </div>
                <ShimmerAvatar size="w-10 h-10" />
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[...Array(4)].map((_, i) => (
                    <ShimmerStats key={i} />
                ))}
            </div>

            {/* Content Cards */}
            <div className="grid md:grid-cols-2 gap-6">
                <ShimmerCard />
                <ShimmerCard />
            </div>
        </div>
    );
}
