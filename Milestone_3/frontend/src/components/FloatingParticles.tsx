'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';

interface Particle {
    id: number;
    x: number;
    y: number;
    size: number;
    duration: number;
    delay: number;
    type: 'heart' | 'dot' | 'cross';
}

export default function FloatingParticles() {
    const [particles, setParticles] = useState<Particle[]>([]);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        const newParticles: Particle[] = [];
        for (let i = 0; i < 20; i++) {
            newParticles.push({
                id: i,
                x: Math.random() * 100,
                y: Math.random() * 100,
                size: Math.random() * 8 + 4,
                duration: Math.random() * 15 + 10,
                delay: Math.random() * 5,
                type: ['heart', 'dot', 'cross'][Math.floor(Math.random() * 3)] as 'heart' | 'dot' | 'cross',
            });
        }
        setParticles(newParticles);
    }, []);

    if (!mounted) return null;

    const renderParticle = (type: string, size: number) => {
        switch (type) {
            case 'heart':
                return (
                    <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: size, height: size }}>
                        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                    </svg>
                );
            case 'cross':
                return (
                    <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: size, height: size }}>
                        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z" />
                    </svg>
                );
            default:
                return (
                    <div
                        style={{ width: size, height: size }}
                        className="rounded-full bg-current"
                    />
                );
        }
    };

    return (
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
            {particles.map((particle) => (
                <motion.div
                    key={particle.id}
                    className="absolute text-red-500/10"
                    style={{
                        left: `${particle.x}%`,
                        top: `${particle.y}%`,
                    }}
                    animate={{
                        y: [0, -30, 0],
                        x: [0, 10, -10, 0],
                        opacity: [0.05, 0.15, 0.05],
                        rotate: [0, 180, 360],
                    }}
                    transition={{
                        duration: particle.duration,
                        repeat: Infinity,
                        delay: particle.delay,
                        ease: 'easeInOut',
                    }}
                >
                    {renderParticle(particle.type, particle.size)}
                </motion.div>
            ))}
        </div>
    );
}
