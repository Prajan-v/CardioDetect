'use client';

import { motion } from 'framer-motion';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '@/context/ThemeContext';

interface ThemeToggleProps {
    size?: 'sm' | 'md' | 'lg';
    showLabel?: boolean;
}

export default function ThemeToggle({ size = 'md', showLabel = false }: ThemeToggleProps) {
    const { theme, toggleTheme } = useTheme();

    const sizes = {
        sm: { button: 'w-12 h-6', circle: 'w-4 h-4', icon: 'w-3 h-3', translate: '24px' },
        md: { button: 'w-14 h-7', circle: 'w-5 h-5', icon: 'w-3.5 h-3.5', translate: '28px' },
        lg: { button: 'w-16 h-8', circle: 'w-6 h-6', icon: 'w-4 h-4', translate: '32px' },
    };

    const s = sizes[size];

    return (
        <div className="flex items-center gap-2">
            {showLabel && (
                <span className="text-sm text-slate-400">
                    {theme === 'dark' ? 'Dark' : 'Light'}
                </span>
            )}
            <button
                onClick={toggleTheme}
                className={`relative ${s.button} rounded-full transition-colors duration-300 ${theme === 'dark'
                        ? 'bg-slate-700'
                        : 'bg-yellow-100'
                    }`}
                aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
                <motion.div
                    className={`absolute top-1 ${s.circle} rounded-full flex items-center justify-center ${theme === 'dark'
                            ? 'bg-slate-900'
                            : 'bg-yellow-400'
                        }`}
                    animate={{
                        left: theme === 'dark' ? '4px' : s.translate,
                    }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                >
                    {theme === 'dark' ? (
                        <Moon className={`${s.icon} text-blue-400`} />
                    ) : (
                        <Sun className={`${s.icon} text-yellow-600`} />
                    )}
                </motion.div>
            </button>
        </div>
    );
}
