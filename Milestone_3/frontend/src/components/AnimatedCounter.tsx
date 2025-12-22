'use client';

import { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

interface AnimatedCounterProps {
    value: number;
    suffix?: string;
    prefix?: string;
    duration?: number;
    className?: string;
}

export default function AnimatedCounter({
    value,
    suffix = '',
    prefix = '',
    duration = 2,
    className = ''
}: AnimatedCounterProps) {
    const [displayValue, setDisplayValue] = useState(0);
    const ref = useRef<HTMLSpanElement>(null);
    const isInView = useInView(ref, { once: true, amount: 0.5 });
    const hasAnimated = useRef(false);

    useEffect(() => {
        if (isInView && !hasAnimated.current) {
            hasAnimated.current = true;

            const startTime = Date.now();
            const endTime = startTime + duration * 1000;

            const animate = () => {
                const now = Date.now();
                const progress = Math.min((now - startTime) / (duration * 1000), 1);

                // Easing function - easeOutExpo
                const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
                const currentValue = value * eased;

                // Format based on whether value has decimals
                if (value % 1 !== 0) {
                    setDisplayValue(Math.round(currentValue * 100) / 100);
                } else {
                    setDisplayValue(Math.round(currentValue));
                }

                if (now < endTime) {
                    requestAnimationFrame(animate);
                } else {
                    setDisplayValue(value);
                }
            };

            requestAnimationFrame(animate);
        }
    }, [isInView, value, duration]);

    // Format the display value
    const formattedValue = value % 1 !== 0
        ? displayValue.toFixed(2)
        : displayValue.toString();

    return (
        <motion.span
            ref={ref}
            className={className}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
        >
            {prefix}{formattedValue}{suffix}
        </motion.span>
    );
}
