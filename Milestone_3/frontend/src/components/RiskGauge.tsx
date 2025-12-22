'use client';

import { motion } from 'framer-motion';

interface RiskGaugeProps {
    value: number; // 0-100
    size?: number;
    label?: string;
    animated?: boolean;
}

export default function RiskGauge({
    value,
    size = 200,
    label = 'Risk Score',
    animated = true,
}: RiskGaugeProps) {
    // Clamp value between 0 and 100
    const clampedValue = Math.max(0, Math.min(100, value));

    // Calculate angle for the needle (0 = -90deg, 100 = 90deg)
    const needleAngle = -90 + (clampedValue / 100) * 180;

    // Determine risk level and color
    const getRiskInfo = (val: number) => {
        if (val < 10) return { level: 'LOW', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.2)' };
        if (val < 25) return { level: 'MODERATE', color: '#eab308', bgColor: 'rgba(234, 179, 8, 0.2)' };
        return { level: 'HIGH', color: '#ef4444', bgColor: 'rgba(239, 68, 68, 0.2)' };
    };

    const riskInfo = getRiskInfo(clampedValue);

    // SVG dimensions
    const strokeWidth = 12;
    const radius = (size - strokeWidth) / 2 - 10;
    const center = size / 2;

    // Arc path for semicircle
    const arcPath = `
        M ${center - radius} ${center}
        A ${radius} ${radius} 0 0 1 ${center + radius} ${center}
    `;

    // Calculate the dash offset for fill animation
    const circumference = Math.PI * radius;
    const dashOffset = circumference - (clampedValue / 100) * circumference;

    return (
        <div className="flex flex-col items-center">
            <div className="relative" style={{ width: size, height: size / 2 + 40 }}>
                <svg
                    width={size}
                    height={size / 2 + 20}
                    viewBox={`0 0 ${size} ${size / 2 + 20}`}
                    className="overflow-visible"
                >
                    {/* Background track */}
                    <path
                        d={arcPath}
                        fill="none"
                        stroke="rgba(255,255,255,0.1)"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                    />

                    {/* Gradient definition */}
                    <defs>
                        <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#22c55e" />
                            <stop offset="40%" stopColor="#eab308" />
                            <stop offset="100%" stopColor="#ef4444" />
                        </linearGradient>
                    </defs>

                    {/* Colored arc */}
                    <motion.path
                        d={arcPath}
                        fill="none"
                        stroke="url(#gaugeGradient)"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: animated ? circumference : dashOffset }}
                        animate={{ strokeDashoffset: dashOffset }}
                        transition={{ duration: animated ? 1.5 : 0, ease: "easeOut" }}
                    />

                    {/* Zone labels */}
                    <text x={center - radius + 10} y={center + 15} fill="#22c55e" fontSize="10" fontWeight="500">LOW</text>
                    <text x={center - 18} y={center - radius + 25} fill="#eab308" fontSize="10" fontWeight="500">MOD</text>
                    <text x={center + radius - 30} y={center + 15} fill="#ef4444" fontSize="10" fontWeight="500">HIGH</text>

                    {/* Needle */}
                    <motion.g
                        initial={{ rotate: animated ? -90 : needleAngle }}
                        animate={{ rotate: needleAngle }}
                        transition={{ duration: animated ? 1.5 : 0, ease: "easeOut" }}
                        style={{ transformOrigin: `${center}px ${center}px` }}
                    >
                        <line
                            x1={center}
                            y1={center}
                            x2={center}
                            y2={center - radius + 20}
                            stroke={riskInfo.color}
                            strokeWidth="3"
                            strokeLinecap="round"
                        />
                        <circle
                            cx={center}
                            cy={center}
                            r="8"
                            fill={riskInfo.color}
                        />
                    </motion.g>
                </svg>

                {/* Value display */}
                <div className="absolute left-1/2 transform -translate-x-1/2" style={{ top: size / 2 - 20 }}>
                    <motion.div
                        initial={{ opacity: 0, scale: 0.5 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: animated ? 0.5 : 0, duration: 0.5 }}
                        className="text-center"
                    >
                        <div className="text-4xl font-bold" style={{ color: riskInfo.color }}>
                            {clampedValue.toFixed(1)}%
                        </div>
                        <div
                            className="text-sm font-semibold px-3 py-1 rounded-full mt-1"
                            style={{
                                color: riskInfo.color,
                                backgroundColor: riskInfo.bgColor
                            }}
                        >
                            {riskInfo.level} RISK
                        </div>
                    </motion.div>
                </div>
            </div>

            {/* Label */}
            <div className="text-slate-400 text-sm mt-2">{label}</div>
        </div>
    );
}
