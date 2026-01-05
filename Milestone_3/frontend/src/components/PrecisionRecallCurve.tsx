'use client';

import { motion } from 'framer-motion';

interface PrecisionRecallCurveProps {
    title?: string;
    modelName?: string;
    averagePrecision?: number;
    data?: { recall: number; precision: number }[];
    color?: string;
    animated?: boolean;
}

// Default PR curve data for Detection Model
const defaultData = [
    { recall: 0, precision: 1.0 },
    { recall: 0.10, precision: 0.98 },
    { recall: 0.25, precision: 0.96 },
    { recall: 0.40, precision: 0.94 },
    { recall: 0.55, precision: 0.91 },
    { recall: 0.70, precision: 0.87 },
    { recall: 0.80, precision: 0.82 },
    { recall: 0.88, precision: 0.75 },
    { recall: 0.94, precision: 0.65 },
    { recall: 0.98, precision: 0.50 },
    { recall: 1.0, precision: 0.42 },
];

export default function PrecisionRecallCurve({
    title = 'Precision-Recall Curve',
    modelName = 'Voting Ensemble',
    averagePrecision = 0.89,
    data = defaultData,
    color = '#3b82f6',
    animated = true,
}: PrecisionRecallCurveProps) {
    const width = 280;
    const height = 220;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Convert data points to SVG path
    const pathData = data
        .map((point, i) => {
            const x = padding + point.recall * chartWidth;
            const y = height - padding - point.precision * chartHeight;
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');

    // Fill path
    const fillPath = `${pathData} L ${width - padding} ${height - padding} L ${padding} ${height - padding} Z`;

    // Find operating point (high precision point)
    const operatingPoint = data.find(p => p.recall >= 0.80 && p.precision >= 0.80) || data[6];

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
                </svg>
                {title}
            </h3>
            <p className="text-xs text-slate-500 mb-4">{modelName} • AP = {averagePrecision.toFixed(2)}</p>

            <div className="flex justify-center">
                <svg width={width} height={height} className="overflow-visible">
                    {/* Grid lines */}
                    {[0.25, 0.5, 0.75].map((val) => (
                        <g key={val}>
                            <line
                                x1={padding}
                                y1={height - padding - val * chartHeight}
                                x2={width - padding}
                                y2={height - padding - val * chartHeight}
                                stroke="rgba(255,255,255,0.1)"
                                strokeDasharray="4"
                            />
                            <line
                                x1={padding + val * chartWidth}
                                y1={padding}
                                x2={padding + val * chartWidth}
                                y2={height - padding}
                                stroke="rgba(255,255,255,0.1)"
                                strokeDasharray="4"
                            />
                        </g>
                    ))}

                    {/* Baseline reference (random classifier) */}
                    <line
                        x1={padding}
                        y1={height - padding - 0.42 * chartHeight}
                        x2={width - padding}
                        y2={height - padding - 0.42 * chartHeight}
                        stroke="rgba(255,255,255,0.2)"
                        strokeDasharray="6"
                        strokeWidth="1"
                    />

                    {/* Area under curve */}
                    <motion.path
                        d={fillPath}
                        fill={color}
                        fillOpacity={0.15}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: animated ? 1 : 0 }}
                    />

                    {/* PR curve */}
                    <motion.path
                        d={pathData}
                        fill="none"
                        stroke={color}
                        strokeWidth="3"
                        strokeLinecap="round"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: animated ? 1.5 : 0, ease: 'easeOut' }}
                    />

                    {/* Operating point */}
                    <motion.circle
                        cx={padding + operatingPoint.recall * chartWidth}
                        cy={height - padding - operatingPoint.precision * chartHeight}
                        r="8"
                        fill={color}
                        fillOpacity={0.3}
                        stroke={color}
                        strokeWidth="2"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: animated ? 1.2 : 0 }}
                    />
                    <motion.circle
                        cx={padding + operatingPoint.recall * chartWidth}
                        cy={height - padding - operatingPoint.precision * chartHeight}
                        r="4"
                        fill={color}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: animated ? 1.2 : 0 }}
                    />

                    {/* Axes */}
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

                    {/* Axis labels */}
                    <text x={width / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-xs">
                        Recall
                    </text>
                    <text x={12} y={height / 2} textAnchor="middle" transform={`rotate(-90, 12, ${height / 2})`} className="fill-slate-400 text-xs">
                        Precision
                    </text>

                    {/* Tick labels */}
                    {[0, 0.5, 1].map((val) => (
                        <g key={val}>
                            <text x={padding + val * chartWidth} y={height - padding + 15} textAnchor="middle" className="fill-slate-500 text-[10px]">
                                {val}
                            </text>
                            <text x={padding - 8} y={height - padding - val * chartHeight + 4} textAnchor="end" className="fill-slate-500 text-[10px]">
                                {val}
                            </text>
                        </g>
                    ))}
                </svg>
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 mt-4 text-xs">
                <span className="flex items-center gap-1.5">
                    <div className="w-3 h-0.5 rounded" style={{ backgroundColor: color }} />
                    <span className="text-slate-400">PR Curve</span>
                </span>
                <span className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded-full border-2" style={{ borderColor: color }} />
                    <span className="text-slate-400">Operating Point</span>
                </span>
            </div>

            {/* AP Badge */}
            <div className="mt-4 pt-3 border-t border-white/10 flex justify-center">
                <div className="px-4 py-2 rounded-lg bg-gradient-to-r from-blue-500/20 to-blue-600/20 border border-blue-500/30">
                    <span className="text-blue-400 font-bold text-lg">AP = {averagePrecision.toFixed(2)}</span>
                    <span className="text-slate-400 text-xs ml-2">High Precision</span>
                </div>
            </div>
        </div>
    );
}
