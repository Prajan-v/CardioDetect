'use client';

import { motion } from 'framer-motion';

interface ROCCurveProps {
    title?: string;
    modelName?: string;
    auc?: number;
    data?: { fpr: number; tpr: number }[];
    color?: string;
    animated?: boolean;
}

// Default ROC curve data for Detection Model (Voting Ensemble)
const defaultData = [
    { fpr: 0, tpr: 0 },
    { fpr: 0.02, tpr: 0.45 },
    { fpr: 0.05, tpr: 0.68 },
    { fpr: 0.08, tpr: 0.78 },
    { fpr: 0.12, tpr: 0.85 },
    { fpr: 0.18, tpr: 0.90 },
    { fpr: 0.25, tpr: 0.93 },
    { fpr: 0.35, tpr: 0.95 },
    { fpr: 0.50, tpr: 0.97 },
    { fpr: 0.70, tpr: 0.98 },
    { fpr: 1.0, tpr: 1.0 },
];

export default function ROCCurve({
    title = 'ROC Curve',
    modelName = 'Voting Ensemble',
    auc = 0.96,
    data = defaultData,
    color = '#22c55e',
    animated = true,
}: ROCCurveProps) {
    const width = 280;
    const height = 220;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Convert data points to SVG path
    const pathData = data
        .map((point, i) => {
            const x = padding + point.fpr * chartWidth;
            const y = height - padding - point.tpr * chartHeight;
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');

    // Fill path (area under curve)
    const fillPath = `${pathData} L ${width - padding} ${height - padding} L ${padding} ${height - padding} Z`;

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <svg className="w-5 h-5 text-green-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
                {title}
            </h3>
            <p className="text-xs text-slate-500 mb-4">{modelName} • AUC = {auc.toFixed(4)}</p>

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

                    {/* Diagonal reference line (random classifier) */}
                    <line
                        x1={padding}
                        y1={height - padding}
                        x2={width - padding}
                        y2={padding}
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

                    {/* ROC curve */}
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

                    {/* Data points */}
                    {data.map((point, i) => (
                        <motion.circle
                            key={i}
                            cx={padding + point.fpr * chartWidth}
                            cy={height - padding - point.tpr * chartHeight}
                            r="4"
                            fill={color}
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: animated ? 0.1 * i : 0 }}
                            className="cursor-pointer hover:r-6"
                        />
                    ))}

                    {/* Axes */}
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

                    {/* Axis labels */}
                    <text x={width / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-xs">
                        False Positive Rate
                    </text>
                    <text x={12} y={height / 2} textAnchor="middle" transform={`rotate(-90, 12, ${height / 2})`} className="fill-slate-400 text-xs">
                        True Positive Rate
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
                    <span className="text-slate-400">ROC Curve</span>
                </span>
                <span className="flex items-center gap-1.5">
                    <div className="w-3 h-0.5 rounded bg-white/20" style={{ background: 'repeating-linear-gradient(90deg, rgba(255,255,255,0.3), rgba(255,255,255,0.3) 2px, transparent 2px, transparent 4px)' }} />
                    <span className="text-slate-400">Random (AUC=0.5)</span>
                </span>
            </div>

            {/* AUC Badge */}
            <div className="mt-4 pt-3 border-t border-white/10 flex justify-center">
                <div className="px-4 py-2 rounded-lg bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-500/30">
                    <span className="text-green-400 font-bold text-lg">AUC = {auc.toFixed(2)}</span>
                    <span className="text-slate-400 text-xs ml-2">Excellent</span>
                </div>
            </div>
        </div>
    );
}
