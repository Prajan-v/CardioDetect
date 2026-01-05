'use client';

import { motion } from 'framer-motion';

interface CalibrationCurveProps {
    title?: string;
    modelName?: string;
    data?: { predicted: number; actual: number }[];
    color?: string;
    animated?: boolean;
}

// Default calibration data for XGBoost model
const defaultData = [
    { predicted: 0.05, actual: 0.04 },
    { predicted: 0.15, actual: 0.13 },
    { predicted: 0.25, actual: 0.23 },
    { predicted: 0.35, actual: 0.38 },
    { predicted: 0.45, actual: 0.47 },
    { predicted: 0.55, actual: 0.54 },
    { predicted: 0.65, actual: 0.62 },
    { predicted: 0.75, actual: 0.73 },
    { predicted: 0.85, actual: 0.88 },
    { predicted: 0.95, actual: 0.96 },
];

export default function CalibrationCurve({
    title = 'Calibration Curve',
    modelName = 'XGBoost Regressor',
    data = defaultData,
    color = '#ec4899',
    animated = true,
}: CalibrationCurveProps) {
    const width = 280;
    const height = 220;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Convert data points to SVG path
    const pathData = data
        .map((point, i) => {
            const x = padding + point.predicted * chartWidth;
            const y = height - padding - point.actual * chartHeight;
            return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');

    // Calculate calibration error (mean absolute error from diagonal)
    const calibrationError = data.reduce((sum, point) => sum + Math.abs(point.predicted - point.actual), 0) / data.length;
    const isWellCalibrated = calibrationError < 0.05;

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <svg className="w-5 h-5 text-pink-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 6v6l4 2" />
                </svg>
                {title}
            </h3>
            <p className="text-xs text-slate-500 mb-4">{modelName} • Reliability Diagram</p>

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

                    {/* Perfect calibration diagonal */}
                    <line
                        x1={padding}
                        y1={height - padding}
                        x2={width - padding}
                        y2={padding}
                        stroke="rgba(34,197,94,0.5)"
                        strokeWidth="2"
                        strokeDasharray="6"
                    />

                    {/* Calibration curve */}
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
                        <motion.g key={i}>
                            {/* Error line from diagonal */}
                            <motion.line
                                x1={padding + point.predicted * chartWidth}
                                y1={height - padding - point.predicted * chartHeight}
                                x2={padding + point.predicted * chartWidth}
                                y2={height - padding - point.actual * chartHeight}
                                stroke={color}
                                strokeOpacity={0.3}
                                strokeWidth="1"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: animated ? 0.1 * i : 0 }}
                            />
                            <motion.circle
                                cx={padding + point.predicted * chartWidth}
                                cy={height - padding - point.actual * chartHeight}
                                r="5"
                                fill={color}
                                initial={{ opacity: 0, scale: 0 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: animated ? 0.1 * i : 0 }}
                            />
                        </motion.g>
                    ))}

                    {/* Axes */}
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

                    {/* Axis labels */}
                    <text x={width / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-xs">
                        Mean Predicted Probability
                    </text>
                    <text x={12} y={height / 2} textAnchor="middle" transform={`rotate(-90, 12, ${height / 2})`} className="fill-slate-400 text-xs">
                        Fraction of Positives
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
                    <span className="text-slate-400">Model Calibration</span>
                </span>
                <span className="flex items-center gap-1.5">
                    <div className="w-3 h-0.5 rounded bg-green-500" style={{ background: 'repeating-linear-gradient(90deg, rgba(34,197,94,0.5), rgba(34,197,94,0.5) 2px, transparent 2px, transparent 4px)' }} />
                    <span className="text-slate-400">Perfect</span>
                </span>
            </div>

            {/* Calibration Status Badge */}
            <div className="mt-4 pt-3 border-t border-white/10 flex justify-center">
                <div className={`px-4 py-2 rounded-lg ${isWellCalibrated
                    ? 'bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-500/30'
                    : 'bg-gradient-to-r from-amber-500/20 to-amber-600/20 border border-amber-500/30'
                    }`}>
                    <span className={`font-bold text-sm ${isWellCalibrated ? 'text-green-400' : 'text-amber-400'}`}>
                        {isWellCalibrated ? 'Well Calibrated' : 'Needs Calibration'}
                    </span>
                    <span className="text-slate-400 text-xs ml-2">ECE: {(calibrationError * 100).toFixed(2)}%</span>
                </div>
            </div>
        </div>
    );
}
