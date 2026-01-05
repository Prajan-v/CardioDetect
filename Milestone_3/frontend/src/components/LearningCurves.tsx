'use client';

import { motion } from 'framer-motion';

interface LearningCurvesProps {
    title?: string;
    modelName?: string;
    trainingData?: number[];
    validationData?: number[];
    epochs?: number;
    metric?: string;
    animated?: boolean;
    techniques?: string[];
}

// Default learning curve data (accuracy over epochs)
const defaultTrainingData = [0.65, 0.78, 0.84, 0.87, 0.89, 0.90, 0.91, 0.915, 0.92, 0.925];
const defaultValidationData = [0.62, 0.75, 0.81, 0.84, 0.86, 0.87, 0.88, 0.885, 0.89, 0.895];
const defaultTechniques = ['5-Fold CV', 'Early Stopping', 'L2 Regularization'];

export default function LearningCurves({
    title = 'Learning Curves',
    modelName = 'XGBoost Ensemble',
    trainingData = defaultTrainingData,
    validationData = defaultValidationData,
    epochs = 10,
    metric = 'Accuracy',
    animated = true,
    techniques = defaultTechniques,
}: LearningCurvesProps) {
    const width = 280;
    const height = 220;
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    const minValue = Math.min(...trainingData, ...validationData) - 0.05;
    const maxValue = Math.max(...trainingData, ...validationData) + 0.02;
    const range = maxValue - minValue;

    const createPath = (data: number[]) => {
        return data
            .map((val, i) => {
                const x = padding + (i / (data.length - 1)) * chartWidth;
                const y = height - padding - ((val - minValue) / range) * chartHeight;
                return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
            })
            .join(' ');
    };

    const trainingPath = createPath(trainingData);
    const validationPath = createPath(validationData);

    // Calculate gap for overfitting indicator
    const finalGap = trainingData[trainingData.length - 1] - validationData[validationData.length - 1];
    const isOverfitting = finalGap > 0.05;
    const status = isOverfitting ? 'Slight Overfitting' : 'Good Fit';

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 3v18h18" />
                    <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
                </svg>
                {title}
            </h3>
            <p className="text-xs text-slate-500 mb-4">{modelName} • {epochs} epochs • {metric}</p>

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
                        </g>
                    ))}

                    {/* Epoch markers */}
                    {[2, 4, 6, 8].map((epoch) => (
                        <line
                            key={epoch}
                            x1={padding + (epoch / epochs) * chartWidth}
                            y1={padding}
                            x2={padding + (epoch / epochs) * chartWidth}
                            y2={height - padding}
                            stroke="rgba(255,255,255,0.05)"
                        />
                    ))}

                    {/* Validation curve (behind) */}
                    <motion.path
                        d={validationPath}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="3"
                        strokeLinecap="round"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: animated ? 1.5 : 0, ease: 'easeOut', delay: 0.2 }}
                    />

                    {/* Training curve (front) */}
                    <motion.path
                        d={trainingPath}
                        fill="none"
                        stroke="#a855f7"
                        strokeWidth="3"
                        strokeLinecap="round"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: animated ? 1.5 : 0, ease: 'easeOut' }}
                    />

                    {/* End points */}
                    <motion.circle
                        cx={width - padding}
                        cy={height - padding - ((trainingData[trainingData.length - 1] - minValue) / range) * chartHeight}
                        r="5"
                        fill="#a855f7"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: animated ? 1.3 : 0 }}
                    />
                    <motion.circle
                        cx={width - padding}
                        cy={height - padding - ((validationData[validationData.length - 1] - minValue) / range) * chartHeight}
                        r="5"
                        fill="#f59e0b"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: animated ? 1.5 : 0 }}
                    />

                    {/* Axes */}
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />

                    {/* Axis labels */}
                    <text x={width / 2} y={height - 8} textAnchor="middle" className="fill-slate-400 text-xs">
                        Epochs
                    </text>
                    <text x={12} y={height / 2} textAnchor="middle" transform={`rotate(-90, 12, ${height / 2})`} className="fill-slate-400 text-xs">
                        {metric}
                    </text>

                    {/* Tick labels */}
                    {[0, 5, 10].map((epoch) => (
                        <text
                            key={epoch}
                            x={padding + (epoch / epochs) * chartWidth}
                            y={height - padding + 15}
                            textAnchor="middle"
                            className="fill-slate-500 text-[10px]"
                        >
                            {epoch}
                        </text>
                    ))}
                    {[minValue, (minValue + maxValue) / 2, maxValue].map((val, i) => (
                        <text
                            key={i}
                            x={padding - 8}
                            y={height - padding - (i * 0.5) * chartHeight + 4}
                            textAnchor="end"
                            className="fill-slate-500 text-[10px]"
                        >
                            {val.toFixed(2)}
                        </text>
                    ))}
                </svg>
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 mt-4 text-xs">
                <span className="flex items-center gap-1.5">
                    <div className="w-3 h-0.5 rounded bg-purple-500" />
                    <span className="text-slate-400">Training ({(trainingData[trainingData.length - 1] * 100).toFixed(1)}%)</span>
                </span>
                <span className="flex items-center gap-1.5">
                    <div className="w-3 h-0.5 rounded bg-amber-500" />
                    <span className="text-slate-400">Validation ({(validationData[validationData.length - 1] * 100).toFixed(1)}%)</span>
                </span>
            </div>

            {/* Status Badge */}
            <div className="mt-4 pt-3 border-t border-white/10 flex flex-col items-center gap-2">
                <div className={`px-4 py-2 rounded-lg ${isOverfitting
                    ? 'bg-gradient-to-r from-amber-500/20 to-amber-600/20 border border-amber-500/30'
                    : 'bg-gradient-to-r from-green-500/20 to-green-600/20 border border-green-500/30'
                    }`}>
                    <span className={`font-bold text-sm ${isOverfitting ? 'text-amber-400' : 'text-green-400'}`}>
                        {status}
                    </span>
                    <span className="text-slate-400 text-xs ml-2">Gap: {(finalGap * 100).toFixed(1)}%</span>
                </div>

                {/* Regularization Techniques */}
                {!isOverfitting && techniques.length > 0 && (
                    <div className="flex flex-wrap justify-center gap-1.5 mt-1">
                        {techniques.map((tech, i) => (
                            <motion.span
                                key={tech}
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: animated ? 1.6 + i * 0.1 : 0 }}
                                className="px-2 py-0.5 rounded-full text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30"
                            >
                                {tech}
                            </motion.span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

