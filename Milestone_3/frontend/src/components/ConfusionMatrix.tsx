'use client';

import { motion } from 'framer-motion';

interface ConfusionMatrixProps {
    truePositive: number;
    falsePositive: number;
    trueNegative: number;
    falseNegative: number;
    labels?: { positive: string; negative: string };
    animated?: boolean;
}

export default function ConfusionMatrix({
    truePositive,
    falsePositive,
    trueNegative,
    falseNegative,
    labels = { positive: 'Disease', negative: 'Healthy' },
    animated = true,
}: ConfusionMatrixProps) {
    const total = truePositive + falsePositive + trueNegative + falseNegative;

    // Calculate percentages
    const cells = [
        { value: trueNegative, label: 'TN', row: 0, col: 0, color: 'bg-green-500', intensity: trueNegative / total },
        { value: falsePositive, label: 'FP', row: 0, col: 1, color: 'bg-red-500', intensity: falsePositive / total },
        { value: falseNegative, label: 'FN', row: 1, col: 0, color: 'bg-red-500', intensity: falseNegative / total },
        { value: truePositive, label: 'TP', row: 1, col: 1, color: 'bg-green-500', intensity: truePositive / total },
    ];

    // Calculate metrics
    const accuracy = ((truePositive + trueNegative) / total * 100).toFixed(1);
    const precision = (truePositive / (truePositive + falsePositive) * 100).toFixed(1);
    const recall = (truePositive / (truePositive + falseNegative) * 100).toFixed(1);
    const f1 = (2 * parseFloat(precision) * parseFloat(recall) / (parseFloat(precision) + parseFloat(recall))).toFixed(1);

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="7" height="7" />
                    <rect x="14" y="3" width="7" height="7" />
                    <rect x="14" y="14" width="7" height="7" />
                    <rect x="3" y="14" width="7" height="7" />
                </svg>
                Confusion Matrix
            </h3>

            <div className="flex flex-col lg:flex-row gap-6">
                {/* Matrix Grid */}
                <div className="flex-1">
                    <div className="flex">
                        <div className="w-20" />
                        <div className="flex-1 text-center text-xs text-slate-400 mb-2">Predicted</div>
                    </div>

                    <div className="flex">
                        <div className="w-20 flex items-center justify-center">
                            <span className="text-xs text-slate-400 -rotate-90 whitespace-nowrap">Actual</span>
                        </div>

                        <div className="flex-1">
                            {/* Header Row */}
                            <div className="grid grid-cols-2 gap-1 mb-1">
                                <div className="text-center text-xs text-slate-400 py-1">{labels.negative}</div>
                                <div className="text-center text-xs text-slate-400 py-1">{labels.positive}</div>
                            </div>

                            {/* Matrix Cells */}
                            <div className="grid grid-cols-2 gap-1">
                                {cells.map((cell, index) => (
                                    <motion.div
                                        key={cell.label}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        transition={{ delay: animated ? index * 0.1 : 0 }}
                                        className={`relative aspect-square rounded-lg overflow-hidden group cursor-pointer`}
                                        style={{
                                            backgroundColor: cell.label === 'TP' || cell.label === 'TN'
                                                ? `rgba(34, 197, 94, ${0.2 + cell.intensity * 0.6})`
                                                : `rgba(239, 68, 68, ${0.2 + cell.intensity * 0.6})`
                                        }}
                                    >
                                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                                            <span className="text-2xl font-bold text-white">{cell.value}</span>
                                            <span className="text-xs text-slate-300">{cell.label}</span>
                                        </div>

                                        {/* Hover tooltip */}
                                        <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                            <div className="text-center px-2">
                                                <div className="text-sm font-medium text-white">
                                                    {cell.label === 'TP' && 'True Positive'}
                                                    {cell.label === 'TN' && 'True Negative'}
                                                    {cell.label === 'FP' && 'False Positive'}
                                                    {cell.label === 'FN' && 'False Negative'}
                                                </div>
                                                <div className="text-xs text-slate-400">
                                                    {((cell.value / total) * 100).toFixed(1)}% of total
                                                </div>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Metrics */}
                <div className="flex-1 space-y-3">
                    <h4 className="text-sm font-medium text-slate-400">Performance Metrics</h4>

                    {[
                        { label: 'Accuracy', value: accuracy, color: 'text-green-400' },
                        { label: 'Precision', value: precision, color: 'text-blue-400' },
                        { label: 'Recall', value: recall, color: 'text-yellow-400' },
                        { label: 'F1 Score', value: f1, color: 'text-purple-400' },
                    ].map((metric, index) => (
                        <motion.div
                            key={metric.label}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: animated ? 0.4 + index * 0.1 : 0 }}
                            className="flex items-center justify-between"
                        >
                            <span className="text-sm text-slate-300">{metric.label}</span>
                            <span className={`text-lg font-bold ${metric.color}`}>{metric.value}%</span>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Legend */}
            <div className="flex gap-4 mt-4 pt-3 border-t border-white/10 text-xs">
                <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-green-500/50" />
                    <span className="text-slate-400">Correct Predictions</span>
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-red-500/50" />
                    <span className="text-slate-400">Incorrect Predictions</span>
                </span>
            </div>
        </div>
    );
}
