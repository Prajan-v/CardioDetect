'use client';

import { motion } from 'framer-motion';

interface FeatureImportance {
    feature: string;
    importance: number;
}

interface FeatureImportanceChartProps {
    features?: FeatureImportance[];
    title?: string;
    animated?: boolean;
}

// Default feature importance based on CardioDetect model
const defaultFeatures: FeatureImportance[] = [
    { feature: 'Age', importance: 0.28 },
    { feature: 'Systolic BP', importance: 0.18 },
    { feature: 'Total Cholesterol', importance: 0.12 },
    { feature: 'HDL Cholesterol', importance: 0.10 },
    { feature: 'Smoking Status', importance: 0.08 },
    { feature: 'Diabetes', importance: 0.07 },
    { feature: 'Sex', importance: 0.06 },
    { feature: 'BP Medication', importance: 0.05 },
    { feature: 'BMI', importance: 0.03 },
    { feature: 'Heart Rate', importance: 0.03 },
];

export default function FeatureImportanceChart({
    features = defaultFeatures,
    title = 'Feature Importance',
    animated = true,
}: FeatureImportanceChartProps) {
    const maxImportance = Math.max(...features.map(f => f.importance));

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-orange-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
                {title}
            </h3>

            <div className="space-y-3">
                {features.map((item, index) => {
                    const percentage = (item.importance / maxImportance) * 100;
                    const hue = 30 + (index * 15); // Orange to red gradient

                    return (
                        <motion.div
                            key={item.feature}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: animated ? index * 0.05 : 0 }}
                        >
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-sm text-slate-300">{item.feature}</span>
                                <span className="text-sm font-medium text-slate-400">
                                    {(item.importance * 100).toFixed(0)}%
                                </span>
                            </div>
                            <div className="h-6 bg-white/5 rounded-lg overflow-hidden">
                                <motion.div
                                    className="h-full rounded-lg"
                                    style={{
                                        background: `linear-gradient(90deg, hsl(${hue}, 80%, 50%), hsl(${hue + 20}, 80%, 40%))`,
                                    }}
                                    initial={{ width: 0 }}
                                    animate={{ width: `${percentage}%` }}
                                    transition={{
                                        duration: animated ? 0.8 : 0,
                                        delay: animated ? index * 0.05 : 0,
                                        ease: 'easeOut',
                                    }}
                                />
                            </div>
                        </motion.div>
                    );
                })}
            </div>

            <div className="mt-4 pt-3 border-t border-white/10 text-xs text-slate-500">
                Based on SHAP values from XGBoost ensemble model
            </div>
        </div>
    );
}
