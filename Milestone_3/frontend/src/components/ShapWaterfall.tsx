'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ShapValue {
    feature: string;
    contribution: number; // Positive = increases risk, Negative = decreases
    value: string | number;
}

// Backend SHAP explanation format
interface ShapExplanation {
    feature: string;
    value?: number;
    impact: number;
    direction: 'increases' | 'decreases';
}

interface ShapWaterfallProps {
    baseValue?: number;
    finalValue: number;
    shapValues?: ShapValue[];
    explanations?: ShapExplanation[]; // From backend SHAP
    formData?: Record<string, string>;
    animated?: boolean;
    initialShowCount?: number; // How many to show initially (default 7)
}

// Calculate SHAP-like contributions from form data
function calculateShapValues(formData: Record<string, string>, baseRisk: number, finalRisk: number): ShapValue[] {
    const contributions: ShapValue[] = [];
    const totalContribution = finalRisk - baseRisk;

    // Age contribution (strongest factor)
    const age = parseInt(formData.age) || 45;
    if (age > 55) {
        contributions.push({ feature: 'Age', contribution: Math.min(5, (age - 55) * 0.3), value: `${age} years` });
    } else if (age < 40) {
        contributions.push({ feature: 'Age', contribution: -2, value: `${age} years` });
    }

    // Blood Pressure
    const bp = parseInt(formData.systolic_bp || formData.trestbps) || 120;
    if (bp > 140) {
        contributions.push({ feature: 'Blood Pressure', contribution: Math.min(4, (bp - 140) * 0.1), value: `${bp} mmHg` });
    } else if (bp < 120) {
        contributions.push({ feature: 'Blood Pressure', contribution: -1, value: `${bp} mmHg` });
    }

    // Cholesterol
    const chol = parseInt(formData.total_cholesterol || formData.chol) || 200;
    if (chol > 240) {
        contributions.push({ feature: 'Cholesterol', contribution: Math.min(3, (chol - 240) * 0.03), value: `${chol} mg/dL` });
    } else if (chol < 200) {
        contributions.push({ feature: 'Cholesterol', contribution: -1.5, value: `${chol} mg/dL` });
    }

    // HDL (protective)
    const hdl = parseInt(formData.hdl_cholesterol) || 50;
    if (hdl > 60) {
        contributions.push({ feature: 'HDL (Good)', contribution: -2, value: `${hdl} mg/dL` });
    } else if (hdl < 40) {
        contributions.push({ feature: 'HDL (Good)', contribution: 2, value: `${hdl} mg/dL` });
    }

    // Smoking
    if (formData.smoking === '1') {
        contributions.push({ feature: 'Smoking', contribution: 3.5, value: 'Yes' });
    } else if (formData.smoking === '0') {
        contributions.push({ feature: 'Non-Smoker', contribution: -1, value: 'No' });
    }

    // Diabetes
    if (formData.diabetes === '1') {
        contributions.push({ feature: 'Diabetes', contribution: 2.5, value: 'Yes' });
    }

    // BMI
    const bmi = parseFloat(formData.bmi) || 25;
    if (bmi > 30) {
        contributions.push({ feature: 'BMI', contribution: Math.min(2, (bmi - 30) * 0.2), value: bmi.toFixed(1) });
    } else if (bmi < 25) {
        contributions.push({ feature: 'BMI', contribution: -0.5, value: bmi.toFixed(1) });
    }

    // Sort by absolute contribution
    return contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

export default function ShapWaterfall({
    baseValue = 9,
    finalValue,
    shapValues,
    explanations, // From backend SHAP
    formData,
    animated = true,
    initialShowCount = 7,
}: ShapWaterfallProps) {
    const [showAll, setShowAll] = useState(false);

    // Priority: backend explanations > shapValues > calculated from formData
    let values: ShapValue[] = [];

    if (explanations && explanations.length > 0) {
        // Convert backend SHAP format to internal format
        values = explanations.map(exp => ({
            feature: exp.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            contribution: exp.impact * 10, // Scale impact to percentage-like values
            value: exp.value?.toFixed(1) || exp.direction,
        }));
    } else if (shapValues && shapValues.length > 0) {
        values = shapValues;
    } else if (formData) {
        values = calculateShapValues(formData, baseValue, finalValue);
    }

    // Sort ALL values by absolute contribution (descending - highest impact first)
    const sortedValues = [...values].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

    // Split into visible and hidden based on showAll state
    const visibleValues = showAll ? sortedValues : sortedValues.slice(0, initialShowCount);
    const hiddenValues = sortedValues.slice(initialShowCount);
    const hasMoreFeatures = hiddenValues.length > 0;

    // Calculate running total for waterfall
    let runningTotal = baseValue;
    const bars = visibleValues.map((item, index) => {
        const start = runningTotal;
        runningTotal += item.contribution;
        return { ...item, start, end: runningTotal, index };
    });

    // If not showing all, add "Other factors" for hidden contributions
    if (!showAll && hasMoreFeatures) {
        const hiddenContribution = hiddenValues.reduce((sum, v) => sum + v.contribution, 0);
        if (Math.abs(hiddenContribution) > 0.1) {
            bars.push({
                feature: `+${hiddenValues.length} Other Factors`,
                contribution: hiddenContribution,
                value: `${hiddenValues.length} more`,
                start: runningTotal,
                end: runningTotal + hiddenContribution,
                index: bars.length,
            });
            runningTotal += hiddenContribution;
        }
    }

    const maxValue = Math.max(finalValue, baseValue, ...bars.map(b => Math.max(b.start, b.end))) + 5;
    const scale = 100 / maxValue;

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 3v18h18" />
                    <path d="M7 16l4-8 4 4 6-6" />
                </svg>
                Why This Prediction?
            </h3>

            <div className="space-y-3">
                {/* Base Value */}
                <div className="flex items-center gap-3">
                    <div className="w-28 text-sm text-slate-400">Base Risk</div>
                    <div className="flex-1 h-8 bg-white/5 rounded-lg relative overflow-hidden">
                        <motion.div
                            className="absolute h-full bg-slate-500 rounded-lg"
                            initial={{ width: 0 }}
                            animate={{ width: `${baseValue * scale}%` }}
                            transition={{ duration: animated ? 0.5 : 0, ease: 'easeOut' }}
                        />
                        <div className="absolute inset-0 flex items-center px-3">
                            <span className="text-sm font-medium text-white">{baseValue.toFixed(1)}%</span>
                        </div>
                    </div>
                </div>

                {/* Contributions with AnimatePresence */}
                <AnimatePresence mode="popLayout">
                    {bars.map((bar, index) => {
                        const isPositive = bar.contribution > 0;
                        const barStart = Math.min(bar.start, bar.end) * scale;
                        const barWidth = Math.abs(bar.contribution) * scale;

                        return (
                            <motion.div
                                key={bar.feature}
                                className="flex items-center gap-3"
                                initial={{ opacity: 0, x: -20, height: 0 }}
                                animate={{ opacity: 1, x: 0, height: 'auto' }}
                                exit={{ opacity: 0, x: -20, height: 0 }}
                                transition={{ delay: animated ? 0.1 + index * 0.05 : 0, duration: 0.2 }}
                            >
                                <div className="w-28 text-sm text-slate-300 truncate" title={bar.feature}>
                                    {bar.feature}
                                </div>
                                <div className="flex-1 h-8 bg-white/5 rounded-lg relative overflow-hidden">
                                    {/* Connection line */}
                                    <div
                                        className="absolute h-0.5 bg-slate-600 top-1/2 -translate-y-1/2"
                                        style={{ left: 0, width: `${barStart}%` }}
                                    />
                                    {/* Contribution bar */}
                                    <motion.div
                                        className={`absolute h-full rounded-lg ${isPositive ? 'bg-red-500/80' : 'bg-green-500/80'}`}
                                        style={{ left: `${barStart}%` }}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${barWidth}%` }}
                                        transition={{ duration: animated ? 0.5 : 0, delay: animated ? 0.3 + index * 0.1 : 0 }}
                                    />
                                </div>
                                <div className={`w-16 text-sm font-medium text-right ${isPositive ? 'text-red-400' : 'text-green-400'}`}>
                                    {isPositive ? '+' : ''}{bar.contribution.toFixed(1)}%
                                </div>
                            </motion.div>
                        );
                    })}

                    {/* Show More / Show Less Button */}
                    {hasMoreFeatures && (
                        <motion.button
                            layout
                            onClick={() => setShowAll(!showAll)}
                            className="w-full mt-2 py-2 px-4 text-sm text-blue-400 hover:text-blue-300 
                                   bg-white/5 hover:bg-white/10 rounded-lg transition-colors
                                   flex items-center justify-center gap-2"
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                        >
                            {showAll ? (
                                <>
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                                    </svg>
                                    Show Less
                                </>
                            ) : (
                                <>
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                    </svg>
                                    Show All {hiddenValues.length} Features
                                </>
                            )}
                        </motion.button>
                    )}
                </AnimatePresence>

                {/* Final Value */}
                <div className="flex items-center gap-3 pt-2 border-t border-white/10">
                    <div className="w-28 text-sm font-semibold text-white">Final Risk</div>
                    <div className="flex-1 h-8 bg-white/5 rounded-lg relative overflow-hidden">
                        <motion.div
                            className={`absolute h-full rounded-lg ${finalValue >= 25 ? 'bg-red-500' : finalValue >= 10 ? 'bg-yellow-500' : 'bg-green-500'
                                }`}
                            initial={{ width: 0 }}
                            animate={{ width: `${finalValue * scale}%` }}
                            transition={{ duration: animated ? 0.8 : 0, delay: animated ? 0.5 + bars.length * 0.1 : 0 }}
                        />
                        <div className="absolute inset-0 flex items-center px-3">
                            <span className="text-sm font-bold text-white">{finalValue.toFixed(1)}%</span>
                        </div>
                    </div>
                    <div className="w-16" />
                </div>
            </div>

            {/* Legend */}
            <div className="flex gap-4 mt-4 pt-3 border-t border-white/10 text-xs">
                <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-red-500" />
                    <span className="text-slate-400">Increases Risk</span>
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded bg-green-500" />
                    <span className="text-slate-400">Decreases Risk</span>
                </span>
            </div>
        </div>
    );
}
