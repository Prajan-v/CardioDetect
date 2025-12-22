'use client';

import { motion } from 'framer-motion';

interface FactorData {
    name: string;
    value: number;
    maxValue: number;
    impact: 'positive' | 'negative' | 'neutral';
}

interface FactorChartProps {
    factors: FactorData[];
    title?: string;
    animated?: boolean;
}

export default function FactorChart({
    factors,
    title = 'Risk Factor Breakdown',
    animated = true,
}: FactorChartProps) {
    const getImpactColor = (impact: 'positive' | 'negative' | 'neutral') => {
        switch (impact) {
            case 'positive': return { bar: 'bg-green-500', text: 'text-green-400', bg: 'bg-green-500/20' };
            case 'negative': return { bar: 'bg-red-500', text: 'text-red-400', bg: 'bg-red-500/20' };
            default: return { bar: 'bg-slate-500', text: 'text-slate-400', bg: 'bg-slate-500/20' };
        }
    };

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 20V10M12 20V4M6 20v-6" />
                </svg>
                {title}
            </h3>

            <div className="space-y-4">
                {factors.map((factor, index) => {
                    const percentage = (factor.value / factor.maxValue) * 100;
                    const colors = getImpactColor(factor.impact);

                    return (
                        <div key={factor.name}>
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-sm text-slate-300">{factor.name}</span>
                                <span className={`text-sm font-medium ${colors.text}`}>
                                    {factor.value}{factor.maxValue > 1 ? '' : '%'}
                                    {factor.impact === 'negative' && ' ⚠'}
                                    {factor.impact === 'positive' && ' ✓'}
                                </span>
                            </div>
                            <div className={`h-3 rounded-full ${colors.bg} overflow-hidden`}>
                                <motion.div
                                    className={`h-full ${colors.bar} rounded-full`}
                                    initial={{ width: 0 }}
                                    animate={{ width: `${Math.min(100, percentage)}%` }}
                                    transition={{
                                        duration: animated ? 0.8 : 0,
                                        delay: animated ? index * 0.1 : 0,
                                        ease: 'easeOut',
                                    }}
                                />
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-white/10 text-xs">
                <span className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-green-500" />
                    <span className="text-slate-400">Healthy</span>
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-slate-400">Risk Factor</span>
                </span>
                <span className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-slate-500" />
                    <span className="text-slate-400">Neutral</span>
                </span>
            </div>
        </div>
    );
}

// Helper function to calculate factors from form data
export function calculateFactors(formData: Record<string, string>, mode: 'detection' | 'prediction'): FactorData[] {
    const factors: FactorData[] = [];

    const age = parseInt(formData.age) || 0;
    factors.push({
        name: 'Age',
        value: age,
        maxValue: 100,
        impact: age > 55 ? 'negative' : age > 45 ? 'neutral' : 'positive',
    });

    const bp = parseInt(formData.systolic_bp || formData.trestbps) || 0;
    if (bp > 0) {
        factors.push({
            name: 'Blood Pressure',
            value: bp,
            maxValue: 200,
            impact: bp > 140 ? 'negative' : bp > 120 ? 'neutral' : 'positive',
        });
    }

    const chol = parseInt(formData.total_cholesterol || formData.chol) || 0;
    if (chol > 0) {
        factors.push({
            name: 'Cholesterol',
            value: chol,
            maxValue: 400,
            impact: chol > 240 ? 'negative' : chol > 200 ? 'neutral' : 'positive',
        });
    }

    const hdl = parseInt(formData.hdl_cholesterol) || 0;
    if (hdl > 0) {
        factors.push({
            name: 'HDL (Good)',
            value: hdl,
            maxValue: 100,
            impact: hdl < 40 ? 'negative' : hdl > 60 ? 'positive' : 'neutral',
        });
    }

    if (formData.smoking === '1') {
        factors.push({
            name: 'Smoking',
            value: 100,
            maxValue: 100,
            impact: 'negative',
        });
    }

    if (formData.diabetes === '1') {
        factors.push({
            name: 'Diabetes',
            value: 100,
            maxValue: 100,
            impact: 'negative',
        });
    }

    const bmi = parseFloat(formData.bmi) || 0;
    if (bmi > 0) {
        factors.push({
            name: 'BMI',
            value: bmi,
            maxValue: 50,
            impact: bmi > 30 ? 'negative' : bmi > 25 ? 'neutral' : 'positive',
        });
    }

    return factors;
}
