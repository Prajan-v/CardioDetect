'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Activity, BarChart3, Brain, Heart, Target, TrendingUp, Zap, ArrowLeft } from 'lucide-react';
import FloatingParticles from '@/components/FloatingParticles';
import AnimatedHeart from '@/components/AnimatedHeart';
import ModelPerformanceCard from '@/components/ModelPerformanceCard';
import ConfusionMatrix from '@/components/ConfusionMatrix';
import FeatureImportanceChart from '@/components/FeatureImportanceChart';
import ROCCurve from '@/components/ROCCurve';
import PrecisionRecallCurve from '@/components/PrecisionRecallCurve';
import LearningCurves from '@/components/LearningCurves';
import CalibrationCurve from '@/components/CalibrationCurve';

export default function AnalyticsPage() {
    return (
        <div className="min-h-screen bg-[#0a0a1a] relative">
            <div className="fixed inset-0 mesh-bg pointer-events-none" />
            <div className="gradient-orb orb-1" />
            <div className="gradient-orb orb-2" />
            <FloatingParticles />

            {/* Navigation */}
            <nav className="relative z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <Link href="/" className="flex items-center gap-3">
                            <div className="w-8 h-8"><AnimatedHeart className="w-full h-full" /></div>
                            <span className="text-xl font-bold gradient-text">CardioDetect</span>
                        </Link>

                        <div className="flex items-center gap-4">
                            <Link
                                href="/dashboard"
                                className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                            >
                                <ArrowLeft className="w-4 h-4" />
                                Back to Dashboard
                            </Link>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="relative z-10 max-w-7xl mx-auto px-6 py-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-8"
                >
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center">
                            <BarChart3 className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <h1 className="text-3xl font-bold text-white">Model Analytics</h1>
                            <p className="text-slate-400">Performance metrics and model insights</p>
                        </div>
                    </div>
                </motion.div>

                {/* Performance Cards */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
                >
                    <ModelPerformanceCard
                        title="Detection Accuracy"
                        value={91.30}
                        icon={<Zap className="w-6 h-6 text-green-400" />}
                        color="bg-green-500/20"
                        description="Powered by a production-optimized voting ensemble model"
                    />
                    <ModelPerformanceCard
                        title="Prediction Accuracy"
                        value={91.63}
                        icon={<TrendingUp className="w-6 h-6 text-blue-400" />}
                        color="bg-blue-500/20"
                        description="XGBoost (16,123 samples)"
                    />
                    <ModelPerformanceCard
                        title="ROC-AUC Score"
                        value={0.98}
                        suffix=""
                        icon={<Target className="w-6 h-6 text-purple-400" />}
                        color="bg-purple-500/20"
                        description="XGBoost Ensemble"
                    />
                    <ModelPerformanceCard
                        title="R² Score"
                        value={0.99}
                        suffix=""
                        icon={<Brain className="w-6 h-6 text-yellow-400" />}
                        color="bg-yellow-500/20"
                        description="Regression accuracy"
                    />
                </motion.div>

                {/* Model Comparison */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.15 }}
                    className="glass-card p-6 mb-8"
                >
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-red-400" />
                        Ensemble Model Comparison
                    </h3>

                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-white/10">
                                    <th className="text-left py-3 px-4 text-slate-400 font-medium">Model</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium">Accuracy</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium">Precision</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium">Recall</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium">F1 Score</th>
                                    <th className="text-center py-3 px-4 text-slate-400 font-medium">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {[
                                    { name: 'XGBoost', acc: 88.2, prec: 86.5, rec: 84.3, f1: 85.4, active: true },
                                    { name: 'LightGBM', acc: 87.5, prec: 85.8, rec: 83.9, f1: 84.8, active: true },
                                    { name: 'Random Forest', acc: 86.1, prec: 84.2, rec: 82.7, f1: 83.4, active: true },
                                    { name: 'Extra Trees', acc: 85.8, prec: 83.9, rec: 82.1, f1: 83.0, active: false },
                                    { name: 'Voting Ensemble', acc: 91.30, prec: 89.7, rec: 88.2, f1: 88.9, active: true, highlight: true },
                                ].map((model) => (
                                    <motion.tr
                                        key={model.name}
                                        className={`border-b border-white/5 ${model.highlight ? 'bg-gradient-to-r from-green-500/10 to-transparent' : ''}`}
                                        whileHover={{ backgroundColor: 'rgba(255,255,255,0.05)' }}
                                    >
                                        <td className="py-3 px-4">
                                            <span className={`font-medium ${model.highlight ? 'text-green-400' : 'text-white'}`}>
                                                {model.name}
                                            </span>
                                        </td>
                                        <td className="text-center py-3 px-4 text-slate-300">{model.acc}%</td>
                                        <td className="text-center py-3 px-4 text-slate-300">{model.prec}%</td>
                                        <td className="text-center py-3 px-4 text-slate-300">{model.rec}%</td>
                                        <td className="text-center py-3 px-4 text-slate-300">{model.f1}%</td>
                                        <td className="text-center py-3 px-4">
                                            <span className={`px-2 py-1 rounded-full text-xs ${model.active ? 'bg-green-500/20 text-green-400' : 'bg-slate-500/20 text-slate-400'
                                                }`}>
                                                {model.active ? 'Active' : 'Inactive'}
                                            </span>
                                        </td>
                                    </motion.tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>

                {/* Confusion Matrices - Both Models */}
                <div className="grid lg:grid-cols-2 gap-8 mb-8">
                    {/* Detection Model Confusion Matrix */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        <div className="glass-card p-6">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <Zap className="w-5 h-5 text-green-400" />
                                Detection Model - Voting Ensemble (91.30%)
                            </h3>
                            <p className="text-xs text-slate-500 mb-3">Kaggle Heart Disease • 918 samples • 21 features</p>
                            <ConfusionMatrix
                                truePositive={460}
                                falsePositive={40}
                                trueNegative={378}
                                falseNegative={40}
                                labels={{ positive: 'Disease', negative: 'Healthy' }}
                            />
                        </div>
                    </motion.div>

                    {/* Prediction Model Confusion Matrix */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        <div className="glass-card p-6">
                            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-blue-400" />
                                Prediction Model - XGBoost Regressor (91.63%)
                            </h3>
                            <p className="text-xs text-slate-500 mb-3">Framingham + Kaggle • 16,123 samples • 34 features</p>
                            <ConfusionMatrix
                                truePositive={3215}
                                falsePositive={789}
                                trueNegative={11560}
                                falseNegative={559}
                                labels={{ positive: 'High Risk', negative: 'Low Risk' }}
                            />
                        </div>
                    </motion.div>
                </div>

                {/* Feature Importance - Both Models */}
                <div className="grid lg:grid-cols-2 gap-8 mb-8">
                    {/* Detection Model Features */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.35 }}
                    >
                        <div className="glass-card p-6">
                            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                                <Zap className="w-5 h-5 text-green-400" />
                                Detection Model Features
                            </h3>
                            <p className="text-xs text-slate-500 mb-4">Clinical indicators • Stress test data • ECG analysis</p>
                            <FeatureImportanceChart
                                title=""
                                features={[
                                    { feature: 'ST Slope (ECG)', importance: 0.24 },
                                    { feature: 'Chest Pain Type', importance: 0.18 },
                                    { feature: 'Max Heart Rate', importance: 0.15 },
                                    { feature: 'Exercise Angina', importance: 0.12 },
                                    { feature: 'Oldpeak (ST Depression)', importance: 0.10 },
                                    { feature: 'Age', importance: 0.08 },
                                    { feature: 'Cholesterol', importance: 0.05 },
                                    { feature: 'Resting ECG', importance: 0.04 },
                                    { feature: 'Resting BP', importance: 0.03 },
                                    { feature: 'Fasting Blood Sugar', importance: 0.01 },
                                ]}
                            />
                        </div>
                    </motion.div>

                    {/* Prediction Model Features */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.4 }}
                    >
                        <div className="glass-card p-6">
                            <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                                <TrendingUp className="w-5 h-5 text-blue-400" />
                                Prediction Model Features
                            </h3>
                            <p className="text-xs text-slate-500 mb-4">Framingham Risk Score • 10-year CVD risk factors</p>
                            <FeatureImportanceChart
                                title=""
                                features={[
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
                                ]}
                            />
                        </div>
                    </motion.div>
                </div>

                {/* Model Performance Curves */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.45 }}
                    className="mb-8"
                >
                    <div className="glass-card p-6 mb-4">
                        <h3 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                            <svg className="w-5 h-5 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M3 3v18h18" />
                                <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3" />
                            </svg>
                            Model Performance Curves
                        </h3>
                        <p className="text-sm text-slate-400">
                            Detailed performance analysis including ROC-AUC, Precision-Recall, Learning, and Calibration curves
                        </p>
                    </div>

                    {/* Detection Model Section */}
                    <div className="mb-6">
                        <div className="flex items-center gap-2 mb-4">
                            <Zap className="w-5 h-5 text-green-400" />
                            <h4 className="text-md font-semibold text-white">Detection Model - Voting Ensemble</h4>
                            <span className="text-xs text-slate-500">(918 samples • 21 features)</span>
                        </div>
                        <div className="grid lg:grid-cols-2 xl:grid-cols-4 gap-4">
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.5 }}>
                                <ROCCurve
                                    title="ROC Curve"
                                    modelName="Voting Ensemble"
                                    auc={0.9600}
                                    color="#22c55e"
                                />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.55 }}>
                                <PrecisionRecallCurve
                                    title="Precision-Recall"
                                    modelName="Voting Ensemble"
                                    averagePrecision={0.89}
                                    color="#22c55e"
                                />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.6 }}>
                                <LearningCurves
                                    title="Learning Curves"
                                    modelName="Voting Ensemble"
                                    epochs={10}
                                    metric="Accuracy"
                                    trainingData={[0.78, 0.84, 0.87, 0.89, 0.90, 0.905, 0.91, 0.912, 0.914, 0.916]}
                                    validationData={[0.76, 0.82, 0.85, 0.87, 0.885, 0.895, 0.902, 0.906, 0.909, 0.912]}
                                    techniques={['5-Fold CV', 'Soft Voting', 'Ensemble Avg']}
                                />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.65 }}>
                                <CalibrationCurve
                                    title="Calibration"
                                    modelName="Voting Ensemble"
                                    color="#22c55e"
                                />
                            </motion.div>
                        </div>
                    </div>

                    {/* Prediction Model Section */}
                    <div>
                        <div className="flex items-center gap-2 mb-4">
                            <TrendingUp className="w-5 h-5 text-blue-400" />
                            <h4 className="text-md font-semibold text-white">Prediction Model - XGBoost Regressor</h4>
                            <span className="text-xs text-slate-500">(16,123 samples • 34 features)</span>
                        </div>
                        <div className="grid lg:grid-cols-2 xl:grid-cols-4 gap-4">
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.7 }}>
                                <ROCCurve
                                    title="ROC Curve"
                                    modelName="XGBoost Regressor"
                                    auc={0.9800}
                                    color="#3b82f6"
                                    data={[
                                        { fpr: 0, tpr: 0 },
                                        { fpr: 0.01, tpr: 0.52 },
                                        { fpr: 0.03, tpr: 0.72 },
                                        { fpr: 0.06, tpr: 0.82 },
                                        { fpr: 0.10, tpr: 0.88 },
                                        { fpr: 0.15, tpr: 0.92 },
                                        { fpr: 0.22, tpr: 0.95 },
                                        { fpr: 0.32, tpr: 0.97 },
                                        { fpr: 0.48, tpr: 0.98 },
                                        { fpr: 0.68, tpr: 0.99 },
                                        { fpr: 1.0, tpr: 1.0 },
                                    ]}
                                />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.75 }}>
                                <PrecisionRecallCurve
                                    title="Precision-Recall"
                                    modelName="XGBoost Regressor"
                                    averagePrecision={0.92}
                                    color="#3b82f6"
                                    data={[
                                        { recall: 0, precision: 1.0 },
                                        { recall: 0.12, precision: 0.97 },
                                        { recall: 0.28, precision: 0.95 },
                                        { recall: 0.45, precision: 0.92 },
                                        { recall: 0.60, precision: 0.88 },
                                        { recall: 0.72, precision: 0.84 },
                                        { recall: 0.82, precision: 0.78 },
                                        { recall: 0.90, precision: 0.70 },
                                        { recall: 0.95, precision: 0.58 },
                                        { recall: 0.98, precision: 0.45 },
                                        { recall: 1.0, precision: 0.38 },
                                    ]}
                                />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.8 }}>
                                <LearningCurves
                                    title="Learning Curves"
                                    modelName="XGBoost Regressor"
                                    epochs={10}
                                    metric="R² Score"
                                    trainingData={[0.72, 0.80, 0.85, 0.88, 0.90, 0.91, 0.92, 0.925, 0.93, 0.935]}
                                    validationData={[0.70, 0.78, 0.83, 0.86, 0.88, 0.895, 0.91, 0.915, 0.92, 0.925]}
                                    techniques={['Early Stopping', 'L2 Reg', 'Tree Pruning']}
                                />
                            </motion.div>
                            <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.85 }}>
                                <CalibrationCurve
                                    title="Calibration"
                                    modelName="XGBoost Regressor"
                                    color="#3b82f6"
                                    data={[
                                        { predicted: 0.05, actual: 0.06 },
                                        { predicted: 0.15, actual: 0.14 },
                                        { predicted: 0.25, actual: 0.24 },
                                        { predicted: 0.35, actual: 0.36 },
                                        { predicted: 0.45, actual: 0.44 },
                                        { predicted: 0.55, actual: 0.56 },
                                        { predicted: 0.65, actual: 0.64 },
                                        { predicted: 0.75, actual: 0.76 },
                                        { predicted: 0.85, actual: 0.84 },
                                        { predicted: 0.95, actual: 0.94 },
                                    ]}
                                />
                            </motion.div>
                        </div>
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                    className="glass-card p-6"
                >
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <Heart className="w-5 h-5 text-red-400" />
                        Model Architecture
                    </h3>

                    <div className="grid md:grid-cols-3 gap-6 text-sm">
                        <div>
                            <h4 className="font-medium text-slate-300 mb-2">Detection Model</h4>
                            <ul className="space-y-1 text-slate-400">
                                <li>• Dataset: Kaggle Heart Disease (918 samples)</li>
                                <li>• Features: 21 (11 base + 10 engineered)</li>
                                <li>• Architecture: Powered by an optimized ensemble of tree-based machine learning models</li>
                                <li>• Validation: 5-fold Cross-Validation</li>
                                <li>• Accuracy: 91.30%</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-medium text-slate-300 mb-2">Prediction Model</h4>
                            <ul className="space-y-1 text-slate-400">
                                <li>• Dataset: Framingham + Kaggle (16,123 samples)</li>
                                <li>• Features: 34 (11 base + 23 derived)</li>
                                <li>• Architecture: XGBoost Regressor</li>
                                <li>• Accuracy: 91.63%</li>
                            </ul>
                        </div>
                        <div>
                            <h4 className="font-medium text-slate-300 mb-2">Explainability</h4>
                            <ul className="space-y-1 text-slate-400">
                                <li>• SHAP values for feature attribution</li>
                                <li>• Per-patient risk explanations</li>
                                <li>• Clinical guideline alignment</li>
                                <li>• Regulatory compliance ready</li>
                            </ul>
                        </div>
                    </div>
                    <div className="mt-6 pt-4 border-t border-white/10">
                        <h4 className="font-medium text-amber-400 mb-2">🔬 Future Improvements</h4>
                        <ul className="space-y-1 text-slate-400 text-sm">
                            <li>• <span className="text-amber-300">ECG Images</span> - Deep learning analysis of electrocardiogram signals</li>
                            <li>• <span className="text-amber-300">Medical Scans</span> - CT/MRI cardiac imaging for structural analysis</li>
                            <li>• <span className="text-amber-300">Wearable Data</span> - Real-time heart rate variability from smartwatches</li>
                        </ul>
                    </div>
                </motion.div>
            </main>
        </div>
    );
}
