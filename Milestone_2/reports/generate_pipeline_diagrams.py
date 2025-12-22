"""
Generate professional sklearn-style pipeline diagrams for CardioDetect
Creates publication-quality diagrams for the Milestone 2 report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path('/Users/prajanv/CardioDetect/Milestone_2/reports')

# Color scheme
COLORS = {
    'primary': '#2980b9',      # Blue
    'secondary': '#27ae60',    # Green  
    'accent': '#8e44ad',       # Purple
    'warning': '#e67e22',      # Orange
    'dark': '#2c3e50',         # Dark blue-gray
    'light': '#ecf0f1',        # Light gray
    'transformer': '#3498db',  # Light blue
    'model': '#e74c3c',        # Red
    'ensemble': '#f39c12',     # Yellow-orange
}

def draw_rounded_box(ax, x, y, width, height, color, text, fontsize=10, text_color='white', alpha=0.9):
    """Draw a rounded rectangle with centered text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='white', linewidth=2, alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, 
            color=text_color, fontweight='bold', wrap=True)
    return box

def draw_arrow(ax, start, end, color='#34495e'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=15,
        lw=2,
        color=color
    )
    ax.add_patch(arrow)

def create_detection_pipeline_diagram():
    """Create Detection Pipeline diagram (sklearn-style)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(7, 9.5, 'Detection Pipeline Architecture', ha='center', va='center',
            fontsize=18, fontweight='bold', color=COLORS['dark'])
    ax.text(7, 9.0, 'Heart Disease Detection • Accuracy: 91.45%', ha='center', va='center',
            fontsize=12, color='#7f8c8d')
    
    # Main Pipeline box
    main_box = FancyBboxPatch(
        (0.5, 1.5), 13, 7, boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#f8f9fa', edgecolor=COLORS['primary'], linewidth=3, alpha=0.5
    )
    ax.add_patch(main_box)
    ax.text(7, 8.2, 'DetectionPipeline', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['primary'])
    
    # Input features box
    draw_rounded_box(ax, 2, 7, 2.5, 1.2, COLORS['dark'], 
                     'Input Features\n(13 UCI Features)', fontsize=9)
    
    # Preprocessing section
    preproc_box = FancyBboxPatch(
        (3.5, 4.5), 3.5, 3, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#e8f4f8', edgecolor=COLORS['transformer'], linewidth=2, alpha=0.8
    )
    ax.add_patch(preproc_box)
    ax.text(5.25, 7.2, 'Preprocessing', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['transformer'])
    
    # SimpleImputer
    draw_rounded_box(ax, 5.25, 6.3, 2.8, 0.7, COLORS['transformer'],
                     'SimpleImputer(median)', fontsize=9)
    
    # StandardScaler
    draw_rounded_box(ax, 5.25, 5.3, 2.8, 0.7, COLORS['transformer'],
                     'StandardScaler', fontsize=9)
    
    # Arrow from input to preprocessing
    draw_arrow(ax, (3.25, 7), (3.7, 7))
    # Arrow between imputer and scaler
    draw_arrow(ax, (5.25, 5.9), (5.25, 5.7))
    
    # Voting Ensemble section
    ensemble_box = FancyBboxPatch(
        (7.5, 2.5), 5.5, 5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#fef9e7', edgecolor=COLORS['ensemble'], linewidth=2, alpha=0.8
    )
    ax.add_patch(ensemble_box)
    ax.text(10.25, 7.2, 'VotingClassifier (soft)', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['ensemble'])
    
    # Individual models in ensemble
    models = [
        ('XGBoost', 6.3, '#e74c3c'),
        ('LightGBM', 5.4, '#9b59b6'),
        ('RandomForest', 4.5, '#27ae60'),
        ('ExtraTrees', 3.6, '#3498db'),
    ]
    
    for name, y_pos, color in models:
        draw_rounded_box(ax, 10.25, y_pos, 3.5, 0.7, color, name, fontsize=9)
    
    # Arrow from preprocessing to ensemble
    draw_arrow(ax, (7, 5.3), (7.7, 5.3))
    
    # Output
    draw_rounded_box(ax, 10.25, 2.2, 3.5, 0.8, COLORS['model'],
                     'Binary Classification\n(Disease/No Disease)', fontsize=9)
    
    # Arrow from ensemble to output
    draw_arrow(ax, (10.25, 3.2), (10.25, 2.7))
    
    # Feature list
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    feature_text = 'Features: ' + ', '.join(features[:7]) + '\n' + ', '.join(features[7:])
    ax.text(2, 1.0, feature_text, ha='left', va='center', fontsize=8, 
            color='#7f8c8d', style='italic')
    
    # Legend
    ax.text(11.5, 1.0, '✓ Accuracy: 91.45%', ha='left', fontsize=10, color=COLORS['secondary'])
    ax.text(11.5, 0.6, '✓ AUC-ROC: ~0.96', ha='left', fontsize=10, color=COLORS['secondary'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'detection_pipeline_diagram.png', dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: detection_pipeline_diagram.png")

def create_prediction_pipeline_diagram():
    """Create Prediction Pipeline diagram (sklearn-style)"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(7, 9.5, 'Prediction Pipeline Architecture', ha='center', va='center',
            fontsize=18, fontweight='bold', color=COLORS['dark'])
    ax.text(7, 9.0, '10-Year CHD Risk Prediction • Accuracy: 91.63%', ha='center', va='center',
            fontsize=12, color='#7f8c8d')
    
    # Main Pipeline box
    main_box = FancyBboxPatch(
        (0.5, 1.5), 13, 7, boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#f8f9fa', edgecolor=COLORS['secondary'], linewidth=3, alpha=0.5
    )
    ax.add_patch(main_box)
    ax.text(7, 8.2, 'PredictionPipeline', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['secondary'])
    
    # Input features box
    draw_rounded_box(ax, 2, 6.5, 2.5, 1.5, COLORS['dark'], 
                     'Input Features\n(10 Resting\nVitals)', fontsize=9)
    
    # Feature Engineering section
    fe_box = FancyBboxPatch(
        (3.5, 4), 3.8, 4, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#e8f8f5', edgecolor=COLORS['secondary'], linewidth=2, alpha=0.8
    )
    ax.add_patch(fe_box)
    ax.text(5.4, 7.7, 'FeatureEngineer', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['secondary'])
    
    # Feature engineering steps
    fe_steps = [
        ('Risk Flags', 6.8),
        ('Pulse Pressure', 6.0),
        ('Log Transforms', 5.2),
        ('Interactions', 4.4),
    ]
    for name, y_pos in fe_steps:
        draw_rounded_box(ax, 5.4, y_pos, 3, 0.6, COLORS['secondary'], name, fontsize=9)
    
    # Arrow from input to feature engineering
    draw_arrow(ax, (3.25, 6.5), (3.7, 6.5))
    
    # Preprocessing section
    preproc_box = FancyBboxPatch(
        (7.8, 5), 2.5, 2.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#e8f4f8', edgecolor=COLORS['transformer'], linewidth=2, alpha=0.8
    )
    ax.add_patch(preproc_box)
    ax.text(9.05, 7.2, 'Preprocessing', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['transformer'])
    
    draw_rounded_box(ax, 9.05, 6.3, 2, 0.6, COLORS['transformer'], 
                     'SimpleImputer', fontsize=9)
    draw_rounded_box(ax, 9.05, 5.5, 2, 0.6, COLORS['transformer'], 
                     'StandardScaler', fontsize=9)
    
    # Arrow from FE to preprocessing
    draw_arrow(ax, (7.3, 5.5), (7.95, 5.5))
    
    # Model section
    model_box = FancyBboxPatch(
        (10.8, 4.5), 2.5, 3.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#fdedec', edgecolor=COLORS['model'], linewidth=2, alpha=0.8
    )
    ax.add_patch(model_box)
    ax.text(12.05, 7.7, 'Classifier', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['model'])
    
    draw_rounded_box(ax, 12.05, 6.5, 2, 1.2, COLORS['model'], 
                     'XGBClassifier\n(Optimized)', fontsize=9)
    draw_rounded_box(ax, 12.05, 5.0, 2, 0.8, '#8e44ad', 
                     'Threshold\nTuning', fontsize=8)
    
    # Arrow from preprocessing to model
    draw_arrow(ax, (10.05, 5.9), (10.95, 5.9))
    
    # Risk Categorization
    risk_box = FancyBboxPatch(
        (3.5, 1.8), 9.5, 1.8, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#fef9e7', edgecolor=COLORS['warning'], linewidth=2, alpha=0.8
    )
    ax.add_patch(risk_box)
    ax.text(8.25, 3.4, 'Risk Categorization', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['warning'])
    
    # Risk levels
    draw_rounded_box(ax, 5, 2.4, 2, 0.7, '#27ae60', 'LOW\n<10%', fontsize=8)
    draw_rounded_box(ax, 8.25, 2.4, 2.2, 0.7, '#f39c12', 'MODERATE\n10-20%', fontsize=8)
    draw_rounded_box(ax, 11.5, 2.4, 2, 0.7, '#e74c3c', 'HIGH\n>20%', fontsize=8)
    
    # Arrow from model to risk
    draw_arrow(ax, (12.05, 4.5), (12.05, 3.7))
    
    # Legend
    ax.text(1, 1.0, '✓ Accuracy: 91.63%  ✓ AUC-ROC: 0.9567  ✓ F1-Score: 88.34%', 
            ha='left', fontsize=9, color=COLORS['secondary'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prediction_pipeline_diagram.png', dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: prediction_pipeline_diagram.png")

def create_integrated_pipeline_diagram():
    """Create Integrated Pipeline diagram showing full OCR → Models flow"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(8, 11.5, 'Integrated Dual-Model Pipeline', ha='center', va='center',
            fontsize=20, fontweight='bold', color=COLORS['dark'])
    ax.text(8, 11.0, 'End-to-End: Document → OCR → Detection/Prediction → Risk Assessment', 
            ha='center', va='center', fontsize=12, color='#7f8c8d')
    
    # Main Pipeline box
    main_box = FancyBboxPatch(
        (0.5, 1), 15, 9.5, boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#f8f9fa', edgecolor=COLORS['primary'], linewidth=3, alpha=0.3
    )
    ax.add_patch(main_box)
    ax.text(8, 10.2, 'DualModelPipeline', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['primary'])
    
    # Stage 1: Document Input
    draw_rounded_box(ax, 2, 9, 2.5, 1.2, COLORS['dark'], 
                     'Document Input\n(PDF/Image)', fontsize=9)
    
    # Stage 2: OCR Module
    ocr_box = FancyBboxPatch(
        (0.8, 6), 4, 2.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#e8f4f8', edgecolor=COLORS['transformer'], linewidth=2, alpha=0.8
    )
    ax.add_patch(ocr_box)
    ax.text(2.8, 8.2, 'EnhancedMedicalOCR', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['transformer'])
    
    draw_rounded_box(ax, 2.8, 7.3, 3.2, 0.5, COLORS['transformer'], 
                     'Image Preprocessing', fontsize=8)
    draw_rounded_box(ax, 2.8, 6.6, 3.2, 0.5, COLORS['transformer'], 
                     'Tesseract + EasyOCR', fontsize=8)
    
    # Arrow from input to OCR
    draw_arrow(ax, (2, 8.4), (2, 8.3))
    
    # Stage 3: Field Extraction
    extract_box = FancyBboxPatch(
        (0.8, 3.5), 4, 2, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#fef9e7', edgecolor=COLORS['warning'], linewidth=2, alpha=0.8
    )
    ax.add_patch(extract_box)
    ax.text(2.8, 5.2, 'Field Extraction', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['warning'])
    
    draw_rounded_box(ax, 2.8, 4.4, 3.2, 0.5, COLORS['warning'], 
                     'Regex Pattern Matching', fontsize=8)
    draw_rounded_box(ax, 2.8, 3.8, 3.2, 0.5, COLORS['warning'], 
                     'Value Validation', fontsize=8)
    
    # Arrow from OCR to extraction
    draw_arrow(ax, (2.8, 6.2), (2.8, 5.5))
    
    # Stage 4: Detection Model (LEFT branch)
    det_box = FancyBboxPatch(
        (5.5, 5.5), 4.5, 3.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#fdedec', edgecolor=COLORS['model'], linewidth=2, alpha=0.8
    )
    ax.add_patch(det_box)
    ax.text(7.75, 8.7, 'Detection Model', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['model'])
    ax.text(7.75, 8.3, '(91.45% Accuracy)', ha='center', fontsize=9, 
            color='#7f8c8d')
    
    draw_rounded_box(ax, 7.75, 7.4, 3.8, 0.6, COLORS['model'], 
                     'UCI Features Check', fontsize=8)
    draw_rounded_box(ax, 7.75, 6.6, 3.8, 0.6, '#e74c3c', 
                     '4-Model Voting Ensemble', fontsize=8)
    draw_rounded_box(ax, 7.75, 5.8, 3.8, 0.6, '#c0392b', 
                     'Disease Status Output', fontsize=8)
    
    # Stage 5: Prediction Model (RIGHT branch)
    pred_box = FancyBboxPatch(
        (10.5, 5.5), 4.5, 3.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#e8f8f5', edgecolor=COLORS['secondary'], linewidth=2, alpha=0.8
    )
    ax.add_patch(pred_box)
    ax.text(12.75, 8.7, 'Prediction Model', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['secondary'])
    ax.text(12.75, 8.3, '(91.63% Accuracy)', ha='center', fontsize=9, 
            color='#7f8c8d')
    
    draw_rounded_box(ax, 12.75, 7.4, 3.8, 0.6, COLORS['secondary'], 
                     'Feature Engineering', fontsize=8)
    draw_rounded_box(ax, 12.75, 6.6, 3.8, 0.6, '#27ae60', 
                     'XGBoost Classifier', fontsize=8)
    draw_rounded_box(ax, 12.75, 5.8, 3.8, 0.6, '#1e8449', 
                     '10-Year Risk Output', fontsize=8)
    
    # Arrows to both models
    draw_arrow(ax, (4.8, 4.5), (5.7, 5.5))
    draw_arrow(ax, (4.8, 4.5), (10.7, 5.5))
    
    # Stage 6: Clinical Assessment
    clinical_box = FancyBboxPatch(
        (5.5, 2.5), 4.5, 2.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#f5eef8', edgecolor='#8e44ad', linewidth=2, alpha=0.8
    )
    ax.add_patch(clinical_box)
    ax.text(7.75, 4.7, 'Clinical Assessment', ha='center', fontsize=10, 
            fontweight='bold', color='#8e44ad')
    
    draw_rounded_box(ax, 7.75, 3.8, 3.8, 0.5, '#8e44ad', 
                     'Guideline-Based Scoring', fontsize=8)
    draw_rounded_box(ax, 7.75, 3.1, 3.8, 0.5, '#9b59b6', 
                     'Risk Factor Analysis', fontsize=8)
    
    # Stage 7: Final Output
    output_box = FancyBboxPatch(
        (10.5, 2.5), 4.5, 2.5, boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor='#eafaf1', edgecolor=COLORS['secondary'], linewidth=2, alpha=0.8
    )
    ax.add_patch(output_box)
    ax.text(12.75, 4.7, 'Risk Assessment', ha='center', fontsize=10, 
            fontweight='bold', color=COLORS['secondary'])
    
    draw_rounded_box(ax, 12.75, 3.8, 3.8, 0.5, '#27ae60', 
                     'Risk Categorization', fontsize=8)
    draw_rounded_box(ax, 12.75, 3.1, 3.8, 0.5, '#2ecc71', 
                     'Recommendations', fontsize=8)
    
    # Arrows from models to outputs
    draw_arrow(ax, (7.75, 5.5), (7.75, 5))
    draw_arrow(ax, (12.75, 5.5), (12.75, 5))
    draw_arrow(ax, (10, 3.5), (10.5, 3.5))
    
    # Final output
    draw_rounded_box(ax, 12.75, 1.8, 4, 0.8, COLORS['dark'], 
                     'Comprehensive Risk Report', fontsize=9)
    draw_arrow(ax, (12.75, 2.5), (12.75, 2.2))
    
    # Usage modes legend
    ax.text(1, 1.5, 'Usage Modes:', ha='left', fontsize=10, fontweight='bold', color=COLORS['dark'])
    ax.text(1, 1.0, '• Detection Only: Uses UCI stress-test features → Disease status', 
            ha='left', fontsize=9, color='#7f8c8d')
    ax.text(1, 0.6, '• Prediction Only: Uses resting vitals → 10-year CHD risk', 
            ha='left', fontsize=9, color='#7f8c8d')
    ax.text(1, 0.2, '• Integrated: OCR extracts data → Runs both models → Combined assessment', 
            ha='left', fontsize=9, color='#7f8c8d')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'integrated_pipeline_diagram.png', dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: integrated_pipeline_diagram.png")

def create_model_comparison_diagram():
    """Create model comparison visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Model Comparison: Detection vs Prediction', ha='center', 
            fontsize=16, fontweight='bold', color=COLORS['dark'])
    
    # Detection Model Column
    det_header = FancyBboxPatch(
        (0.5, 5.5), 5, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['model'], edgecolor='white', linewidth=2
    )
    ax.add_patch(det_header)
    ax.text(3, 6.2, 'Detection Model', ha='center', fontsize=14, 
            fontweight='bold', color='white')
    ax.text(3, 5.8, '91.45% Accuracy', ha='center', fontsize=11, color='white')
    
    det_metrics = [
        ('Dataset', 'UCI Heart Disease (303 samples)'),
        ('Features', '13 clinical stress-test features'),
        ('Algorithm', '4-Model Voting Ensemble'),
        ('Output', 'Binary: Disease/No Disease'),
        ('Precision', '~90%'),
        ('Recall', '~93%'),
        ('AUC-ROC', '~0.96'),
    ]
    
    for i, (metric, value) in enumerate(det_metrics):
        y = 5.0 - i * 0.6
        ax.text(0.7, y, f'{metric}:', ha='left', fontsize=9, fontweight='bold', color=COLORS['dark'])
        ax.text(2.5, y, value, ha='left', fontsize=9, color='#555')
    
    # Prediction Model Column
    pred_header = FancyBboxPatch(
        (6.5, 5.5), 5, 1.3, boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['secondary'], edgecolor='white', linewidth=2
    )
    ax.add_patch(pred_header)
    ax.text(9, 6.2, 'Prediction Model', ha='center', fontsize=14, 
            fontweight='bold', color='white')
    ax.text(9, 5.8, '91.63% Accuracy', ha='center', fontsize=11, color='white')
    
    pred_metrics = [
        ('Dataset', 'Framingham 5k (5000+ samples)'),
        ('Features', '8 engineered from 10 vitals'),
        ('Algorithm', 'Optimized XGBoost'),
        ('Output', '3-tier: Low/Moderate/High'),
        ('Precision', '89.47%'),
        ('Recall', '87.23%'),
        ('AUC-ROC', '0.9567'),
    ]
    
    for i, (metric, value) in enumerate(pred_metrics):
        y = 5.0 - i * 0.6
        ax.text(6.7, y, f'{metric}:', ha='left', fontsize=9, fontweight='bold', color=COLORS['dark'])
        ax.text(8.5, y, value, ha='left', fontsize=9, color='#555')
    
    # Divider line
    ax.axvline(x=6, ymin=0.1, ymax=0.85, color='#bdc3c7', linewidth=2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison_diagram.png', dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: model_comparison_diagram.png")

def create_data_flow_diagram():
    """Create data flow visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(7, 5.5, 'Data Processing Flow', ha='center', fontsize=16, 
            fontweight='bold', color=COLORS['dark'])
    
    # Flow boxes
    steps = [
        (1.5, 'Raw\nDocument', COLORS['dark']),
        (4, 'OCR\nExtraction', COLORS['transformer']),
        (6.5, 'Feature\nEngineering', COLORS['warning']),
        (9, 'Model\nInference', COLORS['model']),
        (11.5, 'Risk\nAssessment', COLORS['secondary']),
    ]
    
    for x, text, color in steps:
        draw_rounded_box(ax, x, 3, 2, 1.5, color, text, fontsize=10)
    
    # Arrows
    arrow_positions = [(2.5, 3), (5, 3), (7.5, 3), (10, 3)]
    for start_x, y in arrow_positions:
        draw_arrow(ax, (start_x, y), (start_x + 0.5, y))
    
    # Bottom annotations
    annotations = [
        (1.5, 'PDF/Image\nInput'),
        (4, 'Tesseract\nEasyOCR'),
        (6.5, 'Risk Flags\nLog Transforms'),
        (9, 'XGBoost\nEnsemble'),
        (11.5, 'Low/Med/High\nRecommendations'),
    ]
    
    for x, text in annotations:
        ax.text(x, 1.5, text, ha='center', fontsize=8, color='#7f8c8d', style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_flow_diagram.png', dpi=150, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Created: data_flow_diagram.png")

if __name__ == "__main__":
    print("🎨 Generating Pipeline Diagrams...")
    print("=" * 50)
    create_detection_pipeline_diagram()
    create_prediction_pipeline_diagram()
    create_integrated_pipeline_diagram()
    create_model_comparison_diagram()
    create_data_flow_diagram()
    print("=" * 50)
    print("✅ All diagrams generated successfully!")
