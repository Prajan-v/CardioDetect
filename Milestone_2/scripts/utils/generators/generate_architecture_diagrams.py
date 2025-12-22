
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

REPORTS_DIR = Path('/Users/prajanv/CardioDetect/Milestone_2/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def draw_box(ax, xy, width, height, text, color='#E3F2FD', fontsize=10):
    rect = patches.Rectangle(xy, width, height, linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    cx = xy[0] + width/2
    cy = xy[1] + height/2
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, wrap=True)
    return (cx, cy, width, height)

def draw_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))

def generate_detection_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Nodes
    # Input
    draw_box(ax, (0.5, 1.5), 1.5, 1.0, "Input Data\n(13 UCI Features)", color='#BBDEFB')
    # Preprocessing
    draw_box(ax, (2.5, 1.5), 2.0, 1.0, "SimpleImputer\nStandardScaler", color='#E1F5FE')
    
    # Ensemble Box
    rect = patches.Rectangle((5.0, 0.5), 2.5, 3.0, linewidth=1, edgecolor='#0D47A1', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(6.25, 3.7, "Voting Ensemble", ha='center', fontsize=9, color='#0D47A1')
    
    draw_box(ax, (5.2, 2.8), 0.9, 0.5, "XGBoost", color='#C8E6C9', fontsize=8)
    draw_box(ax, (6.4, 2.8), 0.9, 0.5, "LightGBM", color='#C8E6C9', fontsize=8)
    draw_box(ax, (5.2, 1.8), 0.9, 0.5, "RandomForest", color='#C8E6C9', fontsize=8)
    draw_box(ax, (6.4, 1.8), 0.9, 0.5, "ExtraTrees", color='#C8E6C9', fontsize=8)
    
    draw_box(ax, (5.8, 0.7), 1.0, 0.8, "Soft Voting\nAggregator", color='#FFF9C4', fontsize=9)
    
    # Output
    draw_box(ax, (8.0, 1.5), 1.5, 1.0, "Detection Result\nAccuracy: 91.80%", color='#FFCDD2')
    
    # Arrows
    draw_arrow(ax, (2.0, 2.0), (2.5, 2.0)) # Input -> Pre
    draw_arrow(ax, (4.5, 2.0), (5.0, 2.0)) # Pre -> Ensemble
    
    draw_arrow(ax, (7.5, 2.0), (8.0, 2.0)) # Ensemble -> Output
    
    plt.tight_layout()
    output_path = REPORTS_DIR / 'detection_pipeline_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Detection Diagram: {output_path}")

def generate_prediction_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input
    draw_box(ax, (3.0, 6.5), 2.0, 1.0, "Input Vitals\n(10 Resting Metrics)", color='#E1BEE7')
    
    # Feature Eng
    rect = patches.Rectangle((1.0, 3.5), 6.0, 2.5, linewidth=1, edgecolor='#4A148C', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(4.0, 6.2, "Feature Engineering Layer", ha='center', fontsize=9, color='#4A148C')
    
    draw_box(ax, (1.2, 4.8), 1.5, 0.8, "Risk Flags", color='#D1C4E9', fontsize=8)
    draw_box(ax, (3.2, 4.8), 1.5, 0.8, "Log Transforms", color='#D1C4E9', fontsize=8)
    draw_box(ax, (5.2, 4.8), 1.5, 0.8, "Interactions", color='#D1C4E9', fontsize=8)
    draw_box(ax, (3.2, 3.7), 1.5, 0.8, "Risk Score", color='#D1C4E9', fontsize=8)
    
    # Scaler/Model
    draw_box(ax, (3.0, 2.2), 2.0, 0.8, "StandardScaler", color='#F8BBD0')
    draw_box(ax, (3.0, 1.0), 2.0, 0.8, "XGBoost (Optimized)\nAccuracy: 91.60%", color='#C5E1A5')
    
    # Arrows
    draw_arrow(ax, (4.0, 6.5), (4.0, 6.0))
    draw_arrow(ax, (4.0, 3.5), (4.0, 3.0))
    draw_arrow(ax, (4.0, 2.2), (4.0, 1.8))
    
    output_path = REPORTS_DIR / 'prediction_pipeline_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Prediction Diagram: {output_path}")

def generate_integrated_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    draw_box(ax, (0.5, 2.5), 1.5, 1.0, "Document\n(PDF/Image)", color='#FFECB3')
    
    # OCR
    draw_box(ax, (2.5, 2.5), 2.0, 1.0, "OCR Engine\n(Tesseract/EasyOCR)", color='#FFE0B2')
    
    # Router
    draw_box(ax, (5.0, 2.5), 1.5, 1.0, "Data Routing\n& Validation", color='#B2DFDB')
    
    # Models
    draw_box(ax, (7.0, 3.5), 2.0, 1.0, "Detection Model\n(91.80%)", color='#BBDEFB')
    draw_box(ax, (7.0, 1.5), 2.0, 1.0, "Prediction Model\n(91.60%)", color='#E1BEE7')
    
    # Output
    draw_box(ax, (9.2, 2.5), 0.8, 1.0, "Report", color='#C8E6C9')
    
    # Arrows
    draw_arrow(ax, (2.0, 3.0), (2.5, 3.0))
    draw_arrow(ax, (4.5, 3.0), (5.0, 3.0))
    draw_arrow(ax, (6.5, 3.0), (7.0, 4.0))
    draw_arrow(ax, (6.5, 3.0), (7.0, 2.0))
    draw_arrow(ax, (9.0, 4.0), (9.2, 3.0))
    draw_arrow(ax, (9.0, 2.0), (9.2, 3.0))
    
    output_path = REPORTS_DIR / 'integrated_pipeline_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Integrated Diagram: {output_path}")

def generate_model_comparison_diagram():
    fig, ax = plt.subplots(figsize=(10, 6)) # Slightly taller for better spacing
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, "Model Comparison & Selection Strategy", ha='center', fontsize=14, fontweight='bold', fontname='Arial')
    
    # --- Detection Card (Left) ---
    # Shadow/Border effect (Blue)
    # Note: Rectangle does not support boxstyle, use FancyBboxPatch
    rect_border = patches.FancyBboxPatch((1.0, 1.0), 3.8, 4.0, linewidth=2, edgecolor='#2196F3', facecolor='white', boxstyle='round,pad=0.1')
    ax.add_patch(rect_border)
    
    # Header Background (#e3f2fd)
    # Headers are usually rectangular, but let's keep it simple or use FancyBboxPatch for top rounded? 
    # Just standard Rectangle for header strip is fine inside the border.
    header_rect = patches.Rectangle((1.0, 4.2), 3.8, 0.8, linewidth=0, facecolor='#e3f2fd')
    ax.add_patch(header_rect)
    
    # Card Title
    ax.text(2.9, 4.6, "Detection Model", ha='center', va='center', fontsize=12, fontweight='bold', color='#0D47A1', fontname='Arial')
    
    # Line
    ax.plot([1.0, 4.8], [4.2, 4.2], color='#2196F3', linewidth=1)
    
    # Content
    x_left = 1.2
    y_start = 3.8
    gap = 0.6
    
    ax.text(x_left, y_start, "Accuracy: 91.80%", fontsize=11, fontweight='bold', color='black', fontname='Arial')
    ax.text(x_left, y_start - gap, "Input: Stress Test Data (13 features)", fontsize=10, fontname='Arial')
    ax.text(x_left, y_start - 2*gap, "Method: Voting Ensemble", fontsize=10, fontname='Arial')
    ax.text(x_left, y_start - 3*gap, "Use Case: Hospital/Cardiac Unit", fontsize=10, fontname='Arial')
    
    
    # --- Prediction Card (Right) ---
    # Shadow/Border effect (Purple)
    rect_border_r = patches.FancyBboxPatch((5.2, 1.0), 3.8, 4.0, linewidth=2, edgecolor='#9C27B0', facecolor='white', boxstyle='round,pad=0.1')
    ax.add_patch(rect_border_r)
    
    # Header Background (#f3e5f5)
    header_rect_r = patches.Rectangle((5.2, 4.2), 3.8, 0.8, linewidth=0, facecolor='#f3e5f5')
    ax.add_patch(header_rect_r)
    
    # Card Title
    ax.text(7.1, 4.5, "Prediction Model", ha='center', va='center', fontsize=12, fontweight='bold', color='#4A148C', fontname='Arial')
    
    # Line
    ax.plot([5.2, 9.0], [4.2, 4.2], color='#9C27B0', linewidth=1)
    
    # Content
    x_right = 5.4
    
    ax.text(x_right, y_start, "Accuracy: 91.60%", fontsize=11, fontweight='bold', color='black', fontname='Arial')
    ax.text(x_right, y_start - gap, "Input: Resting Vitals (10 features)", fontsize=10, fontname='Arial')
    ax.text(x_right, y_start - 2*gap, "Method: XGBoost (Optimized)", fontsize=10, fontname='Arial')
    ax.text(x_right, y_start - 3*gap, "Use Case: Routine Checkup/Clinic", fontsize=10, fontname='Arial')
    
    output_path = REPORTS_DIR / 'model_comparison_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated Comparison Diagram: {output_path}")

if __name__ == "__main__":
    generate_detection_pipeline_diagram()
    generate_prediction_pipeline_diagram()
    generate_integrated_pipeline_diagram()
    generate_model_comparison_diagram()
