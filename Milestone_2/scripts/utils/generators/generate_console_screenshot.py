
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

REPORTS_DIR = Path('/Users/prajanv/CardioDetect/Milestone_2/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_console_screenshot():
    width = 800
    height = 400
    bg_color = (30, 30, 30) # Dark gray console
    text_color = (200, 200, 200) # Light gray
    green_color = (50, 200, 50) # Green for success
    
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # Try to load a monospace font, fallback to default
    try:
        # MacOS typical location
        font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 14)
    except:
        try:
             font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", 14)
        except:
             font = ImageFont.load_default()

    lines = [
        "user@CardioDetect:~/pipeline $ python integrated_pipeline.py --input data/SYN-001.png",
        "[INFO] Initializing CardioDetect v2.0 Integrated Pipeline...",
        "[INFO] Loaded extraction patterns for 10 fields.",
        "[INFO] OCR Engine: Tesseract + EasyOCR ready.",
        "----------------------------------------------------------------",
        "[STEP 1] Document Loaded: SYN-001.png (Type: IMAGE)",
        "[STEP 2] OCR Extraction Complete. Confidence: 0.94",
        "         -> Extracted: {'age': 60, 'sex': 1, 'cp': 1, 'trestbps': 140, 'chol': 220, ...}",
        "[STEP 3] Model Selection: Full Features Detected.",
        "         -> Loading Detection Model (Accuracy: 91.80%)... DONE",
        "         -> Loading Prediction Model (Accuracy: 91.60%)... DONE",
        "[STEP 4] Risk Assessment Running...",
        "         -> Detection Result: POSITIVE (Prob: 0.88)",
        "         -> Prediction Result: HIGH RISK (Prob: 0.24)",
        "----------------------------------------------------------------",
        "[SUCCESS] Report generated: reports/patient_report_SYN-001.pdf",
        "user@CardioDetect:~/pipeline $ _"
    ]
    
    y = 20
    x = 20
    line_height = 20
    
    for line in lines:
        col = text_color
        if "user@" in line:
            col = (100, 255, 100) # Green prompt
        if "[SUCCESS]" in line:
            col = green_color
        
        d.text((x, y), line, font=font, fill=col)
        y += line_height

    output_path = REPORTS_DIR / 'syn001_success.png'
    img.save(output_path)
    print(f"✅ Generated Console Screenshot: {output_path}")

def generate_results_screenshot():
    # User Prompt Logic:
    # CLI-style UI footer component
    # Background: Dark Grey/Black (#1e1e1e)
    # Text Color: Light Grey (#cccccc)
    # Font: Monospace
    # Content: CardioDetect v2.0 | Detection (91.80%) | Prediction (91.60%) ...
    # Styling: Green percentages
    
    width = 800
    height = 500
    bg_color = (255, 255, 255) # White background for main report
    
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # Fonts
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        # Monospace for footer
        mono_font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 13)
        mono_font_bold = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", 13) 
    except:
        font = ImageFont.load_default()
        title_font = font
        mono_font = font
        mono_font_bold = font

    # --- Main Report Content (Matches previous logic, just recreating consistent background) ---
    # Header
    d.rectangle([(0, 0), (width, 60)], fill=(0, 40, 80))
    d.text((20, 15), "CardioDetect v2.0 - Risk Assessment Report", font=title_font, fill=(255, 255, 255))
    
    y = 80
    lines = [
        "Patient ID: SYN-001 | Date: 2025-12-07",
        "--------------------------------------------------------------------------------",
        "EXTRACTED VITALS:",
        "• Age: 60  |  Sex: Male (1)",
        "• Blood Pressure: 140/90 mmHg",
        "• Cholesterol: 220 mg/dL",
        "--------------------------------------------------------------------------------",
        "MODEL PREDICTIONS:",
        "1. Detection Model (Ensemble):",
        "   -> Result: POSITIVE for Heart Disease",
        "   -> Probability: 88.4%",
        "",
        "2. Prediction Model (XGBoost):",
        "   -> 10-Year Risk: > 20% (High Risk)",
        "   -> Probability: 24.1%",
        "--------------------------------------------------------------------------------",
        "CLINICAL RECOMMENDATION:",
        "Immediate medical consultation recommended."
    ]
    
    for line in lines:
        d.text((20, y), line, font=font, fill=(0, 0, 0))
        y += 25
        
    # --- Footer Implementation (User Prompt) ---
    footer_height = 40
    footer_y = height - footer_height
    d.rectangle([(0, footer_y), (width, height)], fill=(30, 30, 30)) # #1e1e1e approx
    
    # Text segments
    # Content: CardioDetect v2.0 | Detection (91.80%) | Prediction (91.60%) | Generated via Integrated Pipeline
    
    x_pos = 15
    y_pos = footer_y + 12
    gray = (204, 204, 204) # #cccccc
    green = (50, 205, 50)  # LimeGreen-ish
    
    def draw_seg(text, color):
        nonlocal x_pos
        d.text((x_pos, y_pos), text, font=mono_font, fill=color)
        bbox = d.textbbox((x_pos, y_pos), text, font=mono_font)
        x_pos += (bbox[2] - bbox[0])
    
    draw_seg("CardioDetect v2.0 | Detection (", gray)
    draw_seg("91.80%", green)
    draw_seg(") | Prediction (", gray)
    draw_seg("91.60%", green)
    draw_seg(") | Generated via Integrated Pipeline", gray)
    
    output_path = REPORTS_DIR / 'results_screenshot_footer.png'
    img.save(output_path)
    print(f"✅ Generated Results Screenshot: {output_path}")

if __name__ == "__main__":
    generate_console_screenshot()
    generate_results_screenshot()
