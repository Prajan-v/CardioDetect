"""
Batch test UltraOCR on ALL synthetic medical reports
"""
import sys
sys.path.insert(0, '/Users/prajanv/CardioDetect/Milestone_2/pipeline')

from ultra_ocr import UltraOCR
from pathlib import Path

def batch_test():
    print("=" * 70)
    print("🔬 ULTRA OCR - BATCH TEST ON ALL SYNTHETIC REPORTS")
    print("=" * 70)
    
    ocr = UltraOCR(verbose=False)
    
    report_dir = Path('/Users/prajanv/CardioDetect/Milestone_2/Medical_report/Synthetic_report')
    
    results = []
    
    for filepath in sorted(report_dir.glob("SYN-*.png")):
        result = ocr.extract_from_file(str(filepath))
        num_fields = result.get('num_fields', 0)
        conf = result.get('confidence', 0) * 100
        results.append((filepath.name, num_fields, conf))
        
    # Print summary table
    print(f"\n{'Report':<25} {'Fields':<10} {'Confidence':<12} {'Status'}")
    print("-" * 60)
    
    total_fields = 0
    total_conf = 0
    perfect = 0
    
    for name, fields, conf in results:
        status = "✅ PERFECT" if fields >= 10 else "⚠️ PARTIAL" if fields >= 5 else "❌ LOW"
        if fields >= 10:
            perfect += 1
        print(f"{name:<25} {fields:<10} {conf:.1f}%{'':<6} {status}")
        total_fields += fields
        total_conf += conf
    
    avg_fields = total_fields / len(results) if results else 0
    avg_conf = total_conf / len(results) if results else 0
    
    print("-" * 60)
    print(f"{'AVERAGE':<25} {avg_fields:.1f}{'':<8} {avg_conf:.1f}%")
    print(f"{'PERFECT EXTRACTION':<25} {perfect}/{len(results)} ({perfect/len(results)*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    batch_test()
