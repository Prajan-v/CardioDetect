"""Test the robust OCR pipeline on lbmaske samples."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robust_ocr_pipeline import RobustOCRPipeline, extract_medical_fields


def test_single_file(file_path: Path) -> None:
    """Test a single file with detailed output."""
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    result = extract_medical_fields(file_path, verbose=True)

    print(f"\n--- RESULT ---")
    print(f"Success: {result.success}")
    print(f"Document Type: {result.document_type.value}")
    print(f"Engine Used: {result.engine_used.value}")
    print(f"Confidence: {result.overall_confidence:.1f}%")
    print(f"Quality: {result.quality_score}")
    print(f"Fallback Used: {result.fallback_used}")

    print(f"\n--- RAW TEXT (first 500 chars) ---")
    print(result.raw_text[:500])

    print(f"\n--- EXTRACTED FIELDS ({len(result.fields)}) ---")
    for name, field in result.fields.items():
        val_status = "✓" if field.validated else "⚠"
        print(f"  {val_status} {name}: {field.value} (conf={field.confidence:.0f}%)")

    if result.errors:
        print(f"\n--- ERRORS ---")
        for err in result.errors:
            print(f"  ! {err}")

    print()


def test_batch(directory: Path, limit: int = 10) -> None:
    """Test multiple files and show summary statistics."""
    print("=" * 80)
    print(f"BATCH TEST: {directory}")
    print("=" * 80)

    # Find all image files
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.pdf", "*.docx"]
    files = []
    for ext in extensions:
        files.extend(directory.glob(ext))

    files = sorted(files)[:limit]
    print(f"Found {len(files)} files to process\n")

    pipeline = RobustOCRPipeline(verbose=False)

    results = []
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {file_path.name}...", end=" ")
        result = pipeline.extract(file_path)

        status = "✓" if result.success else "✗"
        print(f"{status} conf={result.overall_confidence:.1f}% fields={len(result.fields)}")

        results.append({
            "file": file_path.name,
            "success": result.success,
            "confidence": result.overall_confidence,
            "quality": result.quality_score,
            "fields": len(result.fields),
            "engine": result.engine_used.value,
            "fallback": result.fallback_used,
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for r in results if r["success"])
    avg_conf = sum(r["confidence"] for r in results) / total if total > 0 else 0
    good_count = sum(1 for r in results if r["confidence"] >= 70)

    print(f"Total files: {total}")
    print(f"Successful: {successful}/{total} ({100*successful/total:.1f}%)")
    print(f"Average confidence: {avg_conf:.1f}%")
    print(f"Files with >70% confidence: {good_count}/{total} ({100*good_count/total:.1f}%)")

    # Quality breakdown
    quality_counts = {}
    for r in results:
        q = r["quality"]
        quality_counts[q] = quality_counts.get(q, 0) + 1
    print(f"\nQuality breakdown:")
    for q, count in sorted(quality_counts.items()):
        print(f"  {q}: {count}")

    # Engine breakdown
    engine_counts = {}
    for r in results:
        e = r["engine"]
        engine_counts[e] = engine_counts.get(e, 0) + 1
    print(f"\nEngine breakdown:")
    for e, count in sorted(engine_counts.items()):
        print(f"  {e}: {count}")

    # Top 5 best
    print(f"\nTop 5 by confidence:")
    for r in sorted(results, key=lambda x: x["confidence"], reverse=True)[:5]:
        print(f"  {r['file'][:50]}: {r['confidence']:.1f}%")

    # Bottom 5 worst
    print(f"\nBottom 5 by confidence:")
    for r in sorted(results, key=lambda x: x["confidence"])[:5]:
        print(f"  {r['file'][:50]}: {r['confidence']:.1f}%")


def main():
    project_root = Path(__file__).resolve().parents[1]
    lbmaske_dir = project_root / "Medical_report" / "Sample_reports" / "lbmaske"

    if len(sys.argv) > 1:
        # Test single file
        test_single_file(Path(sys.argv[1]))
    elif lbmaske_dir.exists():
        # Test batch on lbmaske
        test_batch(lbmaske_dir, limit=30)
    else:
        print(f"Directory not found: {lbmaske_dir}")
        print("Usage:")
        print("  python test_robust_ocr.py                  # Test batch on lbmaske")
        print("  python test_robust_ocr.py <file_path>      # Test single file")


if __name__ == "__main__":
    main()
