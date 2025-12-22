"""Production System Benchmark.

Validates:
- OCR accuracy on synthetic images
- Model prediction accuracy on test data
- End-to-end system performance
- Edge case handling

Run with: python experiments/production_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.production_ocr import ProductionOCR
from src.production_model import ProductionModel
from src.production_pipeline import ProductionPipeline, predict_risk_from_data
from src.mlp_tuning import load_splits, encode_categorical_features


def benchmark_model_accuracy():
    """Benchmark model accuracy on test set."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: MODEL ACCURACY ON TEST SET")
    print("=" * 70)

    # Load test data
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features(
        X_train, X_val, X_test
    )

    print(f"Test set size: {len(X_test)}")

    # Initialize production model
    model = ProductionModel(verbose=False, enable_shap=False)

    # Run predictions on test set
    predictions = []
    probabilities = []
    confidences = []

    print("Running predictions...")
    start_time = time.time()

    for idx in range(len(X_test_enc)):
        features = X_test_enc.iloc[idx]
        features_scaled = model.scaler.transform(features.values.reshape(1, -1))

        proba = model.model.predict_proba(features_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0

        predictions.append(pred)
        probabilities.append(proba)

    elapsed = time.time() - start_time

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    y_true = y_test.values

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix
    )

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    auc = roc_auc_score(y_true, probabilities)
    cm = confusion_matrix(y_true, predictions)

    print(f"\n--- Results ---")
    print(f"Accuracy:  {accuracy:.4f} (target: ≥0.9359)")
    print(f"Precision: {precision:.4f} (target: ≥0.88)")
    print(f"Recall:    {recall:.4f} (target: ≥0.9190)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print(f"\nProcessing time: {elapsed:.2f}s ({len(X_test)/elapsed:.1f} predictions/sec)")

    # Check success criteria
    success = True
    if accuracy < 0.9359:
        print(f"⚠️  Accuracy below target (0.9359)")
        success = False
    if recall < 0.9190:
        print(f"⚠️  Recall below target (0.9190)")
        success = False

    # Critical: Check false negatives on HIGH risk
    high_risk_fn = cm[1, 0]  # False negatives
    print(f"\nCritical False Negatives (HIGH→LOW): {high_risk_fn}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "false_negatives": high_risk_fn,
        "success": success,
    }


def benchmark_risk_level_accuracy():
    """Benchmark risk level classification accuracy."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: RISK LEVEL CLASSIFICATION")
    print("=" * 70)

    # Define test patients by expected risk level
    # Note: This is 10-year cardiovascular risk, where age is a dominant factor
    test_cases = [
        # HIGH risk cases (elderly with risk factors)
        {"name": "HIGH-1", "expected": "HIGH", "data": {
            "age": 72, "sex": 1, "systolic_bp": 170, "total_cholesterol": 260,
            "fasting_glucose": 155, "bmi": 34.5, "smoking": 1, "diabetes": 1
        }},
        {"name": "HIGH-2", "expected": "HIGH", "data": {
            "age": 68, "sex": 1, "systolic_bp": 165, "total_cholesterol": 245,
            "fasting_glucose": 148, "bmi": 32.8, "smoking": 1, "diabetes": 1
        }},
        {"name": "HIGH-3", "expected": "HIGH", "data": {
            "age": 66, "sex": 1, "systolic_bp": 155, "total_cholesterol": 250,
            "fasting_glucose": 160, "bmi": 36.2, "smoking": 1, "diabetes": 1
        }},
        # LOW risk cases (young, healthy)
        {"name": "LOW-1", "expected": "LOW", "data": {
            "age": 28, "sex": 0, "systolic_bp": 112, "total_cholesterol": 178,
            "fasting_glucose": 88, "bmi": 22.3, "smoking": 0, "diabetes": 0
        }},
        {"name": "LOW-2", "expected": "LOW", "data": {
            "age": 32, "sex": 0, "systolic_bp": 108, "total_cholesterol": 165,
            "fasting_glucose": 82, "bmi": 21.0, "smoking": 0, "diabetes": 0
        }},
        {"name": "LOW-3", "expected": "LOW", "data": {
            "age": 45, "sex": 0, "systolic_bp": 118, "total_cholesterol": 185,
            "fasting_glucose": 92, "bmi": 23.8, "smoking": 0, "diabetes": 0
        }},
        # MEDIUM risk cases (middle-aged with some factors)
        {"name": "MEDIUM-1", "expected": "MEDIUM", "data": {
            "age": 58, "sex": 1, "systolic_bp": 138, "total_cholesterol": 215,
            "fasting_glucose": 110, "bmi": 28.4, "smoking": 0, "diabetes": 0
        }},
        {"name": "MEDIUM-2", "expected": "MEDIUM", "data": {
            "age": 62, "sex": 1, "systolic_bp": 146, "total_cholesterol": 232,
            "fasting_glucose": 118, "bmi": 29.7, "smoking": 1, "diabetes": 0
        }},
        # Edge cases - clinical reality for 10-year risk model
        {"name": "EDGE-elderly-healthy", "expected": "HIGH", "data": {
            # Elderly have elevated 10-year risk even with good vitals
            "age": 78, "sex": 1, "systolic_bp": 120, "total_cholesterol": 185,
            "fasting_glucose": 92, "bmi": 23.5, "smoking": 0, "diabetes": 0
        }},
        {"name": "EDGE-young-risky", "expected": "LOW", "data": {
            # Young have low 10-year risk even with risk factors (high lifetime risk)
            "age": 32, "sex": 1, "systolic_bp": 150, "total_cholesterol": 245,
            "fasting_glucose": 145, "bmi": 36.5, "smoking": 1, "diabetes": 1
        }},
    ]

    model = ProductionModel(verbose=False, enable_shap=False)

    results = {
        "HIGH": {"correct": 0, "total": 0},
        "MEDIUM": {"correct": 0, "total": 0},
        "LOW": {"correct": 0, "total": 0},
    }

    false_negatives = []  # HIGH predicted as LOW

    print("\n--- Per-Case Results ---")
    for case in test_cases:
        prediction = model.predict(case["data"])
        expected = case["expected"]
        predicted = prediction.risk_level

        results[expected]["total"] += 1
        correct = predicted == expected

        # Allow MEDIUM to be adjacent (HIGH or LOW)
        if expected == "MEDIUM" and predicted in ["HIGH", "LOW"]:
            # Not ideal but not critical
            pass

        if correct:
            results[expected]["correct"] += 1
            status = "✓"
        else:
            status = "✗"

        # Track critical false negatives
        if expected == "HIGH" and predicted == "LOW":
            false_negatives.append(case["name"])

        print(f"  {status} {case['name']}: expected={expected}, predicted={predicted} "
              f"(prob={prediction.risk_probability:.1f}%)")

    print("\n--- Summary by Risk Level ---")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        correct = results[level]["correct"]
        total = results[level]["total"]
        acc = (correct / total * 100) if total > 0 else 0
        target = "≥70%" if level == "MEDIUM" else "≥90%"
        status = "✓" if acc >= 70 else "⚠️"
        print(f"  {status} {level}: {correct}/{total} ({acc:.1f}%) [target: {target}]")

    print(f"\n--- Critical Check ---")
    if false_negatives:
        print(f"  ⚠️  HIGH→LOW false negatives: {false_negatives}")
    else:
        print(f"  ✓ Zero HIGH→LOW false negatives")

    return results, false_negatives


def benchmark_ocr_on_synthetic():
    """Benchmark OCR on synthetic images."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: OCR ACCURACY ON SYNTHETIC IMAGES")
    print("=" * 70)

    images_dir = PROJECT_ROOT / "experiments" / "synthetic_images"
    csv_path = PROJECT_ROOT / "experiments" / "synthetic_ground_truth.csv"

    if not csv_path.exists():
        print("⚠️  Synthetic ground truth not found. Skipping OCR benchmark.")
        print(f"   Run: python experiments/generate_synthetic_patients.py")
        return None

    if not images_dir.exists():
        print("⚠️  Synthetic images not found. Skipping OCR benchmark.")
        print(f"   Run: python experiments/generate_synthetic_images.py")
        return None

    ground_truth = pd.read_csv(csv_path)
    ocr = ProductionOCR(verbose=False)

    results = []
    # Map ground truth field names to OCR field names
    field_mapping = {
        "age": "age",
        "systolic_bp": "systolic_bp",
        "cholesterol": "total_cholesterol",
        "glucose": "fasting_glucose",
    }

    print(f"\nTesting {len(ground_truth)} synthetic images...")

    for _, patient in ground_truth.iterrows():
        patient_id = patient["id"]
        image_path = images_dir / f"{patient_id}.png"

        if not image_path.exists():
            print(f"  ⚠️  Missing: {patient_id}.png")
            continue

        ocr_result = ocr.extract(image_path)

        # Check accuracy on key fields
        correct = 0
        for ground_field, ocr_field in field_mapping.items():
            ground_val = float(patient.get(ground_field, 0))
            if ocr_field in ocr_result.fields:
                ocr_val = float(ocr_result.fields[ocr_field].value)
                if ground_val > 0:
                    rel_err = abs(ground_val - ocr_val) / ground_val
                    if rel_err <= 0.05:
                        correct += 1

        accuracy = (correct / len(field_mapping)) * 100
        results.append({
            "patient_id": patient_id,
            "accuracy": accuracy,
            "ocr_confidence": ocr_result.overall_confidence,
            "quality": ocr_result.document_quality,
        })

        status = "✓" if accuracy >= 75 else "✗"
        print(f"  {status} {patient_id}: {accuracy:.1f}% accuracy, "
              f"{ocr_result.overall_confidence:.1f}% confidence")

    if results:
        df = pd.DataFrame(results)
        avg_accuracy = df["accuracy"].mean()
        perfect = (df["accuracy"] == 100).sum()
        good = (df["accuracy"] >= 75).sum()

        print(f"\n--- Summary ---")
        print(f"  Average OCR accuracy: {avg_accuracy:.1f}%")
        print(f"  Perfect (100%): {perfect}/{len(df)}")
        print(f"  Good (≥75%): {good}/{len(df)}")

        return df

    return None


def benchmark_uncertainty_and_explainability():
    """Benchmark uncertainty quantification and explainability."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: UNCERTAINTY & EXPLAINABILITY")
    print("=" * 70)

    model = ProductionModel(verbose=False, enable_shap=False)

    test_cases = [
        {"name": "Clear HIGH", "data": {
            "age": 75, "systolic_bp": 180, "total_cholesterol": 280,
            "fasting_glucose": 180, "bmi": 38, "smoking": 1, "diabetes": 1
        }},
        {"name": "Clear LOW", "data": {
            "age": 25, "systolic_bp": 110, "total_cholesterol": 160,
            "fasting_glucose": 80, "bmi": 22, "smoking": 0, "diabetes": 0
        }},
        {"name": "Uncertain (borderline)", "data": {
            "age": 55, "systolic_bp": 135, "total_cholesterol": 210,
            "fasting_glucose": 105, "bmi": 27, "smoking": 0, "diabetes": 0
        }},
        {"name": "Missing data", "data": {
            "age": 60, "systolic_bp": 140,
            # Missing cholesterol, glucose, etc.
        }},
    ]

    print("\n--- Results ---")
    for case in test_cases:
        result = model.predict(case["data"])

        print(f"\n{case['name']}:")
        print(f"  Risk: {result.risk_level} ({result.risk_probability:.1f}%)")
        print(f"  Confidence: {result.confidence:.1f}%")
        print(f"  Epistemic uncertainty: {result.epistemic_uncertainty:.4f}")
        print(f"  Aleatoric uncertainty: {result.aleatoric_uncertainty:.4f}")
        print(f"  Needs review: {result.needs_review}")
        if result.explanation:
            print(f"  Explanation: {result.explanation[:80]}...")

    # Verify clear cases have high confidence
    clear_high = model.predict(test_cases[0]["data"])
    clear_low = model.predict(test_cases[1]["data"])

    print(f"\n--- Confidence Check ---")
    if clear_high.confidence > 50:
        print(f"  ✓ Clear HIGH case has decent confidence: {clear_high.confidence:.1f}%")
    else:
        print(f"  ⚠️  Clear HIGH case has low confidence: {clear_high.confidence:.1f}%")

    if clear_low.confidence > 50:
        print(f"  ✓ Clear LOW case has decent confidence: {clear_low.confidence:.1f}%")
    else:
        print(f"  ⚠️  Clear LOW case has low confidence: {clear_low.confidence:.1f}%")


def benchmark_pipeline_performance():
    """Benchmark end-to-end pipeline performance."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: PIPELINE PERFORMANCE")
    print("=" * 70)

    # Test direct prediction performance
    test_patient = {
        "age": 55,
        "systolic_bp": 140,
        "total_cholesterol": 220,
        "fasting_glucose": 100,
        "bmi": 28,
        "smoking": 0,
        "diabetes": 0,
    }

    print("\nWarming up...")
    _ = predict_risk_from_data(test_patient)

    print("Running 100 predictions...")
    start = time.time()
    for _ in range(100):
        predict_risk_from_data(test_patient)
    elapsed = time.time() - start

    avg_time = elapsed / 100 * 1000  # Convert to ms

    print(f"\n--- Results ---")
    print(f"  Total time for 100 predictions: {elapsed:.2f}s")
    print(f"  Average time per prediction: {avg_time:.1f}ms")
    print(f"  Throughput: {100/elapsed:.1f} predictions/second")

    if avg_time < 100:
        print(f"  ✓ Performance: EXCELLENT (<100ms)")
    elif avg_time < 500:
        print(f"  ✓ Performance: GOOD (<500ms)")
    else:
        print(f"  ⚠️  Performance: SLOW (>500ms)")


def run_all_benchmarks():
    """Run all benchmarks and generate summary."""
    print("\n" + "=" * 70)
    print("CARDIODETECT PRODUCTION SYSTEM BENCHMARK")
    print("=" * 70)
    print(f"Started at: {pd.Timestamp.now()}")

    results = {}

    # Benchmark 1: Model accuracy
    results["model"] = benchmark_model_accuracy()

    # Benchmark 2: Risk level accuracy
    level_results, false_negs = benchmark_risk_level_accuracy()
    results["risk_levels"] = level_results
    results["false_negatives"] = false_negs

    # Benchmark 3: OCR accuracy
    results["ocr"] = benchmark_ocr_on_synthetic()

    # Benchmark 4: Uncertainty & Explainability
    benchmark_uncertainty_and_explainability()

    # Benchmark 5: Performance
    benchmark_pipeline_performance()

    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    success_criteria = []

    # Model accuracy
    if results["model"]:
        model = results["model"]
        success_criteria.append(("Accuracy ≥93.59%", model["accuracy"] >= 0.9359))
        success_criteria.append(("Recall ≥91.90%", model["recall"] >= 0.9190))
        success_criteria.append(("Zero HIGH→LOW FN", len(results["false_negatives"]) == 0))

    # Risk level accuracy
    if results["risk_levels"]:
        rl = results["risk_levels"]
        high_acc = rl["HIGH"]["correct"] / rl["HIGH"]["total"] if rl["HIGH"]["total"] > 0 else 0
        low_acc = rl["LOW"]["correct"] / rl["LOW"]["total"] if rl["LOW"]["total"] > 0 else 0
        success_criteria.append(("HIGH accuracy ≥90%", high_acc >= 0.9))
        success_criteria.append(("LOW accuracy ≥90%", low_acc >= 0.9))

    # OCR accuracy
    if results["ocr"] is not None:
        ocr_avg = results["ocr"]["accuracy"].mean()
        success_criteria.append(("OCR accuracy ≥85%", ocr_avg >= 85))

    print("\n--- Success Criteria ---")
    all_passed = True
    for criterion, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")
        if not passed:
            all_passed = False

    print("\n--- Overall Status ---")
    if all_passed:
        print("  🎉 ALL CRITERIA PASSED - System is production-ready!")
    else:
        print("  ⚠️  Some criteria failed - Review needed")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
