"""
Complete Accuracy Test - OCR + MLP
Tests all metrics: Accuracy, Precision, Recall, F1, ROC-AUC
Author: CardioDetect Project
Date: November 30, 2025
"""

import sys
from pathlib import Path
import numpy as np
import joblib
from difflib import SequenceMatcher

# Add src to path so I can import internal modules
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("                    CARDIODETECT - COMPLETE ACCURACY TEST")
print("=" * 80)
print()

# ============================================================================
# TEST 1: MLP MODEL - ALL METRICS
# ============================================================================
print("\n" + "┌" + "─" * 78 + "┐")
print("│" + " " * 25 + "TEST 1: MLP v2 Best Model" + " " * 28 + "│")
print("└" + "─" * 78 + "┘")
print()

mlp_success = False
mlp_results = {}

try:
    from src.mlp_tuning import load_splits, encode_categorical_features
    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        precision_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    print("⏳ Step 1/4: Loading test data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    # Apply the same one-hot encoding used during training so that
    # the feature schema matches the one used for mlp_v2_best.
    X_train_enc, X_val_enc, X_test_enc = encode_categorical_features(
        X_train, X_val, X_test
    )
    print(f"   ✅ Loaded {len(X_test_enc)} encoded test patients")

    print("\n⏳ Step 2/4: Loading mlp_v2_best model artifact...")
    model_path = Path("models/mlp_v2_best.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    print(f"   ✅ Model loaded: {type(model).__name__}")

    print("\n⏳ Step 3/4: Making predictions on test set...")
    X_test_scaled = scaler.transform(X_test_enc.values)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    print("   ✅ Predictions completed")

    print("\n⏳ Step 4/4: Computing all metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value (=precision)

    print("   ✅ All metrics computed")

    # Store results
    mlp_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    print()
    print("=" * 80)
    print("                         MLP v2 BEST - TEST SET RESULTS")
    print("=" * 80)
    print()
    print(f"  Test Set:              {len(y_test)} patients")
    print(
        f"  Positive (CHD=1):      {sum(y_test)} patients "
        f"({100 * sum(y_test) / len(y_test):.1f}%)"
    )
    print(
        f"  Negative (CHD=0):      {len(y_test) - sum(y_test)} patients "
        f"({100 * (len(y_test) - sum(y_test)) / len(y_test):.1f}%)"
    )
    print()
    print("─" * 80)
    print("                            CORE METRICS")
    print("─" * 80)
    print()
    print(f"  🎯 ACCURACY:             {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  🎯 PRECISION:            {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"  🎯 RECALL (Sensitivity): {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"  📊 F1-SCORE:             {f1:.4f}")
    print(f"  📊 ROC-AUC:              {roc_auc:.4f}")
    print(f"  📊 SPECIFICITY:          {specificity:.4f}  ({specificity * 100:.2f}%)")
    print()
    print("─" * 80)
    print("                          CONFUSION MATRIX")
    print("─" * 80)
    print()
    print("                     Predicted")
    print("                   0          1")
    print(f"            0   [[{cm[0, 0]:5d}    {cm[0, 1]:5d}]   (Actual Negative)")
    print(f"  Actual    1    [{cm[1, 0]:5d}    {cm[1, 1]:5d}]]  (Actual Positive)")
    print()
    print(
        f"  True Negatives  (TN): {tn:4d}  "
        f"({100 * tn / (tn + fp):.1f}% of negatives correctly identified)"
    )
    print(
        f"  False Positives (FP): {fp:4d}  "
        f"({100 * fp / (tn + fp):.1f}% of negatives misclassified)"
    )
    print(
        f"  False Negatives (FN): {fn:4d}  "
        f"({100 * fn / (fn + tp):.1f}% of positives missed)"
    )
    print(
        f"  True Positives  (TP): {tp:4d}  "
        f"({100 * tp / (fn + tp):.1f}% of positives caught)"
    )
    print()
    print("─" * 80)
    print("                        CLINICAL METRICS")
    print("─" * 80)
    print()
    print(f"  Positive Predictive Value (PPV):  {ppv:.4f}  ({ppv * 100:.1f}%)")
    print(
        f"    → When model predicts CHD, it's correct {ppv * 100:.1f}% of the time"
    )
    print()
    print(f"  Negative Predictive Value (NPV):  {npv:.4f}  ({npv * 100:.1f}%)")
    print(
        f"    → When model predicts NO CHD, it's correct {npv * 100:.1f}% of the time"
    )
    print()
    print(f"  Sensitivity (Recall):             {recall:.4f}  ({recall * 100:.1f}%)")
    print(
        f"    → Catches {recall * 100:.1f}% of all CHD cases (high = good for screening)"
    )
    print()
    print(f"  Specificity:                      {specificity:.4f}  ({specificity * 100:.1f}%)")
    print(
        f"    → Correctly identifies {specificity * 100:.1f}% of healthy patients"
    )
    print()
    print("─" * 80)
    print("                          TARGET VERIFICATION")
    print("─" * 80)
    print()

    # Check targets
    targets_met = []
    targets_missed = []

    if accuracy >= 0.90:
        targets_met.append(
            f"✅ ACCURACY ≥ 90%:     {accuracy * 100:.2f}% ✓"
        )
    else:
        targets_missed.append(
            f"❌ ACCURACY < 90%:     {accuracy * 100:.2f}%"
        )

    if recall >= 0.80:
        targets_met.append(f"✅ RECALL ≥ 80%:       {recall * 100:.2f}% ✓")
    else:
        targets_missed.append(
            f"⚠️  RECALL < 80%:       {recall * 100:.2f}%"
        )

    if precision >= 0.70:
        targets_met.append(f"✅ PRECISION ≥ 70%:    {precision * 100:.2f}% ✓")
    else:
        targets_missed.append(
            f"⚠️  PRECISION < 70%:    {precision * 100:.2f}%"
        )

    if roc_auc >= 0.90:
        targets_met.append(f"✅ ROC-AUC ≥ 0.90:     {roc_auc:.4f} ✓")
    else:
        targets_missed.append(f"⚠️  ROC-AUC < 0.90:     {roc_auc:.4f}")

    for target in targets_met:
        print(f"  {target}")
    for target in targets_missed:
        print(f"  {target}")

    print()
    print("=" * 80)

    mlp_success = True

except Exception as e:  # noqa: BLE001
    print(f"\n❌ MLP Test Failed: {e}")
    import traceback

    traceback.print_exc()


# ============================================================================
# TEST 2: OCR ACCURACY
# ============================================================================
print("\n\n")
print("┌" + "─" * 78 + "┐")
print("│" + " " * 25 + "TEST 2: OCR Accuracy Test" + " " * 28 + "│")
print("└" + "─" * 78 + "┘")
print()

ocr_success = False
ocr_results = {}
pdf_path = None

try:
    import fitz  # PyMuPDF

    # Locate the CBC report
    print("⏳ Step 1/5: Locating sample medical report...")
    pdf_path = Path(
        "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
    )

    if not pdf_path.exists():
        # Try alternate locations
        alt_paths = [
            Path(
                "data/raw/ocr_external/"
                "CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf"
            ),
            Path("./CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf"),
        ]
        for alt in alt_paths:
            if alt.exists():
                pdf_path = alt
                break

    if not pdf_path.exists():
        raise FileNotFoundError(
            "CBC report PDF not found. Please place it in project root."
        )

    print(f"   ✅ Found: {pdf_path.name}")

    # Extract text with OCR (here: text extraction via PyMuPDF)
    print("\n⏳ Step 2/5: Extracting text with PyMuPDF...")
    doc = fitz.open(pdf_path)
    ocr_text = ""
    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text()
        ocr_text += page_text
        print(f"   → Page {page_num}: {len(page_text)} characters")
    doc.close()
    print(f"   ✅ Total extracted: {len(ocr_text)} characters")

    # Display sample
    print("\n⏳ Step 3/5: OCR Output Preview...")
    print("─" * 80)
    print(ocr_text[:600])
    if len(ocr_text) > 600:
        print(f"... (showing first 600 of {len(ocr_text)} characters)")
    print("─" * 80)

    # Ground truth (manually transcribed from the PDF)
    print("\n⏳ Step 4/5: Comparing with ground truth...")
    ground_truth = """Patient Name: Jeevan
Age: 21 years
Sex: Male
Date: 15/06/2021

HAEMATOLOGY - COMPLETE BLOOD COUNT

Test Name            Result    Unit        Reference Range
Haemoglobin          15.5      g/dL        13.0 - 17.0
RBC Count            5.2       mill/cumm   4.5 - 5.5
PCV                  46.8      %           40 - 50
MCV                  90.0      fL          83 - 101
MCH                  29.8      pg          27 - 32
MCHC                 33.1      g/dL        31.5 - 34.5
RDW                  13.2      %           11.6 - 14.0
Total WBC Count      8500      cumm        4000 - 11000
Neutrophils          65        %           40 - 80
Lymphocytes          30        %           20 - 40
Monocytes            4         %           2 - 10
Eosinophils          1         %           1 - 6
Basophils            0         %           0 - 2
Platelet Count       250000    cumm        150000 - 410000"""

    # Normalize both texts for comparison
    def normalize_text(text: str) -> str:
        """Normalize text for comparison: collapse whitespace & strip symbols."""

        import re as _re

        text = _re.sub(r"\s+", " ", text)
        text = _re.sub(r"[^\w\s\.\-\:\(\)\/]", "", text)
        return text.lower().strip()

    gt_normalized = normalize_text(ground_truth)
    ocr_normalized = normalize_text(ocr_text)

    # Character-level accuracy
    char_matcher = SequenceMatcher(None, gt_normalized, ocr_normalized)
    char_accuracy = char_matcher.ratio()

    # Word-level accuracy
    gt_words = set(gt_normalized.split())
    ocr_words = set(ocr_normalized.split())
    common_words = gt_words.intersection(ocr_words)
    word_accuracy = len(common_words) / len(gt_words) if gt_words else 0.0

    # Field extraction accuracy
    import re

    def extract_field(text: str, pattern: str) -> str | None:
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    fields_to_check = {
        "age": r"age[:\s]+(\d+)",
        "sex": r"sex[:\s]+(male|female)",
        "hemoglobin": r"haemoglobin[:\s]+(\d+\.?\d*)",
        "rbc": r"rbc count[:\s]+(\d+\.?\d*)",
        "wbc": r"wbc count[:\s]+(\d+)",
        "platelet": r"platelet count[:\s]+(\d+)",
    }

    field_matches = 0
    field_total = len(fields_to_check)

    print("\n   Field Extraction:")
    for field_name, pattern in fields_to_check.items():
        gt_value = extract_field(ground_truth, pattern)
        ocr_value = extract_field(ocr_text, pattern)
        match = "✓" if gt_value == ocr_value else "✗"
        field_matches += 1 if gt_value == ocr_value else 0
        print(f"   {match} {field_name}: GT='{gt_value}' | OCR='{ocr_value}'")

    field_accuracy = field_matches / field_total if field_total > 0 else 0.0

    # Store results
    ocr_results = {
        "char_accuracy": char_accuracy,
        "word_accuracy": word_accuracy,
        "field_accuracy": field_accuracy,
        "total_chars": len(ocr_text),
        "words_matched": len(common_words),
        "total_words": len(gt_words),
        "fields_matched": field_matches,
        "total_fields": field_total,
    }

    print("\n⏳ Step 5/5: Computing OCR metrics...")
    print()
    print("=" * 80)
    print("                            OCR ACCURACY RESULTS")
    print("=" * 80)
    print()
    print("─" * 80)
    print("                            ACCURACY METRICS")
    print("─" * 80)
    print()
    print(
        f"  🎯 CHARACTER-LEVEL ACCURACY:  {char_accuracy:.4f}  "
        f"({char_accuracy * 100:.2f}%)"
    )
    print(
        f"  🎯 WORD-LEVEL ACCURACY:       {word_accuracy:.4f}  "
        f"({word_accuracy * 100:.2f}%)"
    )
    print(
        f"  🎯 FIELD EXTRACTION ACCURACY: {field_accuracy:.4f}  "
        f"({field_accuracy * 100:.2f}%)"
    )
    print()
    print(f"  Total Characters Extracted:   {len(ocr_text)}")
    print(f"  Words Matched:                {len(common_words)}/{len(gt_words)}")
    print(f"  Fields Matched:               {field_matches}/{field_total}")
    print()
    print("─" * 80)
    print("                          TARGET VERIFICATION")
    print("─" * 80)
    print()

    if char_accuracy >= 0.95:
        print(
            f"  ✅ CHARACTER ACCURACY ≥ 95%:  {char_accuracy * 100:.2f}% ✓"
        )
    else:
        print(
            f"  ⚠️  CHARACTER ACCURACY < 95%:  {char_accuracy * 100:.2f}%"
        )

    if word_accuracy >= 0.90:
        print(
            f"  ✅ WORD ACCURACY ≥ 90%:       {word_accuracy * 100:.2f}% ✓"
        )
    else:
        print(
            f"  ⚠️  WORD ACCURACY < 90%:       {word_accuracy * 100:.2f}%"
        )

    if field_accuracy >= 0.80:
        print(
            f"  ✅ FIELD ACCURACY ≥ 80%:      {field_accuracy * 100:.2f}% ✓"
        )
    else:
        print(
            f"  ⚠️  FIELD ACCURACY < 80%:      {field_accuracy * 100:.2f}%"
        )

    print()
    print("=" * 80)

    ocr_success = True

except Exception as e:  # noqa: BLE001
    print(f"\n❌ OCR Test Failed: {e}")
    import traceback

    traceback.print_exc()


# ============================================================================
# TEST 3: END-TO-END INTEGRATION
# ============================================================================
print("\n\n")
print("┌" + "─" * 78 + "┐")
print("│" + " " * 20 + "TEST 3: OCR → MLP Integration Test" + " " * 23 + "│")
print("└" + "─" * 78 + "┘")
print()

integration_success = False

try:
    from src.ocr_risk_prediction import run_ocr_risk_prediction

    if pdf_path is None:
        raise RuntimeError("PDF path is not available from OCR test.")

    print("⏳ Testing end-to-end pipeline (OCR → Feature Extraction → MLP)...")
    print()

    result = run_ocr_risk_prediction(str(pdf_path))

    print("=" * 80)
    print("                    END-TO-END PIPELINE RESULTS")
    print("=" * 80)
    print()
    print(f"  Parsed Age:           {result.get('parsed_age', 'N/A')} years")
    sex_code = result.get("parsed_sex_code", None)
    if sex_code == 1:
        sex_label = "Male"
    elif sex_code == 0:
        sex_label = "Female"
    else:
        sex_label = "Unknown"
    print(f"  Parsed Sex:           {sex_label}")
    prob = float(result.get("risk_probability", 0.0))
    print(f"  Risk Probability:     {prob:.4f} ({prob * 100:.2f}%)")
    label = int(result.get("predicted_label", 0))
    print(
        f"  Predicted Risk:       {'HIGH RISK' if label == 1 else 'LOW RISK'}"
    )
    print()
    print("  ✅ Pipeline executed successfully")
    print()
    print("=" * 80)

    integration_success = True

except Exception as e:  # noqa: BLE001
    print(f"❌ Integration Test Failed: {e}")
    import traceback

    traceback.print_exc()


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n")
print("=" * 80)
print("                            FINAL TEST SUMMARY")
print("=" * 80)
print()

if mlp_success:
    print("✅ TEST 1: MLP Model Accuracy")
    print(f"   Accuracy:  {mlp_results['accuracy'] * 100:.2f}%")
    print(f"   Precision: {mlp_results['precision'] * 100:.2f}%")
    print(f"   Recall:    {mlp_results['recall'] * 100:.2f}%")
    print(f"   F1-Score:  {mlp_results['f1']:.4f}")
    print(f"   ROC-AUC:   {mlp_results['roc_auc']:.4f}")
else:
    print("❌ TEST 1: MLP Model - FAILED")

print()

if ocr_success:
    print("✅ TEST 2: OCR Accuracy")
    print(f"   Character Accuracy: {ocr_results['char_accuracy'] * 100:.2f}%")
    print(f"   Word Accuracy:      {ocr_results['word_accuracy'] * 100:.2f}%")
    print(f"   Field Accuracy:     {ocr_results['field_accuracy'] * 100:.2f}%")
else:
    print("❌ TEST 2: OCR Accuracy - FAILED")

print()

if integration_success:
    print("✅ TEST 3: End-to-End Integration - COMPLETED")
else:
    print("❌ TEST 3: End-to-End Integration - FAILED")

print()
print("=" * 80)
print()
print("All tests completed. Review results above.")
print()
print("=" * 80)
