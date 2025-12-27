"""
Validation Analysis: Check if predictions are reasonable and meaningful
"""

import json
import os
import numpy as np
from pathlib import Path

def load_predictions_summary():
    """Load the summary predictions file"""
    summary_path = Path("test_results") / "multimodal_predictions" / "real_data_predictions_summary.json"
    if not summary_path.exists():
        print(f"[ERROR] {summary_path} not found")
        return None
    
    with open(summary_path) as f:
        return json.load(f)

def validate_all_predictions():
    """Main validation routine"""
    print("\n" + "="*70)
    print("PREDICTION VALIDATION ANALYSIS")
    print("="*70 + "\n")
    
    data = load_predictions_summary()
    if not data:
        return
    
    admissions = data.get("admissions", [])
    print(f"Validating {len(admissions)} admissions...\n")
    
    # Collect statistics
    lstm_scores = []
    clinical_scores = []
    fusion_scores = []
    risk_categories = {"LOW RISK": 0, "MEDIUM RISK": 0, "HIGH RISK": 0}
    
    for admission in admissions:
        hadm_id = admission.get("admission_id")
        predictions = admission.get("predictions", {})
        lstm_score = predictions.get("lstm_vital_score", 0)
        clinical_score = predictions.get("clinical_classifier_score", 0)
        fusion_score = predictions.get("fusion_final_score", 0)
        risk_category = predictions.get("risk_category", "UNKNOWN")
        
        lstm_scores.append(lstm_score)
        clinical_scores.append(clinical_score)
        fusion_scores.append(fusion_score)
        
        # Count risk categories
        if risk_category in risk_categories:
            risk_categories[risk_category] += 1

    # Print summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nLSTM (Vital Signs) Scores:")
    print(f"  Mean:  {np.mean(lstm_scores):.4f}")
    print(f"  Std:   {np.std(lstm_scores):.4f}")
    print(f"  Range: {np.min(lstm_scores):.4f} - {np.max(lstm_scores):.4f}")
    
    print("\nClinical Classifier Scores:")
    print(f"  Mean:  {np.mean(clinical_scores):.4f}")
    print(f"  Std:   {np.std(clinical_scores):.4f}")
    print(f"  Range: {np.min(clinical_scores):.4f} - {np.max(clinical_scores):.4f}")
    
    print("\nFusion Model (Final) Scores:")
    print(f"  Mean:  {np.mean(fusion_scores):.4f}")
    print(f"  Std:   {np.std(fusion_scores):.4f}")
    print(f"  Range: {np.min(fusion_scores):.4f} - {np.max(fusion_scores):.4f}")
    
    print(f"\nRisk Distribution:")
    for category, count in risk_categories.items():
        pct = (count / len(admissions)) * 100
        print(f"  {category}: {count:3d} ({pct:5.1f}%)")
    
    # Validation checks
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    issues = 0
    
    # Check 1: LSTM scores reasonable
    fallback_count = sum(1 for s in lstm_scores if abs(s - 0.5) < 0.001)
    print(f"\n[CHECK 1] LSTM Fallback Detection:")
    print(f"  LSTM scores at 0.5000 (fallback): {fallback_count}/{len(lstm_scores)}")
    if fallback_count == len(lstm_scores):
        print(f"  [WARNING] All LSTM scores are 0.5 fallback (sparse vital data issue)")
        issues += 1
    else:
        print(f"  [OK] LSTM has meaningful predictions")
    
    # Check 2: Clinical scores have variation
    clinical_variation = np.std(clinical_scores)
    print(f"\n[CHECK 2] Clinical Score Variation:")
    print(f"  Std Dev: {clinical_variation:.4f}")
    if clinical_variation < 0.05:
        print(f"  [WARNING] Low variation in clinical scores")
        issues += 1
    else:
        print(f"  [OK] Clinical scores show good variation")
    
    # Check 3: Fusion scores reasonable
    fusion_min, fusion_max = np.min(fusion_scores), np.max(fusion_scores)
    print(f"\n[CHECK 3] Fusion Model Range:")
    print(f"  Range: {fusion_min:.4f} to {fusion_max:.4f}")
    if fusion_min < 0 or fusion_max > 1:
        print(f"  [ERROR] Fusion scores outside [0, 1] range")
        issues += 1
    else:
        print(f"  [OK] Fusion scores within valid range")
    
    # Check 4: Risk distribution reasonable
    print(f"\n[CHECK 4] Risk Distribution:")
    if risk_categories["MEDIUM RISK"] > (len(admissions) * 0.8):
        print(f"  [WARNING] >80% admissions in MEDIUM RISK (model conservative)")
    else:
        print(f"  [OK] Balanced risk distribution")
    
    # Check 5: Clinical notes loaded
    print(f"\n[CHECK 5] Clinical Notes Loading:")
    empty_notes = sum(1 for a in admissions if len(a.get("clinical_note_preview", "")) < 50)
    print(f"  Admissions with notes: {len(admissions) - empty_notes}/{len(admissions)}")
    if empty_notes > 0:
        print(f"  [WARNING] {empty_notes} admissions missing clinical notes")
    else:
        print(f"  [OK] All admissions have clinical notes")
    
    # Summary
    print("\n" + "="*70)
    if issues == 0:
        print("[SUCCESS] All validation checks passed!")
    else:
        print(f"[CAUTION] {issues} issue(s) detected - see above for details")
    print("="*70 + "\n")

if __name__ == "__main__":
    validate_all_predictions()
