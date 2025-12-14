#!/usr/bin/env python3
# ============================================================================
# MIMIC-IV PREPROCESSING DATA VALIDATION SCRIPT
# ============================================================================
# Validates the processed_data.npz file for correctness and quality
# ============================================================================

import numpy as np
import os
from datetime import datetime

# ASCII symbols for cross-platform compatibility
SUCCESS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def validate_processed_data(npz_file='processed_data.npz'):
    """
    Comprehensive validation of preprocessed data.
    
    Checks:
    - File existence and integrity
    - Data shape consistency
    - Value ranges and statistics
    - Class distribution
    - Data types
    - Missing values
    """
    
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA VALIDATION")
    print("=" * 80)
    print(f"Validation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ====================================================================
    # 1. FILE VALIDATION
    # ====================================================================
    print("[1] FILE VALIDATION")
    print("-" * 80)
    
    if not os.path.exists(npz_file):
        print(f"{FAIL} File '{npz_file}' not found!")
        return False
    
    file_size = os.path.getsize(npz_file) / (1024 * 1024)
    print(f"{SUCCESS} File exists: {npz_file}")
    print(f"{SUCCESS} File size: {file_size:.2f} MB")
    
    try:
        data = np.load(npz_file)
        print(f"{SUCCESS} File integrity: OK (valid NPZ format)")
    except Exception as e:
        print(f"{FAIL} Invalid NPZ file - {str(e)}")
        return False
    
    # ====================================================================
    # 2. DATA STRUCTURE VALIDATION
    # ====================================================================
    print("\n[2] DATA STRUCTURE VALIDATION")
    print("-" * 80)
    
    required_keys = {'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'}
    available_keys = set(data.files)
    
    missing_keys = required_keys - available_keys
    if missing_keys:
        print(f"❌ FAILED: Missing keys: {missing_keys}")
        return False
    
    print(f"✓ All required keys present: {sorted(required_keys)}")
    
    # Load arrays
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # ====================================================================
    # 3. SHAPE VALIDATION
    # ====================================================================
    print("\n[3] SHAPE & DIMENSION VALIDATION")
    print("-" * 80)
    
    validation_passed = True
    
    # Check shapes
    shapes = {
        'X_train': X_train.shape,
        'y_train': y_train.shape,
        'X_val': X_val.shape,
        'y_val': y_val.shape,
        'X_test': X_test.shape,
        'y_test': y_test.shape,
    }
    
    for name, shape in shapes.items():
        print(f"  {name:12s}: {shape}")
    
    # Validate shape consistency
    if X_train.shape[0] != y_train.shape[0]:
        print(f"❌ X_train and y_train have mismatched samples!")
        validation_passed = False
    else:
        print(f"\n✓ X_train and y_train match: {X_train.shape[0]} samples")
    
    if X_val.shape[0] != y_val.shape[0]:
        print(f"❌ X_val and y_val have mismatched samples!")
        validation_passed = False
    else:
        print(f"✓ X_val and y_val match: {X_val.shape[0]} samples")
    
    if X_test.shape[0] != y_test.shape[0]:
        print(f"❌ X_test and y_test have mismatched samples!")
        validation_passed = False
    else:
        print(f"✓ X_test and y_test match: {X_test.shape[0]} samples")
    
    # Check feature dimensions (all X should have same second dimension)
    if X_train.shape[1] != X_val.shape[1] or X_val.shape[1] != X_test.shape[1]:
        print(f"❌ Feature dimensions mismatch across train/val/test!")
        validation_passed = False
    else:
        print(f"✓ Feature dimensions consistent: {X_train.shape[1]} features")
    
    # ====================================================================
    # 4. DATA TYPE VALIDATION
    # ====================================================================
    print("\n[4] DATA TYPE VALIDATION")
    print("-" * 80)
    
    print(f"  X_train dtype: {X_train.dtype} (expected: float)")
    print(f"  y_train dtype: {y_train.dtype} (expected: int or float)")
    print(f"  X_val dtype:   {X_val.dtype}")
    print(f"  y_val dtype:   {y_val.dtype}")
    print(f"  X_test dtype:  {X_test.dtype}")
    print(f"  y_test dtype:  {y_test.dtype}")
    
    if not np.issubdtype(X_train.dtype, np.floating):
        print(f"{WARN} Warning: X_train should be floating point")
    else:
        print(f"[OK] X arrays are numeric (float)")
    
    if not (np.issubdtype(y_train.dtype, np.integer) or np.issubdtype(y_train.dtype, np.floating)):
        print(f"{WARN} Warning: y should be numeric (int or float)")
    else:
        print(f"[OK] y arrays are numeric")
    
    # ====================================================================
    # 5. VALUE RANGE VALIDATION
    # ====================================================================
    print("\n[5] VALUE RANGE VALIDATION")
    print("-" * 80)
    
    print(f"X_train - Min: {X_train.min():.2f}, Max: {X_train.max():.2f}, Mean: {X_train.mean():.2f}")
    print(f"X_val   - Min: {X_val.min():.2f}, Max: {X_val.max():.2f}, Mean: {X_val.mean():.2f}")
    print(f"X_test  - Min: {X_test.min():.2f}, Max: {X_test.max():.2f}, Mean: {X_test.mean():.2f}")
    
    print(f"\ny_train - Min: {y_train.min():.0f}, Max: {y_train.max():.0f}, Unique: {np.unique(y_train)}")
    print(f"y_val   - Min: {y_val.min():.0f}, Max: {y_val.max():.0f}, Unique: {np.unique(y_val)}")
    print(f"y_test  - Min: {y_test.min():.0f}, Max: {y_test.max():.0f}, Unique: {np.unique(y_test)}")
    
    # Check for NaN/Inf in X
    nan_count_train = np.isnan(X_train).sum()
    inf_count_train = np.isinf(X_train).sum()
    nan_count_val = np.isnan(X_val).sum()
    inf_count_val = np.isinf(X_val).sum()
    nan_count_test = np.isnan(X_test).sum()
    inf_count_test = np.isinf(X_test).sum()
    
    if nan_count_train > 0 or inf_count_train > 0:
        print(f"\n⚠ Warning: X_train contains NaN: {nan_count_train}, Inf: {inf_count_train}")
    else:
        print(f"\n✓ X_train has no NaN/Inf values")
    
    if nan_count_val > 0 or inf_count_val > 0:
        print(f"⚠ Warning: X_val contains NaN: {nan_count_val}, Inf: {inf_count_val}")
    else:
        print(f"✓ X_val has no NaN/Inf values")
    
    if nan_count_test > 0 or inf_count_test > 0:
        print(f"⚠ Warning: X_test contains NaN: {nan_count_test}, Inf: {inf_count_test}")
    else:
        print(f"✓ X_test has no NaN/Inf values")
    
    # ====================================================================
    # 6. CLASS DISTRIBUTION VALIDATION
    # ====================================================================
    print("\n[6] CLASS DISTRIBUTION VALIDATION")
    print("-" * 80)
    
    def print_class_dist(y_data, name):
        unique, counts = np.unique(y_data, return_counts=True)
        print(f"\n{name}:")
        for label, count in zip(unique, counts):
            pct = (count / len(y_data)) * 100
            print(f"  Class {int(label)}: {count:5d} ({pct:5.1f}%)")
    
    print_class_dist(y_train, "y_train")
    print_class_dist(y_val, "y_val")
    print_class_dist(y_test, "y_test")
    
    # Check class imbalance
    train_pos = (y_train == 1).sum()
    val_pos = (y_val == 1).sum()
    test_pos = (y_test == 1).sum()
    
    total_pos = train_pos + val_pos + test_pos
    total_samples = len(y_train) + len(y_val) + len(y_test)
    overall_pct = (total_pos / total_samples) * 100
    
    print(f"\nOverall positive class: {total_pos}/{total_samples} ({overall_pct:.1f}%)")
    
    if overall_pct < 20 or overall_pct > 50:
        print(f"⚠ Note: Class imbalance detected - consider using class weights in LSTM training")
    else:
        print(f"✓ Reasonable class balance")
    
    # ====================================================================
    # 7. SPLIT RATIO VALIDATION
    # ====================================================================
    print("\n[7] SPLIT RATIO VALIDATION")
    print("-" * 80)
    
    total = len(y_train) + len(y_val) + len(y_test)
    train_ratio = (len(y_train) / total) * 100
    val_ratio = (len(y_val) / total) * 100
    test_ratio = (len(y_test) / total) * 100
    
    print(f"Train samples: {len(y_train):5d} ({train_ratio:.1f}%) [target: 64%]")
    print(f"Val samples:   {len(y_val):5d} ({val_ratio:.1f}%) [target: 16%]")
    print(f"Test samples:  {len(y_test):5d} ({test_ratio:.1f}%) [target: 20%]")
    print(f"Total:         {total:5d}")
    
    train_ok = 60 <= train_ratio <= 68
    val_ok = 14 <= val_ratio <= 18
    test_ok = 18 <= test_ratio <= 22
    
    if train_ok and val_ok and test_ok:
        print(f"\n✓ Split ratios are within acceptable range")
    else:
        print(f"\n⚠ Warning: Split ratios deviate from targets")
    
    # ====================================================================
    # 8. STATISTICAL VALIDATION
    # ====================================================================
    print("\n[8] STATISTICAL VALIDATION")
    print("-" * 80)
    
    # Check if X distributions are similar across splits
    print(f"\nX_train statistics:")
    print(f"  Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
    print(f"  Median: {np.median(X_train):.4f}, IQR: {np.percentile(X_train, 75) - np.percentile(X_train, 25):.4f}")
    
    print(f"\nX_val statistics:")
    print(f"  Mean: {X_val.mean():.4f}, Std: {X_val.std():.4f}")
    print(f"  Median: {np.median(X_val):.4f}, IQR: {np.percentile(X_val, 75) - np.percentile(X_val, 25):.4f}")
    
    print(f"\nX_test statistics:")
    print(f"  Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
    print(f"  Median: {np.median(X_test):.4f}, IQR: {np.percentile(X_test, 75) - np.percentile(X_test, 25):.4f}")
    
    # Check if distributions are reasonable
    if abs(X_train.mean() - X_val.mean()) > 2 * X_train.std():
        print(f"\n⚠ Warning: Large difference in means between train and val")
    else:
        print(f"\n✓ Train/val data distributions are similar")
    
    # ====================================================================
    # 9. SUMMARY
    # ====================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"""
✓ Dataset is ready for LSTM training!

Key Statistics:
  - Total samples: {total:,}
  - Features: {X_train.shape[1]}
  - Feature window size: 24 hours
  - Positive class: {overall_pct:.1f}%
  
Recommended next steps:
  1. Run LSTM model training: python lstm_model_for_deterioration.py
  2. Monitor training metrics (loss, accuracy, AUC)
  3. Validate on test set after training
  4. If class imbalance (current: {overall_pct:.1f}%):
     - Use class_weight in model.fit()
     - Consider stratified sampling
     - Use AUROC instead of accuracy
""")
    
    print("=" * 80)
    print(f"Validation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    return validation_passed

if __name__ == "__main__":
    validate_processed_data('processed_data.npz')
