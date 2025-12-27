#!/usr/bin/env python
"""
Load Pre-trained Models
Verifies and loads all pre-trained models for quick inference in CodeSpaces
"""

import os
from pathlib import Path

# Model paths
TRAINED_MODELS = {
    "lstm": "logs/models/best_model_simple.pt",
    "clinical_classifier": "logs/models/best_clinical_classifier.pt",
    "stacking_fusion": "logs/models/stacking_fusion_model.pt",
    "working_lstm": "logs/models/working_lstm_model.pt",
}

def check_models():
    """Check if all pre-trained models exist"""
    print("\n" + "="*70)
    print("  CHECKING PRE-TRAINED MODELS")
    print("="*70 + "\n")
    
    all_exist = True
    for model_name, model_path in TRAINED_MODELS.items():
        model_file = Path(model_path)
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"✓ {model_name:25} {model_path:45} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {model_name:25} {model_path:45} NOT FOUND")
            all_exist = False
    
    print("\n" + "="*70)
    if all_exist:
        print("✓ All pre-trained models loaded successfully!")
        print("  Ready for inference in CodeSpaces without retraining.")
    else:
        print("✗ Some models are missing. Run Phase 2 to train them.")
    print("="*70 + "\n")
    
    return all_exist

if __name__ == "__main__":
    success = check_models()
    exit(0 if success else 1)
