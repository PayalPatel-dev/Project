#!/usr/bin/env python
"""
BITS Project - Pipeline Execution Manager
Runs training phases with user input control
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import shutil

# Model paths
TRAINED_MODELS = {
    "lstm": "logs/models/best_model_simple.pt",
    "clinical_classifier": "logs/models/best_clinical_classifier.pt",
    "stacking_fusion": "logs/models/stacking_fusion_model.pt",
    "working_lstm": "logs/models/working_lstm_model.pt",
}

# Phase definitions
PHASES = {
    "1": {
        "name": "Data Preparation",
        "scripts": [
            "scripts/download_and_prepare_clinical_notes.py",
            "scripts/check_vital_availability.py",
        ],
        "time": "10-15 min"
    },
    "2": {
        "name": "Model Training",
        "scripts": [
            "scripts/lstm_model_simple.py",
            "scripts/clinical_note_classifier.py",
            "scripts/fusion_model.py",
        ],
        "time": "30-45 min",
        "can_skip": True
    },
    "3": {
        "name": "Testing & Validation",
        "scripts": [
            "test/test_with_real_mimic_data.py",
            "test/validate_predictions.py",
        ],
        "time": "5-10 min"
    },
    "4": {
        "name": "Load Pre-trained Models (Skip Training)",
        "scripts": [
            "scripts/load_pretrained_models.py",
        ],
        "time": "<1 min"
    },
}

def check_trained_models():
    """Check if all required trained models exist"""
    missing = []
    for model_name, model_path in TRAINED_MODELS.items():
        if not Path(model_path).exists():
            missing.append(model_path)
    
    if missing:
        print("\n⚠ Missing trained models:")
        for m in missing:
            print(f"   • {m}")
        print("\nYou can either:")
        print("  • Run Phase 2 to train models")
        print("  • Ensure models are in the paths above\n")
        return False
    return True

def print_banner():
    """Display project banner"""
    print("\n" + "="*70)
    print("  BITS PROJECT - TRAINING PIPELINE MANAGER")
    print("="*70 + "\n")

def print_menu():
    """Display menu options"""
    print("Available Phases:\n")
    for key, phase in PHASES.items():
        print(f"  {key}. {phase['name']} (~{phase['time']})")
        for script in phase['scripts']:
            print(f"     • {script}")
        print()
    print("  0. Exit")
    print("  A. Run ALL phases (1→2→3)")
    print()

def validate_databases():
    """Check if required databases exist"""
    required = [
        "logs/data/mimic_iv.db",
        "logs/data/mimic_notes_complete_records.db"
    ]
    missing = [f for f in required if not Path(f).exists()]
    
    if missing:
        print("❌ ERROR: Missing required databases:")
        for f in missing:
            print(f"   • {f}")
        return False
    return True

def run_script(script_path):
    """Execute a Python script with virtual environment activated"""
    # Use venv Python directly (automatically uses venv)
    python_exe = Path("venv/Scripts/python.exe")
    
    if not python_exe.exists():
        print(f"❌ Virtual environment not found at {python_exe}")
        print("   Run: python -m venv venv")
        return False
    
    print(f"\n▶ Running: {script_path}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [str(python_exe), script_path],
            check=False,
            timeout=3600,  # 1 hour timeout
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✓ {script_path} completed successfully\n")
            return True
        else:
            print(f"✗ {script_path} failed (exit code: {result.returncode})\n")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {script_path} timed out (>1 hour)\n")
        return False
    except Exception as e:
        print(f"✗ Error running {script_path}: {e}\n")
        return False

def run_phase(phase_key):
    """Run all scripts in a phase"""
    if phase_key not in PHASES:
        print("❌ Invalid phase")
        return False
    
    phase = PHASES[phase_key]
    
    # Skip Phase 2 if models already exist
    if phase_key == "2":
        if check_trained_models():
            print(f"\n{'='*70}")
            print(f"  PHASE {phase_key}: {phase['name']}")
            print(f"{'='*70}")
            print("✓ Pre-trained models found!")
            print("  Skipping training phase (models already trained).")
            print(f"{'='*70}\n")
            return True
    
    print(f"\n{'='*70}")
    print(f"  PHASE {phase_key}: {phase['name']}")
    print(f"  Estimated time: {phase['time']}")
    print(f"{'='*70}\n")
    
    failed = []
    for idx, script in enumerate(phase['scripts'], 1):
        print(f"[{idx}/{len(phase['scripts'])}]", end=" ")
        
        if not run_script(script):
            failed.append(script)
        
        if idx < len(phase['scripts']):
            print("Waiting 2 seconds before next script...\n")
            time.sleep(2)
    
    # Summary
    print(f"{'='*70}")
    if failed:
        print(f"⚠ Phase {phase_key} completed with {len(failed)} error(s):")
        for f in failed:
            print(f"  • {f}")
        return False
    else:
        print(f"✓ Phase {phase_key} completed successfully!")
    print(f"{'='*70}\n")
    return True

def run_all_phases():
    """Run all phases sequentially"""
    print("\n" + "="*70)
    print("  RUNNING ALL PHASES (1→2→3)")
    print(f"  Total estimated time: ~1-1.5 hours")
    print("="*70)
    
    start_time = time.time()
    results = {}
    
    for phase_key in ["1", "2", "3"]:
        if not run_phase(phase_key):
            print(f"\n⚠ Stopping at Phase {phase_key}")
            results[phase_key] = "FAILED"
            break
        results[phase_key] = "SUCCESS"
    
    # Final summary
    elapsed = (time.time() - start_time) / 60
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"Elapsed time: {elapsed:.1f} minutes\n")
    
    for phase_key in ["1", "2", "3"]:
        status = results.get(phase_key, "SKIPPED")
        symbol = "✓" if status == "SUCCESS" else "✗" if status == "FAILED" else "⊘"
        print(f"  {symbol} Phase {phase_key}: {status}")
    
    print("="*70 + "\n")

def main():
    """Main menu loop"""
    print_banner()
    
    # Check databases
    if not validate_databases():
        print("\n❌ Cannot proceed. Ensure databases exist in data/ folder.\n")
        return
    
    print("✓ Databases validated\n")
    
    while True:
        print_menu()
        choice = input("Select option: ").strip().upper()
        
        if choice == "0":
            print("\nExiting...\n")
            break
        elif choice == "A":
            run_all_phases()
        elif choice in PHASES:
            run_phase(choice)
            input("Press Enter to continue...")
        else:
            print("❌ Invalid option\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⊘ Interrupted by user\n")
        sys.exit(0)
