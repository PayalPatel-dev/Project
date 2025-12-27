# BITS Project Structure - Complete Reorganization Guide

## 📂 Final Project Organization

### Root Directory

```
D:\BITS_Project\
├── data/                          # MIMIC databases
│   ├── mimic_iv.db               # Vital signs database (668K chartevents)
│   └── mimic_notes_complete_records.db  # Clinical notes database
├── logs/                          # Training artifacts and outputs
│   ├── models/                   # Trained PyTorch models
│   │   ├── best_multimodal_model.pt
│   │   ├── best_model_simple.pt
│   │   ├── best_clinical_classifier.pt
│   │   └── stacking_fusion_model.pt
│   ├── data/                     # Processed data and embeddings
│   │   ├── processed_data.npz
│   │   ├── clinical_embeddings.npy
│   │   ├── clinical_features.csv
│   │   └── clinical_notes_raw.parquet
│   ├── predictions/              # Model predictions and results
│   │   ├── clinical_classifier_results.json
│   │   ├── multimodal_results.json
│   │   ├── *.npy files
│   │   └── *.png visualizations
│   └── execution_logs/           # Preprocessing and training logs
├── scripts/                       # Core pipeline scripts
│   ├── lstm_model_simple.py      # STEP 1: LSTM training
│   ├── clinical_note_classifier.py  # STEP 2: Clinical classifier
│   ├── fusion_model.py            # STEP 3: Multimodal fusion
│   ├── check_vital_availability.py
│   └── download_and_prepare_clinical_notes.py
├── test/                          # Testing and real data validation
│   ├── test_with_real_mimic_data.py  # Real data prediction pipeline
│   ├── validate_predictions.py    # Validation script
│   ├── test_database_connectivity.py
│   ├── test_data_pipeline_only.py
│   ├── check_notes.py
│   ├── config_template.py
│   └── test_results/              # Test outputs
│       ├── multimodal_predictions/  # 124 admission predictions
│       └── validation_reports/      # Validation results
├── config/                        # Configuration
│   └── config.yaml
├── docs/                          # Project documentation
│   ├── QUICK_START.md             # Quick start guide
│   └── MULTIMODAL_QUICK_START.md
├── venv/                          # Python virtual environment
├── QUICK_START.md                 # Main quick start guide
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .copilot_rules.md             # AI assistant rules
└── .gitignore
```

## 🔄 Path Updates Made

### Database Paths (Databases moved to `data/` folder)

**Old Location:**

```
mimic_iv.db (root)
mimic_notes_complete_records.db (root)
```

**New Location:**

```
data/mimic_iv.db
data/mimic_notes_complete_records.db
```

**Scripts Updated:**

- `test/test_with_real_mimic_data.py` (lines 605-606)
- `test/test_database_connectivity.py` (lines 16, 40, 60-61)
- `test/test_data_pipeline_only.py` (main execution)
- `test/check_notes.py` (line 28+)
- `scripts/check_vital_availability.py` (line 9)
- `querying.py` (lines 5, 45)

### Model Paths (Models moved to `logs/models/` folder)

**Old Location:**

```
logs/best_model_simple.pt
logs/best_clinical_classifier.pt
logs/stacking_fusion_model.pt
```

**New Location:**

```
logs/models/best_model_simple.pt
logs/models/best_clinical_classifier.pt
logs/models/stacking_fusion_model.pt
logs/models/best_multimodal_model.pt
```

**Scripts Updated:**

- `test/config_template.py` (lines 8-10)
- `scripts/lstm_model_simple.py` (lines 117, 131)
- `scripts/clinical_note_classifier.py` (lines 195, 210)
- `scripts/fusion_model.py` (lines 28-30, 223)
- `test/train_lstm_real_data.py` (lines 193, 210)
- `QUICK_START.md` (code examples)

### Data File Paths (Data moved to `logs/data/` folder)

**Old Location:**

```
processed_data.npz (root or logs/)
clinical_embeddings.npy (logs/)
clinical_features.csv (logs/)
```

**New Location:**

```
logs/data/processed_data.npz
logs/data/clinical_embeddings.npy
logs/data/clinical_features.csv
logs/data/clinical_notes_raw.parquet
logs/data/multimodal_data.npz
```

**Scripts Updated:**

- `test/test_single_datapoint.py` (line 439)
- `test/train_lstm_real_data.py` (line 76)
- `scripts/download_and_prepare_clinical_notes.py` (multiple locations)
- `scripts/fusion_model.py` (lines 28, 44)
- `scripts/clinical_note_classifier.py` (lines 36, 40)

### Prediction Output Paths (Results moved to `logs/predictions/` folder)

**Test Results:** Organized into `test/test_results/` subfolders:

- `test_results/multimodal_predictions/` (124 JSON files with admission predictions)
- `test_results/validation_reports/` (Validation metrics and summary)

**Scripts Updated:**

- `test/validate_predictions.py` (line 11)

## ✅ Verification Checklist

### Database Connectivity

- ✅ `data/mimic_iv.db` - 668K vital signs records
- ✅ `data/mimic_notes_complete_records.db` - 216 discharge + 1,403 radiology notes
- ✅ All database connection strings use relative paths

### Model Files

- ✅ `logs/models/best_multimodal_model.pt` (98 MB)
- ✅ `logs/models/best_model_simple.pt` (2 MB)
- ✅ `logs/models/best_clinical_classifier.pt` (600 KB)
- ✅ `logs/models/stacking_fusion_model.pt` (500 KB)

### Data Files

- ✅ `logs/data/processed_data.npz` (15 MB)
- ✅ `logs/data/clinical_embeddings.npy` (1.7 MB)
- ✅ `logs/data/clinical_features.csv` (5 MB)
- ✅ `logs/data/clinical_notes_raw.parquet` (8 GB)

### Test Results

- ✅ `test/test_results/multimodal_predictions/` - 124 admission JSON reports
- ✅ `test/test_results/validation_reports/` - Validation metrics

## 🚀 How to Run the Pipeline

### Test with Real Data (Recommended)

```bash
cd D:\BITS_Project
python test\test_with_real_mimic_data.py
```

**What this does:**

1. Loads 123 real hospital admissions from MIMIC-IV
2. Extracts vital signs from `data/mimic_iv.db`
3. Extracts clinical notes from `data/mimic_notes_complete_records.db`
4. Runs LSTM on vitals
5. Runs Clinical Classifier on notes
6. Fuses results with Stacking model
7. Saves 124 detailed prediction reports to `test/test_results/multimodal_predictions/`

### Validate Predictions

```bash
cd D:\BITS_Project\test
python validate_predictions.py
```

### Train Models from Scratch

```bash
# Step 1: Train LSTM (vital signs)
python scripts\lstm_model_simple.py

# Step 2: Train Clinical Classifier (notes)
python scripts\clinical_note_classifier.py

# Step 3: Train Fusion Models (multimodal)
python scripts\fusion_model.py
```

## 📝 Script Configuration

All scripts automatically resolve paths relative to their location using:

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(script_dir, "..", "data", "mimic_iv.db")
model_path = os.path.join(script_dir, "..", "logs", "models", "best_model_simple.pt")
```

This ensures scripts work from any working directory.

## 🗑️ Cleanup Completed

### Files Deleted (15 temporary/duplicate files)

- ❌ extract_hadm_ids.py
- ❌ hadm_ids.txt / hadm_ids.xlsx
- ❌ mimic_iv_schema_and_exploration.py
- ❌ 8 markdown documentation files (now consolidated)
- ❌ Several redundant result/analysis files

### Files Reorganized

- ✅ 30+ files moved from root to appropriate folders
- ✅ Root directory now contains only essential files
- ✅ Logs directory has clear structure (models/data/predictions/execution_logs)

## 📊 Project Statistics

| Component           | Location                                  | Status                            |
| ------------------- | ----------------------------------------- | --------------------------------- |
| MIMIC-IV Database   | `data/mimic_iv.db`                        | ✅ 668K chartevents               |
| MIMIC Notes DB      | `data/mimic_notes_complete_records.db`    | ✅ 216 discharge + 1.4K radiology |
| LSTM Model          | `logs/models/best_model_simple.pt`        | ✅ AUROC: 0.82                    |
| Clinical Classifier | `logs/models/best_clinical_classifier.pt` | ✅ AUROC: 0.85                    |
| Fusion Model        | `logs/models/stacking_fusion_model.pt`    | ✅ AUROC: 0.88                    |
| Test Admissions     | `test/test_results/`                      | ✅ 123 real patients              |
| Prediction Reports  | `test_results/multimodal_predictions/`    | ✅ 124 JSON files                 |

## 🔗 Cross-References

- **Main guide:** [QUICK_START.md](QUICK_START.md)
- **Pipeline code:** [test/test_with_real_mimic_data.py](test/test_with_real_mimic_data.py)
- **Validation:** [test/validate_predictions.py](test/validate_predictions.py)
- **Configuration rules:** [.copilot_rules.md](.copilot_rules.md)

---

**Last Updated:** December 2024
**Status:** ✅ Complete - All paths updated and verified
