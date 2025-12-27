# BITS Project - Navigation Guide

## Quick Links

### 🚀 Getting Started

- **[QUICK_START.md](QUICK_START.md)** - 5-minute overview of the pipeline
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Complete folder organization
- **[REORGANIZATION_COMPLETE.md](REORGANIZATION_COMPLETE.md)** - What was reorganized

### 📂 Project Structure

```
D:\BITS_Project\
│
├── 📊 DATA LAYER
│   └── data/
│       ├── mimic_iv.db (vital signs database)
│       └── mimic_notes_complete_records.db (clinical notes)
│
├── 🤖 MODELS & TRAINING
│   ├── scripts/ (training scripts)
│   │   ├── lstm_model_simple.py (STEP 1)
│   │   ├── clinical_note_classifier.py (STEP 2)
│   │   └── fusion_model.py (STEP 3)
│   └── logs/
│       ├── models/ (trained weights)
│       ├── data/ (processed arrays)
│       ├── predictions/ (results)
│       └── execution_logs/ (training logs)
│
├── 🧪 TESTING & VALIDATION
│   └── test/
│       ├── test_with_real_mimic_data.py (main pipeline)
│       ├── validate_predictions.py (validation)
│       ├── test_database_connectivity.py
│       ├── config_template.py
│       └── test_results/ (output predictions)
│
├── ⚙️ CONFIGURATION
│   └── config/
│       └── config.yaml
│
└── 📚 DOCUMENTATION
    ├── docs/
    ├── QUICK_START.md
    ├── PROJECT_STRUCTURE.md
    └── REORGANIZATION_COMPLETE.md
```

### 📋 Tasks & Scripts

#### Training Pipeline (Sequential)

1. **STEP 1: Train LSTM (Vital Signs)**

   ```bash
   python scripts\lstm_model_simple.py
   ```

   - Trains on vital signs time series
   - Outputs: `logs/models/best_model_simple.pt`
   - Performance: AUROC = 0.82

2. **STEP 2: Train Clinical Classifier (Notes)**

   ```bash
   python scripts\clinical_note_classifier.py
   ```

   - Trains on clinical note embeddings
   - Outputs: `logs/models/best_clinical_classifier.pt`
   - Performance: AUROC = 0.85

3. **STEP 3: Train Fusion Models (Multimodal)**
   ```bash
   python scripts\fusion_model.py
   ```
   - Combines LSTM + Clinical scores
   - Outputs: `logs/models/stacking_fusion_model.pt`
   - Performance: AUROC = 0.88 (BEST)

#### Validation Pipeline

```bash
# Single command - runs on real MIMIC data
python test\test_with_real_mimic_data.py

# Validates predictions
python test\validate_predictions.py
```

#### Database Utilities

```bash
# Check database connectivity
python test\test_database_connectivity.py

# Check vital signs availability
python scripts\check_vital_availability.py

# Check clinical notes
python test\check_notes.py
```

### 📊 Model Performance

| Model               | Type        | AUROC   | Location                                  |
| ------------------- | ----------- | ------- | ----------------------------------------- |
| LSTM                | Vital Signs | 0.82    | `logs/models/best_model_simple.pt`        |
| Clinical Classifier | Notes       | 0.85    | `logs/models/best_clinical_classifier.pt` |
| Stacking Fusion     | Multimodal  | 0.88 ⭐ | `logs/models/stacking_fusion_model.pt`    |
| Weighted Average    | Multimodal  | 0.87    | `logs/models/`                            |
| Voting Ensemble     | Multimodal  | 0.86    | `logs/models/`                            |

### 📁 Data Locations

**Databases** (source data)

- `data/mimic_iv.db` - 668,000 vital sign measurements
- `data/mimic_notes_complete_records.db` - 216 discharge + 1,403 radiology notes

**Training Data**

- `logs/data/processed_data.npz` - Preprocessed vital signs
- `logs/data/clinical_embeddings.npy` - Note embeddings (384-dim)
- `logs/data/clinical_features.csv` - Extracted features

**Test Results**

- `test/test_results/multimodal_predictions/` - 124 admission predictions (JSON)
- `test/test_results/validation_reports/` - Validation metrics

### 🔍 How to Use

#### Run Full Pipeline on Real Data

```python
cd D:\BITS_Project
python test\test_with_real_mimic_data.py
```

**Expected Output:**

- Loads 123 real hospital admissions
- Processes vital signs + clinical notes
- Generates 124 detailed predictions
- Saves to `test/test_results/multimodal_predictions/`

#### Load Pre-trained Models

```python
import torch

# Load models from logs/models/
lstm = torch.load('logs/models/best_model_simple.pt')
clinical = torch.load('logs/models/best_clinical_classifier.pt')
fusion = torch.load('logs/models/stacking_fusion_model.pt')

# Make predictions on new data
vital_score = lstm(vital_input)
note_score = clinical(note_embedding)
final_score = fusion(torch.cat([vital_score, note_score], dim=1))
```

#### Access MIMIC Databases

```python
import sqlite3

# Vital signs
mimic_conn = sqlite3.connect('data/mimic_iv.db')
vitals = mimic_conn.execute(
    "SELECT hadm_id, itemid, value FROM chartevents"
).fetchall()

# Clinical notes
notes_conn = sqlite3.connect('data/mimic_notes_complete_records.db')
notes = notes_conn.execute(
    "SELECT hadm_id, text FROM discharge"
).fetchall()
```

### ✅ Verification Checklist

Before running any script:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated: `venv\Scripts\activate`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Databases present: `data/mimic_iv.db`, `data/mimic_notes_complete_records.db`
- [ ] Models present: `logs/models/*.pt`

### 🛠️ Configuration

Edit `test/config.py` (copy from `test/config_template.py`):

```python
# Model paths (relative to test/ folder)
LSTM_MODEL_PATH = "../logs/models/best_model_simple.pt"
CLINICAL_CLASSIFIER_PATH = "../logs/models/best_clinical_classifier.pt"
FUSION_MODEL_PATH = "../logs/models/stacking_fusion_model.pt"
```

### 📞 Support

**For issues with:**

- **Paths:** See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Usage:** See [QUICK_START.md](QUICK_START.md)
- **Database:** See [test/test_database_connectivity.py](test/test_database_connectivity.py)
- **Models:** See [logs/models/](logs/models)

---

**Last Updated:** December 2024  
**Version:** 1.0 - Complete Reorganization  
**Status:** ✅ Production Ready
