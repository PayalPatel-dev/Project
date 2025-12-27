# BITS Project - Multimodal Clinical Data Analysis

A comprehensive machine learning pipeline for analyzing MIMIC-IV clinical data using LSTM models and fusion techniques to predict clinical outcomes from structured vital signs and unstructured clinical notes.

## 📋 Overview

This project combines:
- **Structured Data**: Vital signs and clinical measurements from MIMIC-IV database
- **Unstructured Data**: Clinical notes processed with NLP techniques
- **Multiple Models**: LSTM, Clinical Note Classifier, and Stacking Fusion Model
- **Pre-trained Models**: Skip retraining in CodeSpaces—models are included in Git

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment
- Git (with models tracked via Git)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/BITS_Project.git
cd BITS_Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Databases

Ensure required databases exist in `data/` folder:
```bash
data/
├── mimic_iv.db                      # Structured vital signs
└── mimic_notes_complete_records.db  # Clinical notes
```

Download these from [MIMIC-IV](https://mimic.mit.edu/) official source.

### Run Pipeline

```bash
python run_pipeline.py
```

**Menu Options:**
- **Phase 1**: Data Preparation (10-15 min)
- **Phase 2**: Model Training (30-45 min) *[AUTO-SKIPPED if models exist]*
- **Phase 3**: Testing & Validation (5-10 min)
- **Phase 4**: Load Pre-trained Models (<1 min)
- **Option A**: Run ALL phases (1→2→3)

## 📁 Project Structure

```
BITS_Project/
├── config/                  # Configuration files
│   └── config.yaml
├── data/                    # Input databases
│   ├── mimic_iv.db
│   └── mimic_notes_complete_records.db
├── logs/                    # Outputs and checkpoints
│   ├── models/             # Pre-trained models (tracked in Git)
│   │   ├── best_model_simple.pt
│   │   ├── best_clinical_classifier.pt
│   │   ├── stacking_fusion_model.pt
│   │   └── working_lstm_model.pt
│   ├── predictions/        # Model predictions
│   ├── data/              # Processed datasets
│   └── execution_logs/    # Training logs
├── scripts/                 # Core pipeline scripts
│   ├── download_and_prepare_clinical_notes.py
│   ├── check_vital_availability.py
│   ├── lstm_model_simple.py
│   ├── clinical_note_classifier.py
│   ├── fusion_model.py
│   ├── load_pretrained_models.py
│   └── preprocessing_pipeline_for_your_data.py
├── test/                    # Testing & validation
│   ├── test_with_real_mimic_data.py
│   ├── validate_predictions.py
│   └── test_results/
├── reports/                 # Generated analysis reports
├── run_pipeline.py          # Main execution manager
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md               # This file
```

## 🎯 Pipeline Phases

### Phase 1: Data Preparation
- Downloads and prepares clinical notes from MIMIC-IV
- Checks vital sign availability
- Preprocesses structured data
- **Time**: 10-15 minutes

### Phase 2: Model Training
- Trains LSTM model on vital signs
- Trains Clinical Note Classifier
- Trains Fusion Model combining both modalities
- **Time**: 30-45 minutes
- **Note**: ⏭️ **AUTO-SKIPPED** if pre-trained models exist

### Phase 3: Testing & Validation
- Validates models on test data
- Generates predictions
- Creates performance reports
- **Time**: 5-10 minutes

### Phase 4: Load Pre-trained Models
- Verifies all trained models are available
- Shows model sizes and status
- Perfect for CodeSpaces environments
- **Time**: <1 minute

## 💾 Pre-trained Models

All trained models are stored in `logs/models/` and **tracked in Git**:

| Model | Path | Size |
|-------|------|------|
| LSTM | `logs/models/best_model_simple.pt` | ~5-10 MB |
| Clinical Classifier | `logs/models/best_clinical_classifier.pt` | ~5-10 MB |
| Stacking Fusion | `logs/models/stacking_fusion_model.pt` | ~3-5 MB |
| Working LSTM | `logs/models/working_lstm_model.pt` | ~5-10 MB |

### Why Models Are in Git?
- **CodeSpaces**: Automatically downloads trained models—no retraining needed
- **Faster Development**: Use pre-trained models immediately for inference
- **Reproducibility**: Exact same models across all environments

## 🐙 GitHub & CodeSpaces Usage

### Cloning in CodeSpaces

1. Open repo in CodeSpaces
2. Run `python run_pipeline.py`
3. Select **Phase 4** or **Phase 3** to test
4. Models load automatically—**no training needed** ✓

### Performance in CodeSpaces
- Phase 1 (Data Prep): 10-15 min
- Phase 2 (Training): **SKIPPED** (models included)
- Phase 3 (Testing): 5-10 min
- **Total Time**: ~20-25 min instead of 1+ hour

## 📊 Models & Architecture

### LSTM Model (`best_model_simple.pt`)
- Input: Vital signs sequences
- Output: Clinical outcome predictions
- Architecture: 2-layer LSTM with attention

### Clinical Note Classifier (`best_clinical_classifier.pt`)
- Input: Processed clinical notes (TF-IDF/embeddings)
- Output: Note-based outcome predictions
- Architecture: Dense neural network

### Fusion Model (`stacking_fusion_model.pt`)
- Combines predictions from LSTM + Clinical Classifier
- Meta-learner stacking approach
- Improved prediction accuracy

## 📦 Requirements

See `requirements.txt` for full dependency list. Key packages:
- `torch` - Deep learning
- `pandas` - Data manipulation
- `scikit-learn` - ML utilities
- `numpy` - Numerical computing
- `sqlalchemy` - Database queries

Install with:
```bash
pip install -r requirements.txt
```

## 📝 Configuration

Edit `config/config.yaml` to customize:
- Database paths
- Model hyperparameters
- Training settings
- Data preprocessing options

## 📊 Results & Reports

After running pipelines, check:
- **Predictions**: `logs/predictions/`
- **Performance Metrics**: `logs/predictions/*.json`
- **Analysis Reports**: `reports/`

## 🐛 Troubleshooting

### Missing Databases
```
ERROR: Missing required databases
```
**Solution**: Ensure `data/mimic_iv.db` and `data/mimic_notes_complete_records.db` exist

### Models Not Found in Phase 2
```
Missing trained models
```
**Solution**: Run Phase 2 to train, or ensure models exist in `logs/models/`

### Virtual Environment Issues
```bash
# Recreate venv
rm -r venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🔄 Workflow

```
Clone Repo
    ↓
Activate venv
    ↓
Prepare databases
    ↓
Run run_pipeline.py
    ↓
Choose Phase(s)
    ↓
(Phase 2 auto-skipped if models exist)
    ↓
View results in logs/predictions/
```

## 📖 Documentation

- [Dataset Overview](reports/1_DATASET_OVERVIEW.txt)
- [Data Preprocessing](reports/2_DATA_PREPROCESSING.txt)
- [LSTM Architecture](reports/4_LSTM_MODEL_ARCHITECTURE.txt)
- [Clinical Note Classifier](reports/5_CLINICAL_NOTE_CLASSIFIER.txt)
- [Fusion Strategy](reports/6_FUSION_MODEL_STRATEGY.txt)
- [Results & Performance](reports/8_RESULTS_AND_PERFORMANCE.txt)
- [Project Summary](reports/9_PROJECT_SUMMARY.txt)

## 👨‍💻 Development

### Adding New Models
1. Create training script in `scripts/`
2. Update PHASES in `run_pipeline.py`
3. Add model to `TRAINED_MODELS` dict
4. Models auto-tracked in Git

### Testing Changes
```bash
# Run single phase
python run_pipeline.py
# Select Phase 4 to verify models

# Or run specific test
python test/test_with_real_mimic_data.py
```

## 📄 License

[Your License Here]

## 👥 Authors

- BITS Project Team

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For issues, questions, or contributions, open a GitHub issue.

---

**Last Updated**: January 2026  
**Status**: ✓ Production Ready  
**Models Included**: ✓ All 4 models tracked in Git
