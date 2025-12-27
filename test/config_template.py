# Configuration Template
# Copy this file to config.py and fill in your actual API key

# Google Gemini API Configuration
GEMINI_API_KEY = "your-api-key-here"

# Model Paths
LSTM_MODEL_PATH = "../logs/models/best_model_simple.pt"
CLINICAL_CLASSIFIER_PATH = "../logs/models/best_clinical_classifier.pt"
FUSION_MODEL_PATH = "../logs/models/stacking_fusion_model.pt"

# Vital Signs Configuration
VITAL_SIGNS_FEATURES = ["Heart_Rate", "Systolic_BP", "Diastolic_BP", "Respiratory_Rate", "SpO2", "Temperature"]
VITAL_SIGNS_RANGES_NORMAL = {
    "Heart_Rate": (60, 100),
    "Systolic_BP": (100, 140),
    "Diastolic_BP": (60, 90),
    "Respiratory_Rate": (12, 20),
    "SpO2": (95, 100),
    "Temperature": (36.5, 37.5)
}

VITAL_SIGNS_RANGES_ABNORMAL = {
    "Heart_Rate": (110, 150),
    "Systolic_BP": (140, 180),
    "Diastolic_BP": (90, 110),
    "Respiratory_Rate": (24, 35),
    "SpO2": (85, 92),
    "Temperature": (38.5, 39.5)
}

# Prediction Thresholds
RISK_THRESHOLD_LOW = 0.3
RISK_THRESHOLD_HIGH = 0.7

# Device
DEVICE = "cpu"  # or "cuda" if GPU available
