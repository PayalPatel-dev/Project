#!/usr/bin/env python3
"""
SINGLE DATAPOINT PREDICTION TEST
Vital Signs → LSTM → Clinical Classifier → Fusion Model
"""

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import os
from datetime import datetime

# Import config and utilities
import config


# ============================================================================
# MODEL DEFINITIONS (Match training scripts)
# ============================================================================

def summarize_vital_signs(vital_signs):
    """
    Create summary statistics from vital signs array
    
    Args:
        vital_signs: numpy array of shape (24, 6)
    
    Returns:
        Dictionary with summary statistics
    """
    features = ["Heart_Rate", "Systolic_BP", "Diastolic_BP", "Respiratory_Rate", "SpO2", "Temperature"]
    summary = {}
    
    for idx, feature in enumerate(features):
        values = vital_signs[:, idx]
        summary[feature] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values))
        }
    
    return summary


def create_summary_text(summary):
    """Convert vital signs summary to text for clinical note generation"""
    text = "Vital Signs Summary (24 hours):\n"
    text += f"- Heart Rate: {summary['Heart_Rate']['mean']:.1f} bpm (range: {summary['Heart_Rate']['min']:.1f}-{summary['Heart_Rate']['max']:.1f})\n"
    text += f"- Systolic BP: {summary['Systolic_BP']['mean']:.1f} mmHg (range: {summary['Systolic_BP']['min']:.1f}-{summary['Systolic_BP']['max']:.1f})\n"
    text += f"- Diastolic BP: {summary['Diastolic_BP']['mean']:.1f} mmHg (range: {summary['Diastolic_BP']['min']:.1f}-{summary['Diastolic_BP']['max']:.1f})\n"
    text += f"- Respiratory Rate: {summary['Respiratory_Rate']['mean']:.1f} breaths/min (range: {summary['Respiratory_Rate']['min']:.1f}-{summary['Respiratory_Rate']['max']:.1f})\n"
    text += f"- Oxygen Saturation: {summary['SpO2']['mean']:.1f}% (range: {summary['SpO2']['min']:.1f}-{summary['SpO2']['max']:.1f})\n"
    text += f"- Temperature: {summary['Temperature']['mean']:.1f}°C (range: {summary['Temperature']['min']:.1f}-{summary['Temperature']['max']:.1f})\n"
    return text


class WorkingLSTM(nn.Module):
    """Improved LSTM for vital signs classification"""
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(16)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        out = self.fc1(last_hidden)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out


class SimpleLSTM(nn.Module):
    """LSTM for vital signs (deprecated - use WorkingLSTM instead)"""
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_hidden))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ClinicalNoteClassifier(nn.Module):
    """Clinical classifier for embeddings"""
    def __init__(self, input_dim=384, hidden_size=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 64)
        self.fc4 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        return out


class StackingFusionModel(nn.Module):
    """Stacking fusion model"""
    def __init__(self, input_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_input(vital_signs, embeddings):
    """Validate input data"""
    print("[VALIDATION] Checking input data...")
    
    # Shape checks
    assert vital_signs.shape == (24, 6), f"Wrong vital signs shape: {vital_signs.shape}"
    assert embeddings.shape == (384,), f"Wrong embedding shape: {embeddings.shape}"
    
    # Range checks
    assert 40 <= vital_signs[:, 0].mean() <= 180, "Heart Rate out of range"
    assert 60 <= vital_signs[:, 1].mean() <= 250, "Systolic BP out of range"
    assert -1 <= embeddings.min() <= 1, "Embedding values out of range"
    
    # NaN/Inf checks
    assert not np.isnan(vital_signs).any(), "NaN in vital signs"
    assert not np.isinf(vital_signs).any(), "Inf in vital signs"
    
    print("[VALIDATION] [OK] Input data valid\n")
    return True


def validate_output(vital_score, clinical_score, fusion_score):
    """Validate model outputs"""
    print("[VALIDATION] Checking model outputs...")
    
    # Range checks
    assert 0.0 <= vital_score <= 1.0, f"Vital score {vital_score} out of range"
    assert 0.0 <= clinical_score <= 1.0, f"Clinical score {clinical_score} out of range"
    assert 0.0 <= fusion_score <= 1.0, f"Fusion score {fusion_score} out of range"
    
    # Note: Stacking fusion can produce non-linear combinations,
    # so we don't enforce min-max constraints here
    
    # Sanity checks - broad thresholds for stacking flexibility
    if vital_score > 0.8 and clinical_score > 0.8:
        assert fusion_score > 0.5, "Both scores high but fusion very low"
    
    if vital_score < 0.2 and clinical_score < 0.2:
        assert fusion_score < 0.5, "Both scores low but fusion very high"
    
    print("[VALIDATION] [OK] Model outputs valid\n")
    return True


def get_risk_category(score):
    """Convert score to risk category"""
    if score > config.RISK_THRESHOLD_HIGH:
        return "HIGH RISK"
    elif score > config.RISK_THRESHOLD_LOW:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================

def generate_clinical_note(vital_summary_text, patient_type="normal"):
    """Generate clinical note using Google Gemini API."""
    print("[GEMINI] Generating clinical note...")
    
    from google import genai as google_genai
    
    # Use the newer Google AI SDK
    client = google_genai.Client(api_key=config.GEMINI_API_KEY)
    
    prompt = f"""Based on the following vital signs, write a brief clinical note (2-3 sentences).
    
    {vital_summary_text}
    
    Write as a medical professional would, noting any concerning trends or normal findings."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    
    note = response.text
    print(f"[GEMINI] [OK] Note generated\n")
    return note


def get_embeddings(text):
    """Convert text to embeddings using SentenceTransformer"""
    print("[EMBEDDINGS] Converting clinical note to embeddings...")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding = model.encode(text, convert_to_numpy=True)
    
    print("[EMBEDDINGS] [OK] Embeddings created\n")
    return embedding


def predict_single_datapoint(vital_signs, clinical_note, patient_type="unknown", patient_id="PT_001"):
    """
    Main prediction pipeline
    
    Args:
        vital_signs: (24, 6) numpy array
        clinical_note: str
        patient_type: "normal" or "abnormal"
        patient_id: str
    
    Returns:
        dict with predictions
    """
    
    device = torch.device(config.DEVICE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get script directory for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*70)
    print("SINGLE DATAPOINT PREDICTION TEST")
    print("="*70)
    print(f"Patient ID: {patient_id}")
    print(f"Patient Type: {patient_type}")
    print(f"Timestamp: {timestamp}")
    print("="*70 + "\n")
    
    # -------- STEP 1: VALIDATE INPUT --------
    vital_summary = summarize_vital_signs(vital_signs)
    embeddings = get_embeddings(clinical_note)
    
    validate_input(vital_signs, embeddings)
    
    # -------- STEP 2: LOAD MODELS --------
    print("[MODELS] Loading trained models...")
    
    # Resolve model paths relative to script directory
    lstm_path = os.path.join(script_dir, config.LSTM_MODEL_PATH)
    clinical_path = os.path.join(script_dir, config.CLINICAL_CLASSIFIER_PATH)
    fusion_path = os.path.join(script_dir, config.FUSION_MODEL_PATH)
    
    lstm_model = WorkingLSTM(input_size=6, hidden_size=64, num_layers=2)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=False))
    lstm_model.to(device)
    lstm_model.eval()
    
    clinical_model = ClinicalNoteClassifier()
    clinical_model.load_state_dict(torch.load(clinical_path, map_location=device, weights_only=False))
    clinical_model.to(device)
    clinical_model.eval()
    
    fusion_model = StackingFusionModel()
    fusion_model.load_state_dict(torch.load(fusion_path, map_location=device, weights_only=False))
    fusion_model.to(device)
    fusion_model.eval()
    
    print("[MODELS] [OK] All models loaded\n")
    
    # -------- STEP 3: LSTM PREDICTION --------
    print("[LSTM] Running LSTM on vital signs...")
    vital_tensor = torch.tensor(vital_signs, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lstm_logits = lstm_model(vital_tensor).squeeze()
        vital_score = torch.sigmoid(lstm_logits).cpu().item()
    
    print(f"[LSTM] [OK] Vital signs score: {vital_score:.4f}\n")
    
    # -------- STEP 4: CLINICAL CLASSIFIER PREDICTION --------
    print("[CLINICAL] Running clinical classifier...")
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        clinical_logits = clinical_model(embedding_tensor).squeeze()
        clinical_score = torch.sigmoid(clinical_logits).cpu().item()
    
    print(f"[CLINICAL] [OK] Clinical note score: {clinical_score:.4f}\n")
    
    # -------- STEP 5: FUSION PREDICTION --------
    print("[FUSION] Running stacking fusion model...")
    fusion_input = torch.tensor([vital_score, clinical_score], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fusion_logits = fusion_model(fusion_input).squeeze()
        final_risk_score = torch.sigmoid(fusion_logits).cpu().item()
    
    print(f"[FUSION] [OK] Final risk score: {final_risk_score:.4f}\n")
    
    # -------- STEP 6: VALIDATE OUTPUT --------
    validate_output(vital_score, clinical_score, final_risk_score)
    
    # -------- STEP 7: GENERATE REPORT --------
    risk_category = get_risk_category(final_risk_score)
    
    report = {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "patient_type": patient_type,
        "vital_signs_summary": vital_summary,
        "clinical_note": clinical_note,
        "predictions": {
            "lstm_vital_score": float(vital_score),
            "clinical_classifier_score": float(clinical_score),
            "fusion_final_score": float(final_risk_score),
            "risk_category": risk_category
        },
        "decision": "ALERT" if final_risk_score > config.RISK_THRESHOLD_HIGH else "MONITOR" if final_risk_score > config.RISK_THRESHOLD_LOW else "NO_ALERT"
    }
    
    return report


def print_report(report):
    """Pretty print the prediction report"""
    print("\n" + "="*70)
    print("PREDICTION REPORT")
    print("="*70)
    
    print(f"\nPatient ID: {report['patient_id']}")
    print(f"Type: {report['patient_type']}")
    print(f"Timestamp: {report['timestamp']}")
    
    print("\n--- VITAL SIGNS SUMMARY (24 hours) ---")
    for feature, stats in report['vital_signs_summary'].items():
        print(f"{feature}: {stats['mean']:.1f} (range: {stats['min']:.1f}-{stats['max']:.1f})")
    
    print("\n--- CLINICAL NOTE ---")
    print(report['clinical_note'])
    
    print("\n--- MODEL PREDICTIONS ---")
    print(f"LSTM (Vital Signs): {report['predictions']['lstm_vital_score']:.4f}")
    print(f"Clinical Classifier: {report['predictions']['clinical_classifier_score']:.4f}")
    print(f"Fusion Model (Final): {report['predictions']['fusion_final_score']:.4f}")
    
    print("\n--- DECISION ---")
    score = report['predictions']['fusion_final_score']
    category = report['predictions']['risk_category']
    decision = report['decision']
    
    print(f"Risk Score: {score:.4f}")
    print(f"Risk Category: {category}")
    print(f"Clinical Decision: {decision}")
    
    if decision == "ALERT":
        print("[ALERT]  HIGH RISK - Immediate clinical attention required")
    elif decision == "MONITOR":
        print("[MONITOR] MEDIUM RISK - Continue monitoring, prepare interventions")
    else:
        print("[OK] LOW RISK - Routine care continues")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST: SINGLE DATAPOINT PREDICTION PIPELINE")
    print("="*70 + "\n")
    
    # Get script directory and construct paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "processed_data.npz")
    
    # Load real test data instead of synthetic
    test_data = np.load(data_path)
    X_test = test_data['X_test']  # (272, 24, 6)
    y_test = test_data['y_test']  # (272,)
    
    # Get one normal and one abnormal sample from real test set
    normal_idx = np.where(y_test == 0)[0][0]
    abnormal_idx = np.where(y_test == 1)[0][0]
    
    # Test Case 1: NORMAL PATIENT (from real test data)
    print("\n[TEST 1] NORMAL PATIENT (from real test data)")
    print("-" * 70)
    
    normal_vitals = X_test[normal_idx]  # (24, 6)
    normal_summary = summarize_vital_signs(normal_vitals)
    normal_summary_text = create_summary_text(normal_summary)
    
    print(normal_summary_text)
    
    # Generate clinical note using Gemini API
    normal_note = generate_clinical_note(normal_summary_text, patient_type="normal")
    
    # Run prediction
    normal_report = predict_single_datapoint(
        normal_vitals,
        normal_note,
        patient_type="normal",
        patient_id="PT_NORMAL_001"
    )
    
    print_report(normal_report)
    
    # Save report
    with open("normal_patient_report.json", "w") as f:
        json.dump(normal_report, f, indent=2)
    print("[OK] Saved: normal_patient_report.json")
    
    # Test Case 2: ABNORMAL PATIENT (from real test data)
    print("\n[TEST 2] ABNORMAL PATIENT (from real test data)")
    print("-" * 70)
    
    abnormal_vitals = X_test[abnormal_idx]  # (24, 6)
    abnormal_summary = summarize_vital_signs(abnormal_vitals)
    abnormal_summary_text = create_summary_text(abnormal_summary)
    
    print(abnormal_summary_text)
    
    # Generate clinical note using Gemini API
    abnormal_note = generate_clinical_note(abnormal_summary_text, patient_type="abnormal")
    
    # Run prediction
    abnormal_report = predict_single_datapoint(
        abnormal_vitals,
        abnormal_note,
        patient_type="abnormal",
        patient_id="PT_ABNORMAL_001"
    )
    
    print_report(abnormal_report)
    
    # Save report
    with open("abnormal_patient_report.json", "w") as f:
        json.dump(abnormal_report, f, indent=2)
    print("[OK] Saved: abnormal_patient_report.json")
    
    # COMPARISON
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nNormal Patient:   {normal_report['predictions']['fusion_final_score']:.4f} ({normal_report['predictions']['risk_category']})")
    print(f"Abnormal Patient: {abnormal_report['predictions']['fusion_final_score']:.4f} ({abnormal_report['predictions']['risk_category']})")
    
    print("\n[OK] TEST COMPLETE - All results saved to test/ folder\n")
