#!/usr/bin/env python3
"""
MULTIMODAL PREDICTION PIPELINE WITH REAL MIMIC-IV DATA
Vital Signs (from mimic_iv.db) + Clinical Notes (from mimic_notes_complete_records.db)
→ LSTM + Clinical Classifier → Fusion Model → Risk Score
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Import config and utilities
import config


# ============================================================================
# DATABASE UTILITIES
# ============================================================================

class MIMICDataLoader:
    """Unified data loader for MIMIC-IV vitals and clinical notes"""
    
    def __init__(self, mimic_iv_path='mimic_iv.db', notes_path='mimic_notes_complete_records.db'):
        """Initialize database connections"""
        self.mimic_iv_conn = sqlite3.connect(mimic_iv_path)
        self.notes_conn = sqlite3.connect(notes_path)
        
        # Vital sign itemids (fixed in MIMIC-IV)
        self.vital_itemids = [220045, 220051, 220052, 220210, 220277, 223761]
        self.vital_names = ['HR', 'SBP', 'DBP', 'RR', 'SpO2', 'Temp']
        
    def get_vital_signs(self, hadm_id):
        """
        Extract vital signs for an admission
        
        Args:
            hadm_id: Hospital admission ID
        
        Returns:
            DataFrame with columns: charttime, itemid, vital_name, valuenum, valueuom
        """
        query = """
        SELECT 
            ce.charttime,
            ce.itemid,
            di.label AS vital_name,
            ce.valuenum,
            ce.valueuom
        FROM chartevents ce
        INNER JOIN d_items di ON ce.itemid = di.itemid
        WHERE ce.hadm_id = ?
          AND ce.itemid IN (?, ?, ?, ?, ?, ?)
        ORDER BY ce.charttime DESC
        LIMIT 1440
        """
        
        df = pd.read_sql_query(
            query,
            self.mimic_iv_conn,
            params=(hadm_id,) + tuple(self.vital_itemids)
        )
        
        if len(df) == 0:
            return None
        
        df['charttime'] = pd.to_datetime(df['charttime'])
        return df.sort_values('charttime', ascending=True)
    
    def get_discharge_notes(self, hadm_id):
        """
        Extract discharge notes for an admission
        
        Args:
            hadm_id: Hospital admission ID
        
        Returns:
            DataFrame with columns: note_id, charttime, text
        """
        query = """
        SELECT 
            note_id,
            charttime,
            text
        FROM discharge
        WHERE hadm_id = ?
        ORDER BY charttime DESC
        """
        
        df = pd.read_sql_query(query, self.notes_conn, params=(hadm_id,))
        
        if len(df) == 0:
            return None
        
        df['charttime'] = pd.to_datetime(df['charttime'])
        return df
    
    def get_radiology_notes(self, hadm_id):
        """
        Extract radiology notes for an admission
        
        Args:
            hadm_id: Hospital admission ID
        
        Returns:
            DataFrame with columns: note_id, charttime, text
        """
        query = """
        SELECT 
            note_id,
            charttime,
            text
        FROM radiology
        WHERE hadm_id = ?
        ORDER BY charttime DESC
        """
        
        df = pd.read_sql_query(query, self.notes_conn, params=(hadm_id,))
        
        if len(df) == 0:
            return None
        
        df['charttime'] = pd.to_datetime(df['charttime'])
        return df
    
    def get_admission_with_data(self):
        """
        Find an admission with BOTH vital signs AND discharge notes
        
        Returns:
            hadm_id of admission with complete data
        """
        query = """
        SELECT 
            d.hadm_id,
            COUNT(DISTINCT ce.charttime) as vital_count,
            COUNT(DISTINCT d.note_id) as note_count
        FROM discharge d
        LEFT JOIN chartevents ce ON d.hadm_id = ce.hadm_id
            AND ce.itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
        GROUP BY d.hadm_id
        HAVING vital_count > 0 AND note_count > 0
        LIMIT 1
        """
        
        result = pd.read_sql_query(query, self.mimic_iv_conn)
        if len(result) == 0:
            raise Exception("No admission found with both vitals and notes")
        
        return int(result.iloc[0]['hadm_id'])
    
    def get_all_admissions_with_data(self, limit=10):
        """
        Find multiple admissions with BOTH vital signs AND discharge notes
        
        Returns:
            List of hadm_ids
        """
        # Get discharge hadm_ids from notes database
        discharge_query = "SELECT DISTINCT hadm_id FROM discharge"
        discharge_hadm = pd.read_sql_query(discharge_query, self.notes_conn)
        
        # Check which ones have vitals
        results = []
        for hadm_id in discharge_hadm['hadm_id'].values:
            vital_query = f"""
            SELECT COUNT(DISTINCT charttime) as vital_count
            FROM chartevents
            WHERE hadm_id = {int(hadm_id)}
              AND itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
            """
            vital_result = pd.read_sql_query(vital_query, self.mimic_iv_conn)
            vital_count = vital_result['vital_count'].values[0]
            
            if vital_count > 0:
                results.append(int(hadm_id))
            
            if limit > 0 and len(results) >= limit:
                break
        
        return results
    
    def close(self):
        """Close database connections"""
        self.mimic_iv_conn.close()
        self.notes_conn.close()


# ============================================================================
# VITAL SIGNS PROCESSING
# ============================================================================

def reshape_vitals_to_lstm_format(vitals_df, target_hours=24):
    """
    Reshape vital signs DataFrame into (24, 6) numpy array for LSTM
    
    Args:
        vitals_df: DataFrame from get_vital_signs()
        target_hours: Number of hours to include (default 24)
    
    Returns:
        numpy array of shape (24, 6) with vitals [HR, SBP, DBP, RR, SpO2, Temp]
    """
    vital_itemids = [220045, 220051, 220052, 220210, 220277, 223761]
    
    if vitals_df is None or len(vitals_df) == 0:
        # Return zeros if no data
        return np.zeros((target_hours, 6))
    
    # Extract most recent 24-hour window
    vitals_df = vitals_df.copy()
    vitals_df['date'] = vitals_df['charttime'].dt.date
    vitals_df['hour'] = vitals_df['charttime'].dt.hour
    
    # Get most recent date with data
    target_date = vitals_df['date'].max()
    day_data = vitals_df[vitals_df['date'] == target_date].copy()
    
    # Create pivot table: hours x vitals
    pivot = day_data.pivot_table(
        index='hour',
        columns='itemid',
        values='valuenum',
        aggfunc='mean'  # If multiple values per hour, take mean
    )
    
    # Create output array (24 hours x 6 vitals)
    vital_array = np.zeros((target_hours, 6))
    
    # Fill in available data
    for i, itemid in enumerate(vital_itemids):
        if itemid in pivot.columns:
            values = pivot[itemid].values
            vital_array[:len(values), i] = values
    
    return vital_array


def summarize_vital_signs(vital_array):
    """
    Create summary statistics from vital signs array
    
    Args:
        vital_array: numpy array of shape (24, 6)
    
    Returns:
        Dictionary with summary statistics
    """
    vital_names = ["Heart_Rate", "Systolic_BP", "Diastolic_BP", "Respiratory_Rate", "SpO2", "Temperature"]
    summary = {}
    
    for idx, name in enumerate(vital_names):
        values = vital_array[:, idx]
        # Filter out zeros (missing values)
        valid_values = values[values > 0]
        
        if len(valid_values) > 0:
            summary[name] = {
                "mean": float(np.mean(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
                "std": float(np.std(valid_values))
            }
        else:
            summary[name] = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            }
    
    return summary


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class WorkingLSTM(nn.Module):
    """LSTM for vital signs classification"""
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
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


class ClinicalNoteClassifier(nn.Module):
    """Clinical classifier for text embeddings"""
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
    """Meta-learner fusion model combining LSTM and Clinical Classifier predictions"""
    def __init__(self, input_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
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
# PREDICTION PIPELINE
# ============================================================================

def predict_with_real_data(hadm_id, data_loader, device=None):
    """
    Complete prediction pipeline using real MIMIC data
    
    Args:
        hadm_id: Hospital admission ID
        data_loader: MIMICDataLoader instance
        device: torch device (cpu/cuda)
    
    Returns:
        Dictionary with complete prediction report
    """
    if device is None:
        device = torch.device(config.DEVICE)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*70}")
    print(f"PREDICTION PIPELINE - REAL MIMIC DATA")
    print(f"{'='*70}")
    print(f"Admission ID: {hadm_id}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*70}\n")
    
    # -------- STEP 1: LOAD VITAL SIGNS --------
    print("[VITALS] Loading vital signs from mimic_iv.db...")
    vitals_df = data_loader.get_vital_signs(hadm_id)
    
    if vitals_df is None or len(vitals_df) == 0:
        print("[ERROR] No vital signs found for this admission")
        return None
    
    vital_array = reshape_vitals_to_lstm_format(vitals_df)
    vital_summary = summarize_vital_signs(vital_array)
    
    print(f"[VITALS] [OK] Loaded {len(vitals_df)} vital measurements")
    print(f"         Reshaped to LSTM format: {vital_array.shape}\n")
    
    # -------- STEP 2: LOAD CLINICAL NOTES --------
    print("[NOTES] Loading discharge summary from mimic_notes_complete_records.db...")
    discharge_notes_df = data_loader.get_discharge_notes(hadm_id)
    
    if discharge_notes_df is None or len(discharge_notes_df) == 0:
        print("[WARN] No discharge summary found, trying radiology notes...")
        discharge_notes_df = data_loader.get_radiology_notes(hadm_id)
    
    if discharge_notes_df is None or len(discharge_notes_df) == 0:
        print("[ERROR] No clinical notes found for this admission")
        return None
    
    # Use most recent note
    clinical_note = discharge_notes_df.iloc[0]['text']
    
    # Clean up note (remove null values)
    if clinical_note is None or (isinstance(clinical_note, float)):
        clinical_note = "No clinical note available"
    
    print(f"[NOTES] [OK] Loaded {len(discharge_notes_df)} clinical notes")
    print(f"        Note length: {len(str(clinical_note))} characters\n")
    
    # -------- STEP 3: GET EMBEDDINGS --------
    print("[EMBEDDINGS] Converting clinical note to embeddings...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(str(clinical_note), convert_to_numpy=True)
    
    print(f"[EMBEDDINGS] [OK] Shape: {embeddings.shape}\n")
    
    # -------- STEP 4: LOAD MODELS --------
    print("[MODELS] Loading trained models...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    lstm_path = os.path.join(script_dir, config.LSTM_MODEL_PATH)
    clinical_path = os.path.join(script_dir, config.CLINICAL_CLASSIFIER_PATH)
    fusion_path = os.path.join(script_dir, config.FUSION_MODEL_PATH)
    
    # Load LSTM
    lstm_model = WorkingLSTM(input_size=6, hidden_size=64, num_layers=2)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=False))
    lstm_model.to(device)
    lstm_model.eval()
    
    # Load Clinical Classifier
    clinical_model = ClinicalNoteClassifier()
    clinical_model.load_state_dict(torch.load(clinical_path, map_location=device, weights_only=False))
    clinical_model.to(device)
    clinical_model.eval()
    
    # Load Fusion Model
    fusion_model = StackingFusionModel()
    fusion_model.load_state_dict(torch.load(fusion_path, map_location=device, weights_only=False))
    fusion_model.to(device)
    fusion_model.eval()
    
    print("[MODELS] [OK] All models loaded\n")
    
    # -------- STEP 5: LSTM PREDICTION --------
    print("[LSTM] Running LSTM on vital signs...")
    vital_tensor = torch.tensor(vital_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        lstm_logits = lstm_model(vital_tensor).squeeze()
        vital_score = torch.sigmoid(lstm_logits).cpu().item()
        
        # Handle NaN case (all-zero vital data)
        if np.isnan(vital_score) or np.isinf(vital_score):
            vital_score = 0.5  # Default to neutral score
    
    print(f"[LSTM] [OK] Vital signs risk score: {vital_score:.4f}\n")
    
    # -------- STEP 6: CLINICAL CLASSIFIER PREDICTION --------
    print("[CLINICAL] Running clinical classifier on note embeddings...")
    embedding_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        clinical_logits = clinical_model(embedding_tensor).squeeze()
        clinical_score = torch.sigmoid(clinical_logits).cpu().item()
    
    print(f"[CLINICAL] [OK] Clinical note risk score: {clinical_score:.4f}\n")
    
    # -------- STEP 7: FUSION PREDICTION --------
    print("[FUSION] Running stacking fusion model...")
    fusion_input = torch.tensor([vital_score, clinical_score], dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fusion_logits = fusion_model(fusion_input).squeeze()
        final_risk_score = torch.sigmoid(fusion_logits).cpu().item()
        
        # Handle NaN case
        if np.isnan(final_risk_score) or np.isinf(final_risk_score):
            final_risk_score = clinical_score  # Fallback to clinical score
    
    print(f"[FUSION] [OK] Final risk score: {final_risk_score:.4f}\n")
    
    # -------- STEP 8: RISK CATEGORIZATION --------
    if final_risk_score > config.RISK_THRESHOLD_HIGH:
        risk_category = "HIGH RISK"
        decision = "ALERT"
    elif final_risk_score > config.RISK_THRESHOLD_LOW:
        risk_category = "MEDIUM RISK"
        decision = "MONITOR"
    else:
        risk_category = "LOW RISK"
        decision = "NO_ALERT"
    
    # -------- STEP 9: GENERATE REPORT --------
    report = {
        "admission_id": hadm_id,
        "timestamp": timestamp,
        "data_source": "MIMIC-IV (real hospital data)",
        "vital_signs_summary": vital_summary,
        "clinical_note_preview": str(clinical_note)[:500],
        "predictions": {
            "lstm_vital_score": float(vital_score),
            "clinical_classifier_score": float(clinical_score),
            "fusion_final_score": float(final_risk_score),
            "risk_category": risk_category
        },
        "decision": decision
    }
    
    return report


def print_report(report):
    """Pretty print the prediction report"""
    if report is None:
        print("[ERROR] Report is None - prediction failed")
        return
    
    print(f"\n{'='*70}")
    print("PREDICTION REPORT")
    print(f"{'='*70}")
    
    print(f"\nAdmission ID: {report['admission_id']}")
    print(f"Data Source: {report['data_source']}")
    print(f"Timestamp: {report['timestamp']}")
    
    print("\n--- VITAL SIGNS SUMMARY (24 hours) ---")
    for feature, stats in report['vital_signs_summary'].items():
        if stats['mean'] > 0:
            print(f"{feature:20s}: {stats['mean']:7.1f} (range: {stats['min']:7.1f}-{stats['max']:7.1f})")
    
    print("\n--- CLINICAL NOTE PREVIEW ---")
    preview = report['clinical_note_preview']
    print(preview[:300] + ("..." if len(preview) > 300 else ""))
    
    print("\n--- MODEL PREDICTIONS ---")
    print(f"LSTM (Vital Signs):        {report['predictions']['lstm_vital_score']:.4f}")
    print(f"Clinical Classifier:       {report['predictions']['clinical_classifier_score']:.4f}")
    print(f"Fusion Model (Final):      {report['predictions']['fusion_final_score']:.4f}")
    
    print("\n--- CLINICAL DECISION ---")
    score = report['predictions']['fusion_final_score']
    category = report['predictions']['risk_category']
    decision = report['decision']
    
    print(f"Risk Score:     {score:.4f}")
    print(f"Risk Category:  {category}")
    print(f"Decision:       {decision}")
    
    if decision == "ALERT":
        print("\n[!] [ALERT] HIGH RISK - Immediate clinical attention required")
    elif decision == "MONITOR":
        print("\n[*] [MONITOR] MEDIUM RISK - Continue monitoring, prepare interventions")
    else:
        print("\n[OK] [OK] LOW RISK - Routine care continues")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("MULTIMODAL PREDICTION PIPELINE WITH REAL MIMIC DATA")
    print(f"{'='*70}\n")
    
    # Initialize data loader
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mimic_iv_path = os.path.join(script_dir, "..", "mimic_iv.db")
    notes_path = os.path.join(script_dir, "..", "mimic_notes_complete_records.db")
    
    try:
        data_loader = MIMICDataLoader(mimic_iv_path, notes_path)
        
        # Find admissions with both vitals and notes
        print("[DISCOVERY] Finding admissions with vitals and clinical notes...")
        admissions = data_loader.get_all_admissions_with_data(limit=0)  # 0 = all
        print(f"[OK] Found {len(admissions)} admissions with complete data\n")
        
        # Process each admission
        reports = []
        for i, hadm_id in enumerate(admissions, 1):
            print(f"\n[TEST {i}/{len(admissions)}] Processing Admission {hadm_id}")
            print("-" * 70)
            
            report = predict_with_real_data(hadm_id, data_loader)
            
            if report is not None:
                print_report(report)
                reports.append(report)
                
                # Save individual report to test_results subfolder
                results_dir = os.path.join(script_dir, "test_results")
                os.makedirs(results_dir, exist_ok=True)
                output_file = os.path.join(results_dir, f"admission_{hadm_id}_report.json")
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"[OK] Saved: {output_file}\n")
        
        # Save summary to test_results subfolder
        if reports:
            results_dir = os.path.join(script_dir, "test_results")
            os.makedirs(results_dir, exist_ok=True)
            summary_file = os.path.join(results_dir, "real_data_predictions_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_admissions": len(reports),
                    "data_source": "MIMIC-IV (real hospital data)",
                    "admissions": reports
                }, f, indent=2)
            print(f"[OK] Saved summary: {summary_file}")
            
            # Print comparison
            print(f"\n{'='*70}")
            print("SUMMARY - ALL ADMISSIONS")
            print(f"{'='*70}")
            for r in reports:
                score = r['predictions']['fusion_final_score']
                category = r['predictions']['risk_category']
                print(f"Admission {r['admission_id']}: {score:.4f} ({category})")
            print(f"{'='*70}\n")
        
        data_loader.close()
        print("[OK] PIPELINE COMPLETE - All results saved\n")
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
