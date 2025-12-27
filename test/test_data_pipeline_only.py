#!/usr/bin/env python3
"""
LIGHTWEIGHT TEST: Data loading and pipeline validation (no model loading)
Verifies the database connectivity and data reshaping logic
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import json
import os

print(f"\n{'='*70}")
print("DATA PIPELINE VALIDATION TEST (WITHOUT MODEL LOADING)")
print(f"{'='*70}\n")

# ============================================================================
# DATABASE UTILITIES (FROM MAIN SCRIPT)
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
        """Extract vital signs for an admission"""
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
        """Extract discharge notes for an admission"""
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
    
    def get_all_admissions_with_data(self, limit=10):
        """Find multiple admissions with BOTH vital signs AND discharge notes"""
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
            
            if len(results) >= limit:
                break
        
        return results
    
    def close(self):
        """Close database connections"""
        self.mimic_iv_conn.close()
        self.notes_conn.close()


def reshape_vitals_to_lstm_format(vitals_df, target_hours=24):
    """Reshape vital signs DataFrame into (24, 6) numpy array for LSTM"""
    vital_itemids = [220045, 220051, 220052, 220210, 220277, 223761]
    
    if vitals_df is None or len(vitals_df) == 0:
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
    """Create summary statistics from vital signs array"""
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
                "std": float(np.std(valid_values)),
                "valid_hours": int(len(valid_values))
            }
        else:
            summary[name] = {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
                "valid_hours": 0
            }
    
    return summary


# ============================================================================
# TEST PIPELINE
# ============================================================================

def test_data_loading(hadm_id, data_loader):
    """Test data loading and reshaping"""
    
    print(f"\nTesting Admission: {hadm_id}")
    print("-" * 70)
    
    # Step 1: Load vitals
    print("[VITALS] Loading vital signs...")
    vitals_df = data_loader.get_vital_signs(hadm_id)
    
    if vitals_df is None:
        print("[ERROR] No vitals found")
        return None
    
    print(f"  Loaded {len(vitals_df)} vital measurements")
    print(f"  Date range: {vitals_df['charttime'].min()} to {vitals_df['charttime'].max()}")
    print(f"  Itemids: {sorted(vitals_df['itemid'].unique())}")
    
    # Step 2: Reshape to LSTM format
    print("\n[RESHAPE] Converting to LSTM format...")
    vital_array = reshape_vitals_to_lstm_format(vitals_df)
    print(f"  Shape: {vital_array.shape}")
    print(f"  Non-zero values: {np.count_nonzero(vital_array)}")
    
    # Step 3: Summarize
    print("\n[SUMMARY] Computing vital signs statistics...")
    vital_summary = summarize_vital_signs(vital_array)
    for name, stats in vital_summary.items():
        if stats['valid_hours'] > 0:
            print(f"  {name:20s}: mean={stats['mean']:7.1f}, range=[{stats['min']:7.1f}, {stats['max']:7.1f}], hours={stats['valid_hours']}")
    
    # Step 4: Load notes
    print("\n[NOTES] Loading clinical notes...")
    discharge_notes = data_loader.get_discharge_notes(hadm_id)
    
    if discharge_notes is None or len(discharge_notes) == 0:
        print("[WARN] No discharge notes found")
        note_text = ""
        note_length = 0
    else:
        note_text = discharge_notes.iloc[0]['text']
        note_length = len(str(note_text)) if note_text else 0
        print(f"  Loaded {len(discharge_notes)} note(s)")
        print(f"  Note length: {note_length} characters")
        print(f"  Preview: {str(note_text)[:100]}...")
    
    # Return test result
    return {
        'hadm_id': hadm_id,
        'vitals_shape': vital_array.shape,
        'vitals_count': len(vitals_df),
        'vital_summary': vital_summary,
        'note_length': note_length,
        'success': True
    }


if __name__ == "__main__":
    # Initialize data loader
    print("[INIT] Connecting to databases...")
    data_loader = MIMICDataLoader('mimic_iv.db', 'mimic_notes_complete_records.db')
    
    # Find test admissions
    print("[DISCOVERY] Finding admissions with vitals and notes...")
    admissions = data_loader.get_all_admissions_with_data(limit=3)
    print(f"[OK] Found {len(admissions)} admissions\n")
    
    # Test each admission
    results = []
    for i, hadm_id in enumerate(admissions, 1):
        print(f"\n[TEST {i}/{len(admissions)}]")
        result = test_data_loading(hadm_id, data_loader)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{status} | Admission {r['hadm_id']}: {r['vitals_shape']} vitals, {r['note_length']} char notes")
    
    # Save results
    summary_file = "data_pipeline_test_results.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_admissions_tested": len(results),
            "all_passed": all(r['success'] for r in results),
            "results": results
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to {summary_file}")
    
    data_loader.close()
    print(f"\n{'='*70}\n")
