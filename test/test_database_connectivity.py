#!/usr/bin/env python3
"""
Quick validation: Test database connectivity and data availability
"""

import sqlite3
import pandas as pd

print("\n" + "="*70)
print("DATABASE CONNECTIVITY & DATA VALIDATION")
print("="*70 + "\n")

# Test 1: mimic_iv.db
print("[1] Testing mimic_iv.db...")
try:
    mimic_iv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mimic_iv.db")
    conn = sqlite3.connect(mimic_iv_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM chartevents')
    chartevents_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM icustays')
    icustays_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM d_items')
    ditems_count = cursor.fetchone()[0]
    
    print(f"    ✓ chartevents: {chartevents_count:,} rows")
    print(f"    ✓ icustays: {icustays_count:,} rows")
    print(f"    ✓ d_items: {ditems_count:,} rows")
    
    conn.close()
    print("[OK] mimic_iv.db is accessible\n")
except Exception as e:
    print(f"[ERROR] {str(e)}\n")

# Test 2: mimic_notes_complete_records.db
print("[2] Testing mimic_notes_complete_records.db...")
try:
    notes_path = os.path.join(os.path.dirname(__file__), "..", "data", "mimic_notes_complete_records.db")
    conn = sqlite3.connect(notes_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM discharge')
    discharge_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM radiology')
    radiology_count = cursor.fetchone()[0]
    
    print(f"    ✓ discharge: {discharge_count:,} rows")
    print(f"    ✓ radiology: {radiology_count:,} rows")
    
    conn.close()
    print("[OK] mimic_notes_complete_records.db is accessible\n")
except Exception as e:
    print(f"[ERROR] {str(e)}\n")

# Test 3: Find a sample admission with data
print("[3] Finding sample admission with vitals + notes...")
try:
    notes_conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "mimic_notes_complete_records.db"))
    mimic_conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "..", "data", "mimic_iv.db"))
    
    # Get discharge hadm_ids
    discharge_hadm = pd.read_sql_query('SELECT DISTINCT hadm_id FROM discharge', notes_conn)
    
    # Check which ones have vitals
    results = []
    for hadm_id in discharge_hadm['hadm_id'].head(10):
        vital_query = f"""
        SELECT COUNT(DISTINCT charttime) as vital_count
        FROM chartevents
        WHERE hadm_id = {hadm_id}
          AND itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
        """
        vital_result = pd.read_sql_query(vital_query, mimic_conn)
        vital_count = vital_result['vital_count'].values[0]
        
        if vital_count > 0:
            results.append({'hadm_id': hadm_id, 'vital_count': vital_count})
    
    print(f"    Found {len(results)} admissions with vitals + notes (showing first 5):")
    for r in results[:5]:
        print(f"      hadm_id={int(r['hadm_id'])}, vitals={int(r['vital_count'])}")
    
    mimic_conn.close()
    notes_conn.close()
    print("[OK] Sample data is available\n")
except Exception as e:
    print(f"[ERROR] {str(e)}\n")
    import traceback
    traceback.print_exc()

print("="*70)
print("DATABASE VALIDATION COMPLETE")
print("="*70 + "\n")
