# Database Join Guide & Vital Signs Extraction

## Overview
This guide explains how to join `mimic_iv.db` and `mimic_notes_complete_records.db` using `hadm_id` and extract the 6 critical vitals.

---

## Part 1: Database Schema Summary

### MIMIC-IV (mimic_iv.db) - Key Tables

#### Table: `icustays`
- **Purpose**: ICU stay records for each admission
- **Key Columns**: 
  - `subject_id`: Patient ID
  - `hadm_id`: Hospital admission ID (PRIMARY JOIN KEY)
  - `stay_id`: ICU stay ID (links to chartevents)
  - `intime`: When patient entered ICU
  - `outtime`: When patient left ICU
- **Rows**: 140

#### Table: `chartevents`
- **Purpose**: All charted vital signs and measurements during ICU stay
- **Key Columns**:
  - `subject_id`: Patient ID
  - `hadm_id`: Hospital admission ID
  - `stay_id`: ICU stay ID (foreign key to icustays)
  - `charttime`: When measurement was recorded (TEXT: YYYY-MM-DD HH:MM:SS)
  - `itemid`: Item/measurement ID (identifies what vital this is)
  - `valuenum`: Numerical value of measurement
  - `valueuom`: Unit of measurement (bpm, mmHg, %, etc.)
- **Rows**: 668,862 (very large)

#### Table: `d_items`
- **Purpose**: Dictionary/mapping of itemid codes to human-readable labels
- **Key Columns**:
  - `itemid`: Unique item ID
  - `label`: Human-readable name (e.g., "Heart Rate", "Systolic BP")
  - `abbreviation`: Short name
  - `unitname`: Unit of measurement
  - `category`: Category (Vitals, Labs, etc.)
- **Rows**: 4,014

#### Table: `admissions`
- **Purpose**: Hospital admission records
- **Key Columns**:
  - `subject_id`: Patient ID
  - `hadm_id`: Hospital admission ID (PRIMARY JOIN KEY)
  - `admittime`: When admitted to hospital
  - `dischtime`: When discharged from hospital
- **Rows**: 275

---

### MIMIC Notes (mimic_notes_complete_records.db) - Key Tables

#### Table: `discharge`
- **Purpose**: Discharge summaries for each hospital admission
- **Key Columns**:
  - `note_id`: Unique note ID
  - `subject_id`: Patient ID
  - `hadm_id`: Hospital admission ID (PRIMARY JOIN KEY)
  - `charttime`: When note was written
  - `text`: Full text content of discharge summary
- **Rows**: 216
- **Index**: `idx_discharge_hadm` on hadm_id (optimized for joins)

#### Table: `radiology`
- **Purpose**: Radiology reports (X-rays, CT scans, etc.)
- **Key Columns**:
  - `note_id`: Unique report ID
  - `subject_id`: Patient ID
  - `hadm_id`: Hospital admission ID (PRIMARY JOIN KEY, can be NULL)
  - `charttime`: When study was performed
  - `text`: Full text content of radiology report
- **Rows**: 1,403
- **Index**: `idx_radiology_hadm` on hadm_id (optimized for joins)

---

## Part 2: The 6 Critical Vitals (itemids)

Based on MIMIC-IV standards, the 6 vitals are identified by these `itemid` values:

| Vital Sign | MIMIC-IV itemid | Unit | Normal Range |
|------------|-----------------|------|--------------|
| **Heart Rate (HR)** | 220045 | bpm | 60-100 |
| **Systolic BP (SBP)** | 220051 | mmHg | 90-140 |
| **Diastolic BP (DBP)** | 220052 | mmHg | 60-90 |
| **Respiratory Rate (RR)** | 220210 | breaths/min | 12-20 |
| **Oxygen Saturation (SpO2)** | 220277 | % | 95-100 |
| **Temperature (Temp)** | 223761 | °C | 36.5-37.5 |

---

## Part 3: SQL Joins

### Join Pattern

```
mimic_iv.db Tables:
├── icustays (hadm_id, stay_id)
│   └── chartevents (stay_id, itemid)
│       └── d_items (itemid → label)
└── admissions (hadm_id)

↓ JOIN ON ↓

mimic_notes_complete_records.db Tables:
├── discharge (hadm_id)
└── radiology (hadm_id)
```

### Query 1: Get Vital Signs for a Specific Admission

```sql
-- Get all 6 vitals for one admission, last 24 hours before discharge
SELECT 
    ce.hadm_id,
    ce.charttime,
    di.label AS vital_name,
    ce.valuenum,
    ce.valueuom,
    ce.itemid
FROM mimic_iv.chartevents ce
INNER JOIN mimic_iv.d_items di ON ce.itemid = di.itemid
WHERE ce.hadm_id = 20044587  -- Specific admission ID
  AND ce.itemid IN (220045, 220051, 220052, 220210, 220277, 223761)  -- The 6 vitals
ORDER BY ce.charttime DESC
LIMIT 100;  -- Last 100 measurements
```

### Query 2: Get Vitals Paired with Discharge Notes (Cross-Database Join)

```sql
-- Attach both databases and get vitals + discharge summary
ATTACH DATABASE 'mimic_notes_complete_records.db' AS notes_db;

SELECT 
    ce.hadm_id,
    ce.charttime,
    di.label AS vital_name,
    ce.valuenum,
    ce.valueuom,
    d.charttime AS discharge_time,
    d.text AS discharge_summary
FROM chartevents ce
INNER JOIN d_items di ON ce.itemid = di.itemid
INNER JOIN notes_db.discharge d ON ce.hadm_id = d.hadm_id
WHERE ce.itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
  AND ce.charttime >= d.charttime - INTERVAL '24 hours'  -- Vitals within 24 hours before discharge
  AND d.hadm_id = 20044587  -- Specific admission
ORDER BY ce.charttime DESC;
```

### Query 3: Get 24-Hour Vital Signs Window (For LSTM Model Input)

```sql
-- Extract 24-hour window of vitals for model input
-- Returns data suitable for reshaping to (24, 6) for LSTM

SELECT 
    ce.hadm_id,
    DATE(ce.charttime) AS date,
    HOUR(ce.charttime) AS hour,
    di.label AS vital_name,
    ce.valuenum,
    ce.itemid
FROM chartevents ce
INNER JOIN d_items di ON ce.itemid = di.itemid
INNER JOIN icustays icu ON ce.stay_id = icu.stay_id
WHERE ce.hadm_id = 20044587
  AND ce.itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
  AND ce.charttime >= datetime('2113-08-29 00:00:00')
  AND ce.charttime < datetime('2113-08-30 00:00:00')
ORDER BY ce.charttime, ce.itemid;
```

### Query 4: Find All Admissions with Both Vitals AND Notes

```sql
-- Identify admissions that have BOTH vital signs AND discharge notes
-- This is crucial for training/testing with real paired data

ATTACH DATABASE 'mimic_notes_complete_records.db' AS notes_db;

SELECT 
    icu.hadm_id,
    icu.subject_id,
    COUNT(DISTINCT CASE WHEN ce.itemid IN (220045, 220051, 220052, 220210, 220277, 223761) THEN ce.charttime END) AS vital_records,
    COUNT(DISTINCT d.note_id) AS discharge_notes,
    COUNT(DISTINCT r.note_id) AS radiology_reports
FROM icustays icu
LEFT JOIN chartevents ce ON icu.stay_id = ce.stay_id
LEFT JOIN notes_db.discharge d ON icu.hadm_id = d.hadm_id
LEFT JOIN notes_db.radiology r ON icu.hadm_id = r.hadm_id
GROUP BY icu.hadm_id, icu.subject_id
HAVING vital_records > 0 AND (discharge_notes > 0 OR radiology_reports > 0)
ORDER BY vital_records DESC;
```

---

## Part 4: Python Implementation

### Step 1: Connect to Databases

```python
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Connect to both databases
mimic_iv_conn = sqlite3.connect('mimic_iv.db')
notes_conn = sqlite3.connect('mimic_notes_complete_records.db')

mimic_iv_cursor = mimic_iv_conn.cursor()
notes_cursor = notes_conn.cursor()
```

### Step 2: Extract Vitals for a Specific Admission

```python
def get_vital_signs(hadm_id, hours=24, mimic_iv_conn=None):
    """
    Extract vital signs for an admission
    
    Args:
        hadm_id: Hospital admission ID
        hours: Number of hours to look back (default 24)
        mimic_iv_conn: SQLite connection to mimic_iv.db
    
    Returns:
        DataFrame with columns: charttime, vital_name, valuenum, valueuom
    """
    
    vital_itemids = [220045, 220051, 220052, 220210, 220277, 223761]
    itemid_names = {
        220045: 'HR',
        220051: 'SBP', 
        220052: 'DBP',
        220210: 'RR',
        220277: 'SpO2',
        223761: 'Temp'
    }
    
    query = f"""
    SELECT 
        ce.charttime,
        ce.itemid,
        ce.valuenum,
        ce.valueuom,
        di.label
    FROM chartevents ce
    INNER JOIN d_items di ON ce.itemid = di.itemid
    WHERE ce.hadm_id = {hadm_id}
      AND ce.itemid IN ({','.join(map(str, vital_itemids))})
    ORDER BY ce.charttime DESC
    LIMIT 1000
    """
    
    df = pd.read_sql_query(query, mimic_iv_conn)
    df['charttime'] = pd.to_datetime(df['charttime'])
    
    return df
```

### Step 3: Extract Clinical Notes for an Admission

```python
def get_clinical_notes(hadm_id, notes_conn=None, note_type='discharge'):
    """
    Extract clinical notes for an admission
    
    Args:
        hadm_id: Hospital admission ID
        notes_conn: SQLite connection to mimic_notes_complete_records.db
        note_type: 'discharge', 'radiology', or 'both'
    
    Returns:
        DataFrame with columns: note_id, note_type, charttime, text
    """
    
    if note_type == 'discharge' or note_type == 'both':
        query_discharge = """
        SELECT 
            note_id,
            'discharge' as note_type,
            charttime,
            text
        FROM discharge
        WHERE hadm_id = ?
        ORDER BY charttime DESC
        """
        df_discharge = pd.read_sql_query(query_discharge, notes_conn, params=(hadm_id,))
    
    if note_type == 'radiology' or note_type == 'both':
        query_radiology = """
        SELECT 
            note_id,
            'radiology' as note_type,
            charttime,
            text
        FROM radiology
        WHERE hadm_id = ?
        ORDER BY charttime DESC
        """
        df_radiology = pd.read_sql_query(query_radiology, notes_conn, params=(hadm_id,))
    
    if note_type == 'both':
        df = pd.concat([df_discharge, df_radiology], ignore_index=True)
    elif note_type == 'discharge':
        df = df_discharge
    else:
        df = df_radiology
    
    return df
```

### Step 4: Combine Vitals + Notes for a Single Admission

```python
def get_paired_vital_notes(hadm_id, mimic_iv_conn=None, notes_conn=None):
    """
    Get vitals and notes for an admission
    
    Args:
        hadm_id: Hospital admission ID
        mimic_iv_conn: Connection to mimic_iv.db
        notes_conn: Connection to mimic_notes_complete_records.db
    
    Returns:
        Dictionary with 'vitals' and 'notes' DataFrames
    """
    
    vitals_df = get_vital_signs(hadm_id, mimic_iv_conn=mimic_iv_conn)
    notes_df = get_clinical_notes(hadm_id, notes_conn=notes_conn, note_type='both')
    
    return {
        'hadm_id': hadm_id,
        'vitals': vitals_df,
        'notes': notes_df
    }
```

### Step 5: Reshape Vitals to (24, 6) for LSTM

```python
import numpy as np

def reshape_vitals_for_lstm(vitals_df, target_date=None):
    """
    Reshape vital signs into (24, 6) array for LSTM model
    
    Args:
        vitals_df: DataFrame from get_vital_signs()
        target_date: Specific date to extract (defaults to most recent complete day)
    
    Returns:
        numpy array of shape (24, 6) with vitals [HR, SBP, DBP, RR, SpO2, Temp]
    """
    
    vital_order = [220045, 220051, 220052, 220210, 220277, 223761]
    vital_names = ['HR', 'SBP', 'DBP', 'RR', 'SpO2', 'Temp']
    
    # Pivot to get one row per hour with vitals as columns
    vitals_df['date'] = vitals_df['charttime'].dt.date
    vitals_df['hour'] = vitals_df['charttime'].dt.hour
    
    # If no target_date specified, use most recent date with data
    if target_date is None:
        target_date = vitals_df['date'].max()
    
    # Filter to target date
    day_data = vitals_df[vitals_df['date'] == target_date].copy()
    
    # Pivot: rows=hours, columns=vitals
    pivot = day_data.pivot_table(
        index='hour',
        columns='itemid',
        values='valuenum',
        aggfunc='mean'  # If multiple values per hour, take mean
    )
    
    # Create 24x6 array
    vital_array = np.zeros((24, 6))
    
    for i, itemid in enumerate(vital_order):
        if itemid in pivot.columns:
            values = pivot[itemid].values
            # Fill available hours
            vital_array[:len(values), i] = values
    
    return vital_array
```

---

## Part 5: Complete Example Workflow

```python
import sqlite3
import pandas as pd
import numpy as np

# Setup
mimic_iv_path = 'mimic_iv.db'
notes_path = 'mimic_notes_complete_records.db'

mimic_iv_conn = sqlite3.connect(mimic_iv_path)
notes_conn = sqlite3.connect(notes_path)

# Example: Process a single admission
hadm_id = 20044587

# Step 1: Get vitals and notes
data = get_paired_vital_notes(hadm_id, mimic_iv_conn, notes_conn)

print(f"Admission {hadm_id}")
print(f"  - Vital records: {len(data['vitals'])}")
print(f"  - Clinical notes: {len(data['notes'])}")

# Step 2: Reshape vitals for LSTM
vital_array = reshape_vitals_for_lstm(data['vitals'])
print(f"  - LSTM input shape: {vital_array.shape}")  # Should be (24, 6)

# Step 3: Get discharge summary
discharge_notes = data['notes'][data['notes']['note_type'] == 'discharge']
if len(discharge_notes) > 0:
    summary = discharge_notes.iloc[0]['text']
    print(f"  - Discharge summary: {len(summary)} characters")

# Close connections
mimic_iv_conn.close()
notes_conn.close()
```

---

## Part 6: Key SQL Facts for Efficient Querying

### Indexes Available (Performance Optimizations)
- `discharge.hadm_id` - INDEXED ✓
- `radiology.hadm_id` - INDEXED ✓
- `chartevents` - No hadm_id index (large table), but stay_id is implicit

### Data Availability
- Total admissions with notes: 216 (discharge) + 1,403 (radiology)
- Total ICU stays: 140
- Total vital sign records: 668,862
- **Critical**: Not all admissions have vitals; not all with vitals have notes

### Best Practices
1. **Always filter by itemid first** before aggregating large chartevents table
2. **Use icustays.stay_id** to link chartevents instead of direct hadm_id when possible
3. **Check for data availability** using Query 4 before processing
4. **Handle missing values** - vitals may have gaps, especially at night
5. **Use transactions** when doing bulk reads for consistency

---

## Summary: The Join Pattern

```
REQUEST: "Get vitals and notes for hadm_id = 20044587"

┌─────────────────────────────────────┬──────────────────────────────┐
│     MIMIC-IV (mimic_iv.db)          │   MIMIC Notes (notes_db)     │
├─────────────────────────────────────┼──────────────────────────────┤
│                                     │                              │
│ chartevents.hadm_id ────────────────┼──→ discharge.hadm_id         │
│ (668K vital measurements)           │    (216 discharge summaries) │
│                                     │                              │
│ itemid ∈ {220045, 220051, 220052,   │ OR radiology.hadm_id         │
│           220210, 220277, 223761}   │    (1,403 reports)           │
│ = 6 vital signs                     │                              │
│                                     │                              │
└─────────────────────────────────────┴──────────────────────────────┘

RESULT: vitals + clinical notes paired by admission ID
```
