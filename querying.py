import sqlite3
import pandas as pd
import os

# Connect to notes database
script_dir = os.path.dirname(os.path.abspath(__file__))
notes_db_path = os.path.join(script_dir, "data", "mimic_notes_complete_records.db")
notes_conn = sqlite3.connect(notes_db_path)

# Check discharge table structure
discharge_info = pd.read_sql_query(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='discharge'",
    notes_conn
)
print("DISCHARGE TABLE SCHEMA:")
print(discharge_info.iloc[0]['sql'])
print("\n")

# Count total rows
discharge_count = pd.read_sql_query("SELECT COUNT(*) as count FROM discharge", notes_conn)
print(f"Total discharge notes in DB: {discharge_count.iloc[0]['count']}")
print("\n")

# Check if hadm_id column exists and has data
hadm_check = pd.read_sql_query(
    "SELECT COUNT(DISTINCT hadm_id) as unique_hadm FROM discharge WHERE hadm_id IS NOT NULL",
    notes_conn
)
print(f"Unique hadm_ids with data: {hadm_check.iloc[0]['unique_hadm']}")
print("\n")

# Show sample rows
print("SAMPLE DATA (first 5 rows):")
sample = pd.read_sql_query("SELECT * FROM discharge LIMIT 5", notes_conn)
print(sample)
print("\n")

# Check text column
print("TEXT COLUMN CHECK:")
text_check = pd.read_sql_query(
    "SELECT hadm_id, LENGTH(text) as text_length FROM discharge WHERE text IS NOT NULL LIMIT 5",
    notes_conn
)
print(text_check)

# Now check vitals database
vitals_db_path = os.path.join(script_dir, "data", "mimic_iv.db")
vitals_conn = sqlite3.connect(vitals_db_path)

print("\n\n=== VITALS DATABASE ===")
# Get list of all admissions with vitals
vitals_admissions = pd.read_sql_query(
    """
    SELECT DISTINCT hadm_id 
    FROM chartevents 
    WHERE itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
    ORDER BY hadm_id
    LIMIT 10
    """,
    vitals_conn
)
print("Sample hadm_ids with vitals:")
print(vitals_admissions)

# Cross-check: which of these have notes?
print("\n\nCROSS-CHECK - HADM_IDS with BOTH vitals AND notes:")
for hadm in vitals_admissions['hadm_id'].head(10):
    notes_count = pd.read_sql_query(
        f"SELECT COUNT(*) as count FROM discharge WHERE hadm_id = {hadm}",
        notes_conn
    )
    print(f"hadm_id {hadm}: {notes_count.iloc[0]['count']} note(s)")

notes_conn.close()
vitals_conn.close()