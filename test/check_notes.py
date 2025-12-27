import json

with open('real_data_predictions_summary.json') as f:
    data = json.load(f)

admissions = data['admissions']
print(f'Total admissions processed: {len(admissions)}')

with_notes = [a for a in admissions if a['predictions']['clinical_classifier_score'] > 0]
without_notes = [a for a in admissions if a['predictions']['clinical_classifier_score'] == 0]

print(f'\nAdmissions WITH clinical notes: {len(with_notes)}')
print(f'Admissions WITHOUT clinical notes: {len(without_notes)}')

print('\n=== ADMISSIONS WITH NOTES (first 10) ===')
for a in with_notes[:10]:
    print(f"hadm_id {a['admission_id']}: clinical_score={a['predictions']['clinical_classifier_score']:.4f}")

print('\n=== ADMISSIONS WITHOUT NOTES (first 10) ===')
for a in without_notes[:10]:
    print(f"hadm_id {a['admission_id']}: clinical_score={a['predictions']['clinical_classifier_score']}")

# Now check if these hadm_ids exist in the notes database
print('\n=== VERIFYING AGAINST NOTES DATABASE ===')
import sqlite3
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
notes_db_path = os.path.join(script_dir, "..", "data", "mimic_notes_complete_records.db")
notes_conn = sqlite3.connect(notes_db_path)

print('\nAll unique hadm_ids in discharge table:')
all_notes_hadm = pd.read_sql_query(
    "SELECT DISTINCT hadm_id FROM discharge ORDER BY hadm_id", 
    notes_conn
)
notes_hadm_set = set(all_notes_hadm['hadm_id'].values)
print(f'Total hadm_ids with notes: {len(notes_hadm_set)}')

# Check overlap
processed_hadm_set = set(a['admission_id'] for a in admissions)
overlap = processed_hadm_set.intersection(notes_hadm_set)

print(f'\nAdmissions in BOTH pipeline output AND notes DB: {len(overlap)}')
print(f'Admissions in pipeline but NOT in notes DB: {len(processed_hadm_set - notes_hadm_set)}')

# Show some that should have notes but don't in the results
print('\n=== Sample admissions that SHOULD have notes ===')
for hadm in list(notes_hadm_set)[:10]:
    admission_data = [a for a in admissions if a['admission_id'] == hadm]
    if admission_data:
        score = admission_data[0]['predictions']['clinical_classifier_score']
        print(f'hadm_id {hadm}: clinical_score={score:.4f} (should be > 0 if notes loaded)')
    
notes_conn.close()

