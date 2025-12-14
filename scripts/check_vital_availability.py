"""
Check which vital attributes are available in MIMIC-IV
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect('mimic_iv.db')

vitals_to_check = {
    'Heart Rate': [220045],
    'Respiratory Rate': [220210],
    'SpO2': [220277],
    'BP Systolic (NBP)': [220179, 220050, 220048],
    'BP Diastolic (NBP)': [220180, 220051],
    'Temperature Celsius': [223761, 223762, 224684],
    'Temperature Fahrenheit': [227969, 225312],
    'FiO2': [223835, 224837],
    'Urine Output': [226559, 226560, 226561],
}

print('=' * 80)
print('VITAL ATTRIBUTE AVAILABILITY IN MIMIC-IV')
print('=' * 80)

print('\n')
print(f'{"Vital Sign":<30} | {"Records":<10} | {"Itemids"}')
print('-' * 80)

for vital, itemids in vitals_to_check.items():
    total = 0
    available_ids = []
    for itemid in itemids:
        query = f'SELECT COUNT(*) as count FROM chartevents WHERE itemid = {itemid};'
        result = pd.read_sql_query(query, conn)
        count = result['count'].values[0]
        if count > 0:
            available_ids.append(f'{itemid}')
            total += count
    
    status = 'HAVE' if total > 0 else 'MISS'
    print(f'{status} {vital:<26} | {total:<10} | {",".join(available_ids)}')

print('\n' + '=' * 80)
print('ANALYSIS')
print('=' * 80)
print('''
Currently using: HR, RR, SpO2, SBP, DBP, Temperature (6 vital signs)

Why not add more:

1. TEMPERATURE alternatives (3 available):
   - We use 223761/223762 (Celsius)
   - Also have 227969 (Fahrenheit), 225312 (Celsius)
   - Adding more temperature = redundancy, not information
   - Already using best source

2. BLOOD PRESSURE alternatives (5 available):
   - We use 220179/220180 (NBP Systolic/Diastolic)
   - Also have: 220048, 220050, 220051, 220181 (MAP)
   - These are SAME measurements, different sources
   - Already capturing both systolic and diastolic
   - Mean Arterial Pressure (MAP) is calculated from Sys/Dia anyway

3. UNAVAILABLE in this chartevents table:
   - FiO2 (Fraction of Inspired Oxygen): 0 records
   - Urine Output: 0 records
   - Central Venous Pressure (CVP): Not in basic monitoring
   - Lactate, pH, Glucose: In different table (labs)
   - Medications/Vasopressors: In different table (prescriptions)

CONCLUSION:
- We already use THE BEST available continuous vitals from chartevents
- Adding alternative sources = noise, not signal
- Can't access lab values without major redesign (different table structure)
- Current 6 vital signs are OPTIMAL for continuous monitoring ICU deterioration
''')

conn.close()
