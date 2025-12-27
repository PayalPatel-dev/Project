import json
import pandas as pd
import numpy as np
from pathlib import Path

def extract_hadm_ids_from_mimic_data():
    """
    Extract all unique hadm_id values from MIMIC data and save to Excel
    """
    
    print("Loading MIMIC data...")
    
    hadm_ids = set()
    
    # Try to load from the JSON analysis file
    json_file = Path('mimic_data_complete_analysis.json')
    if json_file.exists():
        print(f"Reading from {json_file}...")
        with open(json_file, 'r') as f:
            mimic_data = json.load(f)
        
        # Extract hadm_ids from discharge dataset
        if 'datasets' in mimic_data and 'discharge' in mimic_data['datasets']:
            discharge_data = mimic_data['datasets']['discharge']
            if 'sample_rows' in discharge_data:
                for row in discharge_data['sample_rows']:
                    if 'hadm_id' in row:
                        hadm_ids.add(row['hadm_id'])
    
    # Try to load from parquet file if it exists
    parquet_file = Path('clinical_notes_raw.parquet')
    if parquet_file.exists():
        print(f"Reading from {parquet_file}...")
        try:
            df_parquet = pd.read_parquet(parquet_file)
            if 'hadm_id' in df_parquet.columns:
                hadm_ids.update(df_parquet['hadm_id'].dropna().unique().tolist())
                print(f"Found {len(df_parquet['hadm_id'].unique())} unique hadm_ids in parquet")
        except Exception as e:
            print(f"Error reading parquet: {e}")
    
    # Try to load from SQLite database if it exists
    db_file = Path('mimic_iv.db')
    if db_file.exists():
        print(f"Reading from {db_file}...")
        try:
            import sqlite3
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"Available tables: {tables}")
            
            # Try to get hadm_id from each table
            for table in tables:
                table_name = table[0]
                try:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if 'hadm_id' in columns:
                        print(f"  Extracting hadm_id from {table_name}...")
                        cursor.execute(f"SELECT DISTINCT hadm_id FROM {table_name} WHERE hadm_id IS NOT NULL")
                        ids = cursor.fetchall()
                        hadm_ids.update([id[0] for id in ids])
                        print(f"    Found {len(ids)} unique hadm_ids")
                except Exception as e:
                    print(f"  Error reading {table_name}: {e}")
            
            conn.close()
        except Exception as e:
            print(f"Error reading database: {e}")
    
    # Convert to sorted list
    hadm_ids_list = sorted(list(hadm_ids))
    
    print(f"\nTotal unique hadm_id values found: {len(hadm_ids_list)}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'hadm_id': hadm_ids_list
    })
    
    # Save to Excel
    output_file = Path('hadm_ids.xlsx')
    df.to_excel(output_file, index=False, sheet_name='hadm_ids')
    
    print(f"\nExcel file created: {output_file}")
    print(f"Total records: {len(df)}")
    print(f"\nFirst 10 hadm_ids:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    df = extract_hadm_ids_from_mimic_data()
