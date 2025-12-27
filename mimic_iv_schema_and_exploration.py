import sqlite3
import pandas as pd

# ==============================
# CONFIG
# ==============================
DB_PATH = "mimic_iv.db"
OUTPUT_FILE = "mimic_iv_schema_and_exploration.txt"

# ==============================
# CONNECT TO DATABASE
# ==============================
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("MIMIC-IV DATABASE SCHEMA & DATA EXPLORATION\n")
    f.write("=" * 60 + "\n\n")

    # ==============================
    # GET ALL TABLES
    # ==============================
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    f.write(f"Total Tables Found: {len(tables)}\n\n")

    # ==============================
    # LOOP THROUGH TABLES
    # ==============================
    for table_name in tables:
        table = table_name[0]
        f.write(f"\nTABLE NAME: {table}\n")
        f.write("-" * 50 + "\n")

        # ==============================
        # TABLE SCHEMA
        # ==============================
        f.write("Schema:\n")
        cursor.execute(f"PRAGMA table_info({table});")
        schema = cursor.fetchall()

        for col in schema:
            col_id, name, dtype, notnull, default, pk = col
            f.write(
                f"  - Column: {name} | Type: {dtype} | "
                f"Not Null: {bool(notnull)} | Primary Key: {bool(pk)}\n"
            )

        # ==============================
        # ROW COUNT
        # ==============================
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        f.write(f"\nTotal Rows: {row_count}\n")

        # ==============================
        # SAMPLE DATA
        # ==============================
        f.write("\nSample Data (Top 5 Rows):\n")
        try:
            df_sample = pd.read_sql_query(
                f"SELECT * FROM {table} LIMIT 5", conn
            )
            f.write(df_sample.to_string(index=False))
            f.write("\n")
        except Exception as e:
            f.write(f"Could not fetch sample data: {e}\n")

        f.write("\n" + "=" * 60 + "\n")

# ==============================
# CLOSE CONNECTION
# ==============================
conn.close()

print("TXT file generated successfully:", OUTPUT_FILE)
