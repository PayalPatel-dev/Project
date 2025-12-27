# Real Data Integration: SQL Queries & Code Reference

## Database Join Strategy

### Problem
- Vitals stored in `mimic_iv.db` (chartevents table)
- Notes stored in `mimic_notes_complete_records.db` (discharge/radiology tables)
- Need to pair them by `hadm_id` (hospital admission ID)

### Solution
Use separate database connections and Python logic to join in memory.

---

## Part 1: SQL Queries

### Query 1: Extract Vital Signs for an Admission

```sql
SELECT 
    ce.charttime,
    ce.itemid,
    di.label AS vital_name,
    ce.valuenum,
    ce.valueuom
FROM chartevents ce
INNER JOIN d_items di ON ce.itemid = di.itemid
WHERE ce.hadm_id = 20044587
  AND ce.itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
ORDER BY ce.charttime DESC
LIMIT 1440;
```

**Result**: All vital measurements for admission, 24 hours of data

| charttime | itemid | vital_name | valuenum | valueuom |
|-----------|--------|-----------|----------|----------|
| 2113-08-27 16:00:00 | 220045 | Heart Rate | 92.0 | bpm |
| 2113-08-27 16:00:00 | 220051 | Systolic BP | 61.0 | mmHg |
| ... | ... | ... | ... | ... |

### Query 2: Extract Discharge Notes for an Admission

```sql
SELECT 
    note_id,
    charttime,
    text
FROM discharge
WHERE hadm_id = 20044587
ORDER BY charttime DESC;
```

**Result**: Clinical note(s) for admission

| note_id | charttime | text |
|---------|-----------|------|
| 10023771-DS-21 | 2113-08-30 00:00 | Name: ___<br/>Unit No: ___<br/>... |

### Query 3: Find Admissions with Both Vitals AND Notes

**Strategy**: Query notes database, then check mimic_iv for each one

```sql
-- In mimic_notes_complete_records.db
SELECT DISTINCT hadm_id FROM discharge;

-- For each hadm_id, in mimic_iv.db:
SELECT COUNT(DISTINCT charttime) as vital_count
FROM chartevents
WHERE hadm_id = 20044587
  AND itemid IN (220045, 220051, 220052, 220210, 220277, 223761);
```

**Python Implementation**:
```python
# Get discharge hadm_ids
discharge_hadm = pd.read_sql_query(
    'SELECT DISTINCT hadm_id FROM discharge',
    notes_conn
)

# Check which have vitals
results = []
for hadm_id in discharge_hadm['hadm_id'].values:
    vital_count = pd.read_sql_query(f"""
        SELECT COUNT(*) as count FROM chartevents
        WHERE hadm_id = {int(hadm_id)}
        AND itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
    """, mimic_conn)
    
    if vital_count['count'].values[0] > 0:
        results.append(int(hadm_id))
```

---

## Part 2: Python Code Reference

### MIMICDataLoader Class

```python
class MIMICDataLoader:
    def __init__(self, mimic_iv_path='mimic_iv.db', notes_path='mimic_notes_complete_records.db'):
        self.mimic_iv_conn = sqlite3.connect(mimic_iv_path)
        self.notes_conn = sqlite3.connect(notes_path)
        self.vital_itemids = [220045, 220051, 220052, 220210, 220277, 223761]
    
    def get_vital_signs(self, hadm_id):
        """Extract vitals for admission"""
        query = """
        SELECT ce.charttime, ce.itemid, ce.valuenum
        FROM chartevents ce
        WHERE ce.hadm_id = ? AND ce.itemid IN (?, ?, ?, ?, ?, ?)
        ORDER BY ce.charttime DESC
        """
        df = pd.read_sql_query(query, self.mimic_iv_conn, 
                               params=(hadm_id,) + tuple(self.vital_itemids))
        return df
    
    def get_discharge_notes(self, hadm_id):
        """Extract discharge notes for admission"""
        query = "SELECT note_id, charttime, text FROM discharge WHERE hadm_id = ?"
        df = pd.read_sql_query(query, self.notes_conn, params=(hadm_id,))
        return df
    
    def close(self):
        self.mimic_iv_conn.close()
        self.notes_conn.close()
```

### Vital Reshaping Function

```python
def reshape_vitals_to_lstm_format(vitals_df, target_hours=24):
    """
    Convert DataFrame of measurements to (24, 6) numpy array
    
    Input:
        vitals_df with columns: charttime, itemid, valuenum
    
    Output:
        (24, 6) numpy array with shape [hours, vitals]
        Column order: [HR, SBP, DBP, RR, SpO2, Temp]
    """
    vital_itemids = [220045, 220051, 220052, 220210, 220277, 223761]
    
    # Extract date and hour from timestamps
    vitals_df = vitals_df.copy()
    vitals_df['date'] = vitals_df['charttime'].dt.date
    vitals_df['hour'] = vitals_df['charttime'].dt.hour
    
    # Get most recent date
    target_date = vitals_df['date'].max()
    day_data = vitals_df[vitals_df['date'] == target_date]
    
    # Pivot: hours x vitals (aggregate multiple measurements per hour)
    pivot = day_data.pivot_table(
        index='hour',
        columns='itemid',
        values='valuenum',
        aggfunc='mean'
    )
    
    # Create output array
    vital_array = np.zeros((target_hours, 6))
    
    # Fill with available data
    for i, itemid in enumerate(vital_itemids):
        if itemid in pivot.columns:
            values = pivot[itemid].values
            vital_array[:len(values), i] = values
    
    return vital_array
```

### Complete Data Loading Example

```python
# Initialize
data_loader = MIMICDataLoader('mimic_iv.db', 'mimic_notes_complete_records.db')

# Get a sample admission
hadm_id = 20044587

# Load vitals
vitals_df = data_loader.get_vital_signs(hadm_id)
print(f"Loaded {len(vitals_df)} vital measurements")

# Reshape to LSTM format
vital_array = reshape_vitals_to_lstm_format(vitals_df)
print(f"Vital array shape: {vital_array.shape}")  # (24, 6)

# Load notes
notes_df = data_loader.get_discharge_notes(hadm_id)
print(f"Loaded {len(notes_df)} clinical notes")

# Get note text
note_text = notes_df.iloc[0]['text']
print(f"Note length: {len(note_text)} characters")

# Get embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(note_text)
print(f"Embeddings shape: {embeddings.shape}")  # (384,)

# Close connections
data_loader.close()
```

---

## Part 3: Data Format Reference

### Vital Signs Array Format

After reshaping, vital array has this structure:

```
vital_array.shape = (24, 6)

Rows (hours):    0, 1, 2, ..., 23  (24 hours in a day)
Columns (vitals):
  0: Heart Rate (HR)           [bpm]
  1: Systolic BP (SBP)         [mmHg]
  2: Diastolic BP (DBP)        [mmHg]
  3: Respiratory Rate (RR)     [breaths/min]
  4: Oxygen Saturation (SpO2)  [%]
  5: Temperature (Temp)        [°C]
```

**Example row (hour 14)**:
```
[72.0, 52.0, 75.0, 24.0, 95.0, 98.5]
 HR    SBP   DBP   RR    SpO2  Temp
```

**Missing hours** are filled with zeros

### Embeddings Format

Clinical notes are converted to embeddings:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(note_text)

embeddings.shape = (384,)  # 384-dimensional vector
```

---

## Part 4: Model Input Requirements

### LSTM Model
- **Input**: Vital signs array of shape `(batch_size, 24, 6)`
- **Output**: Risk score 0.0-1.0 (after sigmoid)
- **Example**: 
  ```python
  vital_tensor = torch.tensor(vital_array).unsqueeze(0)  # (1, 24, 6)
  lstm_score = torch.sigmoid(lstm_model(vital_tensor))   # scalar 0-1
  ```

### Clinical Classifier
- **Input**: Embeddings array of shape `(batch_size, 384)`
- **Output**: Risk score 0.0-1.0 (after sigmoid)
- **Example**:
  ```python
  embedding_tensor = torch.tensor(embeddings).unsqueeze(0)  # (1, 384)
  clinical_score = torch.sigmoid(classifier(embedding_tensor))  # scalar 0-1
  ```

### Fusion Model
- **Input**: Concatenated scores `[lstm_score, clinical_score]`
- **Output**: Final risk score 0.0-1.0
- **Example**:
  ```python
  fusion_input = torch.tensor([lstm_score, clinical_score]).unsqueeze(0)  # (1, 2)
  final_score = torch.sigmoid(fusion_model(fusion_input))  # scalar 0-1
  ```

---

## Part 5: Common Issues & Solutions

### Issue 1: Cross-Database Joins Don't Work

**Problem**:
```python
query = """
SELECT * FROM mimic_iv.chartevents ce
JOIN mimic_notes.discharge d ON ce.hadm_id = d.hadm_id
"""
```

**Solution**: Use separate connections and join in Python
```python
mimic_conn = sqlite3.connect('mimic_iv.db')
notes_conn = sqlite3.connect('mimic_notes_complete_records.db')

vitals_df = pd.read_sql_query("SELECT ...", mimic_conn)
notes_df = pd.read_sql_query("SELECT ...", notes_conn)

# Both have hadm_id column for joining
merged = vitals_df.merge(notes_df, on='hadm_id')
```

### Issue 2: Vital Array Has All Zeros

**Problem**: No data loaded from database

**Solution**: Check hadm_id exists in chartevents
```python
query = f"""
SELECT COUNT(*) FROM chartevents 
WHERE hadm_id = {hadm_id} 
AND itemid IN (220045, 220051, 220052, 220210, 220277, 223761)
"""
count = pd.read_sql_query(query, mimic_conn)
```

### Issue 3: Missing Vitals in Array

**Problem**: Not all 6 vitals present for all hours

**Solution**: This is normal - use zeros for missing hours (model handles this)
```python
# Valid - some vitals missing
vital_array[5, :] = [72.0, 0.0, 0.0, 24.0, 95.0, 98.5]
                     # SBP, DBP missing but RR, SpO2 present
```

### Issue 4: Note Text is Empty or NULL

**Problem**: Database returns NULL text field

**Solution**: Validate before processing
```python
note_text = notes_df.iloc[0]['text']
if note_text is None or (isinstance(note_text, float)):
    note_text = "No clinical note available"
embeddings = model.encode(note_text)
```

---

## Part 6: Performance Tips

### 1. Use LIMIT to Avoid Large Queries
```python
# Good - limits to 1440 measurements (60 days max)
query = "SELECT ... FROM chartevents ... LIMIT 1440"

# Bad - loads entire table
query = "SELECT * FROM chartevents WHERE hadm_id = ?"
```

### 2. Filter by itemid First
```python
# Good - filter on vital itemids reduces rows from 668K to dozens
WHERE hadm_id = ? AND itemid IN (220045, 220051, ...)

# Bad - loads all measurements then filters
WHERE hadm_id = ? AND itemid IN (...)
```

### 3. Use Indexed Columns
```sql
-- discharge and radiology tables have indexes on hadm_id
CREATE INDEX idx_discharge_hadm ON discharge(hadm_id)
CREATE INDEX idx_radiology_hadm ON radiology(hadm_id)

-- chartevents has no index on hadm_id but has many rows
```

### 4. Cache Connections
```python
# Good - reuse connection
conn = sqlite3.connect('mimic_iv.db')
for hadm_id in admission_list:
    df = pd.read_sql_query(query, conn, params=(hadm_id,))

# Bad - create new connection each time
for hadm_id in admission_list:
    conn = sqlite3.connect('mimic_iv.db')
    df = pd.read_sql_query(query, conn, params=(hadm_id,))
```

---

## Part 7: Debugging Checklist

✅ Database files exist and are accessible
✅ hadm_id exists in both databases
✅ Vital itemids are correct: [220045, 220051, 220052, 220210, 220277, 223761]
✅ Vital array shape is (24, 6)
✅ Note text is not NULL/empty
✅ Embedding dimension is 384
✅ Model input types are correct (torch.float32)
✅ Model output is in range [0.0, 1.0]

---

**Last Updated**: December 25, 2025  
**Database Version**: MIMIC-IV v2.2  
**Data**: Real Hospital Records (Not Synthetic)
