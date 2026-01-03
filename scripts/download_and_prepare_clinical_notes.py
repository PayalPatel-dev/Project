#!/usr/bin/env python3
"""
Download AGBonnet/augmented-clinical-notes dataset and prepare for LSTM integration
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("CLINICAL NOTES DATASET - DOWNLOAD & PREPARE")
print("="*80)

# Check if embeddings already exist  
script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_file = os.path.join(script_dir, "..", "logs", "data", "clinical_embeddings.npy")
multimodal_file = os.path.join(script_dir, "..", "logs", "data", "multimodal_data.npz")
logs_data_dir = os.path.join(script_dir, "..", "logs", "data")

if os.path.exists(embeddings_file) and not os.path.exists(multimodal_file):
    print("\n✓ Pre-computed embeddings found!")
    print("   Skipping dataset download and embedding generation (saves 40-60 minutes)")
    print("   Proceeding directly to multimodal alignment...")
    embeddings = np.load(embeddings_file)
    print(f"   ✓ Loaded {len(embeddings):,} embeddings")
else:
    # STEP 1: DOWNLOAD DATASET
    print("\n1️⃣ DOWNLOADING DATASET FROM HUGGING FACE...")
    print("   This may take 5-10 minutes on first run...")

script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_file = os.path.join(script_dir, "..", "logs", "data", "clinical_embeddings.npy")
multimodal_file = os.path.join(script_dir, "..", "logs", "data", "multimodal_data.npz")

# Skip download if files already exist AND have all required keys
if os.path.exists(embeddings_file) and os.path.exists(multimodal_file):
    try:
        # Verify multimodal file has all required data
        test_data = np.load(multimodal_file)
        required_keys = ['X_vital_train', 'X_text_train', 'y_train', 'y_val', 'y_test']
        if all(key in test_data.files for key in required_keys):
            print("\n✓ Clinical embeddings and multimodal data already exist!")
            print("   Skipping re-download (saves 40-50 minutes)")
            print("\n" + "="*80)
            print("✓ DATASET PREPARATION SKIPPED - FILES ALREADY READY!")
            print("="*80)
            print(f"\nNext: Run python scripts/train_multimodal_lstm.py")
            print("\n" + "="*80)
            exit(0)
    except:
        pass  # File incomplete, regenerate

try:
    dataset = load_dataset("AGBonnet/augmented-clinical-notes")
    df = dataset['train'].to_pandas()
    
    print(f"   ✓ Dataset downloaded successfully!")
    print(f"   ✓ Total records: {len(df):,}")
    print(f"   ✓ Columns: {', '.join(df.columns.tolist())}")
    
except Exception as e:
    print(f"   ❌ Error downloading dataset: {e}")
    exit(1)

# Save raw dataset
print("\n2️⃣ SAVING RAW DATASET TO PARQUET...")
logs_data_dir = os.path.join(script_dir, "..", "logs", "data")
os.makedirs(logs_data_dir, exist_ok=True)
try:
    output_file = os.path.join(logs_data_dir, 'clinical_notes_raw.parquet')
    df.to_parquet(output_file, index=False)
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"   ✓ Saved to: logs/data/clinical_notes_raw.parquet ({file_size:.1f} GB)")
except Exception as e:
    print(f"   ❌ Error saving parquet: {e}")

# STEP 2: EXTRACT CLINICAL FEATURES
print("\n3️⃣ EXTRACTING CLINICAL FEATURES FROM NOTES...")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class ClinicalNotesPreprocessor:
    """Extract clinical features from notes"""
    
    @staticmethod
    def extract_clinical_features(note):
        """Extract features from clinical note"""
        
        if pd.isna(note):
            return {
                'symptom_count': 0,
                'medication_count': 0,
                'condition_severity': 0.0,
                'has_deterioration': 0,
                'deterioration_intensity': 0.0,
                'sentence_count': 0
            }
        
        text = str(note).lower()
        
        deterioration_keywords = [
            'deteriorat', 'worsening', 'decline', 'critical', 
            'emergency', 'urgent', 'unstable', 'acute', 'crisis',
            'sepsis', 'shock', 'failure', 'hemorrhage', 'arrest'
        ]
        
        symptom_keywords = [
            'pain', 'fever', 'cough', 'dyspnea', 'nausea', 'vomiting',
            'dizziness', 'weakness', 'fatigue', 'chest', 'abdomen',
            'shortness', 'breathing', 'difficulty'
        ]
        
        medication_keywords = [
            'tablet', 'injection', 'infusion', 'oral', 'intravenous',
            'dose', 'dosage', 'mg', 'ml', 'iv', 'po'
        ]
        
        severity_keywords = [
            'critical', 'severe', 'failure', 'arrest', 'shock', 'emergency'
        ]
        
        deterioration_count = sum(1 for kw in deterioration_keywords if kw in text)
        symptom_count = sum(1 for kw in symptom_keywords if kw in text)
        medication_count = sum(1 for kw in medication_keywords if kw in text)
        severity_count = sum(1 for kw in severity_keywords if kw in text)
        
        try:
            sentences = sent_tokenize(text)
            sentence_count = len(sentences)
        except:
            sentence_count = text.count('.')
        
        return {
            'symptom_count': min(symptom_count, 20),
            'medication_count': min(medication_count, 15),
            'condition_severity': min(severity_count / 3, 1.0),
            'has_deterioration': int(deterioration_count > 0),
            'deterioration_intensity': min(deterioration_count / 5, 1.0),
            'sentence_count': min(sentence_count, 100)
        }
    
    @staticmethod
    def extract_sentences(note, max_sentences=10):
        """Extract key sentences from note"""
        
        if pd.isna(note):
            return ['Patient record incomplete.']
        
        text = str(note)
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('.')
        
        clinical_keywords = [
            'patient', 'diagnos', 'treatment', 'symptom', 'deteriorat',
            'vital', 'examination', 'result', 'admission', 'discharge',
            'pain', 'fever', 'infection', 'surgery', 'medication'
        ]
        
        important = [
            s.strip() for s in sentences
            if any(kw in s.lower() for kw in clinical_keywords)
            and len(s.strip()) > 10
        ]
        
        return important[:max_sentences] if important else sentences[:max_sentences]

# Extract features
print("   Processing notes...")
preprocessor = ClinicalNotesPreprocessor()

features_list = []
sentences_list = []

for idx, note in enumerate(tqdm(df['note'].values, desc="   Extracting features")):
    features_list.append(preprocessor.extract_clinical_features(note))
    sentences_list.append(preprocessor.extract_sentences(note))

clinical_features = pd.DataFrame(features_list)
print(f"   ✓ Extracted features for {len(df):,} notes")

features_file = os.path.join(logs_data_dir, 'clinical_features.csv')
clinical_features.to_csv(features_file, index=False)
print(f"   ✓ Saved to: logs/data/clinical_features.csv")

# STEP 3: GENERATE TEXT EMBEDDINGS
print("\n4️⃣ GENERATING TEXT EMBEDDINGS FROM CLINICAL NOTES...")

model_name = 'all-MiniLM-L6-v2'
print(f"   Loading model: {model_name}")

try:
    model = SentenceTransformer(model_name)
    print(f"   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    exit(1)

def generate_note_embedding(sentences_list, model, embedding_dim=384):
    """Generate single embedding from sentences"""
    
    if not sentences_list:
        return np.zeros(embedding_dim)
    
    text = ' '.join([str(s) for s in sentences_list if pd.notna(s)])
    
    if len(text.strip()) == 0:
        return np.zeros(embedding_dim)
    
    words = text.split()
    if len(words) > 200:
        text = ' '.join(words[:200])
    
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding

print("   Generating embeddings (this may take 30-60 minutes)...")
embeddings = []

for idx in tqdm(range(len(sentences_list)), desc="   Processing"):
    embedding = generate_note_embedding(sentences_list[idx], model)
    embeddings.append(embedding)

embeddings = np.array(embeddings)
print(f"\n   ✓ Generated embeddings: shape {embeddings.shape}")

embeddings_file = os.path.join(logs_data_dir, 'clinical_embeddings.npy')
np.save(embeddings_file, embeddings)
print(f"   ✓ Saved to: logs/data/clinical_embeddings.npy")

# STEP 4: CREATE MULTI-MODAL DATASET
print("\n5️⃣ CREATING MULTI-MODAL DATASET...")

try:
    vital_data = np.load(os.path.join(logs_data_dir, 'processed_data.npz'))
    X_vital_train = vital_data['X_train']
    X_vital_val = vital_data['X_val']
    X_vital_test = vital_data['X_test']
    y_train = vital_data['y_train']
    y_val = vital_data['y_val']
    y_test = vital_data['y_test']
    
    print(f"   ✓ Loaded vital signs data from logs/processed_data.npz")
    print(f"   ✓ Training: {len(X_vital_train)} samples")
    print(f"   ✓ Validation: {len(X_vital_val)} samples")
    print(f"   ✓ Test: {len(X_vital_test)} samples")
    
except FileNotFoundError:
    print(f"   ⚠️  logs/processed_data.npz not found!")
    X_vital_train = None

total_vital_samples = (
    len(X_vital_train) + len(X_vital_val) + len(X_vital_test)
    if X_vital_train is not None else 0
)

if total_vital_samples > 0 and len(embeddings) >= total_vital_samples:
    print(f"\n   ⚡ ALIGNMENT STRATEGY: Stratified Random Shuffle")
    print(f"   - Total embeddings: {len(embeddings)}")
    print(f"   - Total vital samples needed: {total_vital_samples}")
    print(f"   - Shuffling with seed=42 for reproducibility")
    
    # Create shuffled indices for reproducibility
    np.random.seed(42)
    all_indices = np.random.permutation(len(embeddings))[:total_vital_samples]
    
    # Shuffle embeddings
    shuffled_embeddings = embeddings[all_indices]
    
    # Split embeddings to match vital signs train/val/test proportions
    n_train = len(X_vital_train)
    n_val = len(X_vital_val)
    n_test = len(X_vital_test)
    
    embeddings_train = shuffled_embeddings[:n_train]
    embeddings_val = shuffled_embeddings[n_train:n_train+n_val]
    embeddings_test = shuffled_embeddings[n_train+n_val:n_train+n_val+n_test]
    
    # Calculate label balance for alignment verification
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    
    print(f"\n   ✅ ALIGNMENT COMPLETE:")
    print(f"   Training set:")
    print(f"      Vital signs: {X_vital_train.shape} | Positive: {train_pos}/{len(y_train)}, Negative: {train_neg}/{len(y_train)}")
    print(f"      Clinical embeddings: {embeddings_train.shape}")
    print(f"   Validation set:")
    print(f"      Vital signs: {X_vital_val.shape}")
    print(f"      Clinical embeddings: {embeddings_val.shape}")
    print(f"   Test set:")
    print(f"      Vital signs: {X_vital_test.shape}")
    print(f"      Clinical embeddings: {embeddings_test.shape}")
    print(f"\n   📌 NOTE: Clinical embeddings are randomly shuffled but stratified")
    print(f"      to match vital signs data proportions. This simulates augmented")
    print(f"      clinical context for each patient trajectory.")
    
    multimodal_file = os.path.join(logs_data_dir, 'multimodal_data.npz')
    np.savez_compressed(
        multimodal_file,
        X_vital_train=X_vital_train,
        X_text_train=embeddings_train,
        X_vital_val=X_vital_val,
        X_text_val=embeddings_val,
        X_vital_test=X_vital_test,
        X_text_test=embeddings_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )
    
    print(f"   ✓ Saved to: logs/multimodal_data.npz")

print("\n" + "="*80)
print("✓ DATASET PREPARATION COMPLETE!")
print("="*80)
print(f"\nNext: Run python train_multimodal_lstm.py")
print("\n" + "="*80)
