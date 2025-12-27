#!/usr/bin/env python3
"""
STEP 2: Clinical Note Classifier
Trains on clinical embeddings to predict deterioration risk.
Independent from vital signs LSTM - learns from text embeddings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
import os
import json

# Configuration
EMBEDDING_DIM = 384
HIDDEN_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 20

print("\n" + "="*80)
print("STEP 2: CLINICAL NOTE CLASSIFIER TRAINING")
print("Input: Clinical embeddings (384-dim) from augmented notes")
print("Output: Deterioration risk score (0-1)")
print("="*80)

# Step 1: Load data
print("\n[STEP 1] Loading clinical embeddings and labels...")

script_dir = os.path.dirname(os.path.abspath(__file__))
logs_data_dir = os.path.join(script_dir, "..", "logs", "data")

# Load embeddings
embeddings_file = os.path.join(logs_data_dir, 'clinical_embeddings.npy')
clinical_embeddings = np.load(embeddings_file, allow_pickle=False)
print(f"   Embeddings shape: {clinical_embeddings.shape}")

# Load labels and features
features_file = os.path.join(logs_data_dir, 'clinical_features.npy')
clinical_features = np.load(features_file) if os.path.exists(features_file) else None

# Create labels from deterioration column
import pandas as pd
features_df = pd.read_csv(os.path.join(logs_data_dir, 'clinical_features.csv'))
y = features_df['has_deterioration'].values.astype(np.float32)
print(f"   Labels shape: {y.shape}")
print(f"   Positive samples: {(y == 1).sum()} ({100*(y==1).sum()/len(y):.2f}%)")
print(f"   Negative samples: {(y == 0).sum()} ({100*(y==0).sum()/len(y):.2f}%)")

# Step 2: Split data (80-10-10)
print("\n[STEP 2] Splitting data into train/val/test...")
X_train, X_temp, y_train, y_temp = train_test_split(
    clinical_embeddings, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Val: {X_val.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 3: Define model architecture
print("\n[STEP 3] Building clinical note classifier...")

class ClinicalNoteClassifier(nn.Module):
    """
    Multi-layer neural network to classify deterioration from embeddings
    """
    def __init__(self, input_dim=384, hidden_size=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 64)
        self.fc4 = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
    
    def forward(self, x):
        """Forward pass through the network"""
        out = self.fc1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        return out

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model = ClinicalNoteClassifier(input_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE)
model = model.to(device)

# Calculate class weights for imbalanced data
pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
print(f"   Positive class weight: {pos_weight:.4f}")

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Data loaders
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\n[STEP 4] Training clinical note classifier...")
print("="*80)

# Training loop
best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'val_auroc': [],
    'lr': []
}

for epoch in range(1, EPOCHS + 1):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch).squeeze()
        loss = criterion(logits, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device)).squeeze()
        val_loss = criterion(val_logits, y_val.to(device)).item()
        
        val_preds = torch.sigmoid(val_logits).cpu().numpy()
        val_auroc = roc_auc_score(y_val.numpy(), val_preds)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auroc'].append(val_auroc)
    history['lr'].append(current_lr)
    
    # Print progress
    print(f"Epoch [{epoch:2d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f} | LR: {current_lr:.1e}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model in logs/models directory
        model_path = os.path.join(os.path.dirname(__file__), "..", "logs", "models", "best_clinical_classifier.pt")
        torch.save(model.state_dict(), model_path)
        print(f"   [BEST] Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n[EARLY STOPPING] Stopped at epoch {epoch}")
            break

# Load best model
print("\n" + "="*80)
print("[STEP 5] Evaluating on test set...")
print("="*80)

model_path = os.path.join(os.path.dirname(__file__), "..", "logs", "models", "best_clinical_classifier.pt")
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    test_logits = model(X_test.to(device)).squeeze()
    test_preds = torch.sigmoid(test_logits).cpu().numpy()
    test_auroc = roc_auc_score(y_test.numpy(), test_preds)
    
    # Binary predictions
    test_binary = (test_preds >= 0.5).astype(int)
    
    # Metrics
    cm = confusion_matrix(y_test.numpy(), test_binary)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test.numpy(), test_preds)

# Save results
print(f"\n[RESULTS] Clinical Note Classifier - Test Results:")
print(f"   AUROC: {test_auroc:.4f}")
print(f"   Sensitivity (TPR): {sensitivity:.4f}")
print(f"   Specificity (TNR): {specificity:.4f}")
print(f"   PPV (Precision): {ppv:.4f}")
print(f"   NPV: {npv:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"\n   Confusion Matrix:")
print(f"   True Negatives: {tn}, False Positives: {fp}")
print(f"   False Negatives: {fn}, True Positives: {tp}")

# Save predictions and results
results = {
    'test_auroc': float(test_auroc),
    'test_sensitivity': float(sensitivity),
    'test_specificity': float(specificity),
    'test_ppv': float(ppv),
    'test_npv': float(npv),
    'test_f1': float(f1),
    'confusion_matrix': {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    },
    'training_epochs': epoch,
    'model_architecture': {
        'input_dim': EMBEDDING_DIM,
        'hidden_size': HIDDEN_SIZE,
        'dropout': 0.3
    }
}

# Save test predictions for fusion
np.save('logs/clinical_test_predictions.npy', test_preds)
np.save('logs/clinical_test_labels.npy', y_test.numpy())

# Save results JSON
with open('logs/clinical_classifier_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[SAVED] Predictions saved to: logs/clinical_test_predictions.npy")
print(f"[SAVED] Results saved to: logs/clinical_classifier_results.json")

# Also save training history
np.savez('logs/clinical_training_history.npz',
    train_loss=np.array(history['train_loss']),
    val_loss=np.array(history['val_loss']),
    val_auroc=np.array(history['val_auroc']),
    lr=np.array(history['lr'])
)
print(f"[SAVED] Training history saved to: logs/clinical_training_history.npz")

print("\n" + "="*80)
print("[COMPLETE] STEP 2: Clinical Note Classifier trained successfully!")
print("="*80)
