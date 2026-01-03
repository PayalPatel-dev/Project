#!/usr/bin/env python3
"""
Simplified LSTM Training Script (avoids sympy import issues)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import os

# Load data
print("\n" + "="*70)
print("LSTM DETERIORATION PREDICTION - SIMPLIFIED TRAINING")
print("="*70)
print("\n[1] Loading preprocessed data...")

data = np.load('logs/data/processed_data.npz')
X_train = torch.tensor(data['X_train'], dtype=torch.float32)
y_train = torch.tensor(data['y_train'], dtype=torch.float32)
X_val = torch.tensor(data['X_val'], dtype=torch.float32)
y_val = torch.tensor(data['y_val'], dtype=torch.float32)
X_test = torch.tensor(data['X_test'], dtype=torch.float32)
y_test = torch.tensor(data['y_test'], dtype=torch.float32)

print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")

# Model architecture
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last time step
        out = self.relu(self.fc1(last_hidden))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Initialize model
print("\n[2] Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

model = SimpleLSTM(input_size=X_train.shape[2])
model = model.to(device)

# Count positive samples for class weight
pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
print(f"  Pos weight: {pos_weight:.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create data loaders
batch_size = 32
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

print("\n[3] Training model...")
print("="*70)

model.train()
best_val_loss = float('inf')
patience = 15
patience_counter = 0
epochs = 30

for epoch in range(1, epochs + 1):
    # Train
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch).squeeze()
        loss = criterion(logits, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device)).squeeze()
        val_loss = criterion(val_logits, y_val.to(device)).item()
        
        val_preds = torch.sigmoid(val_logits).cpu().numpy()
        val_auroc = roc_auc_score(y_val.numpy(), val_preds)
    
    print(f"Epoch [{epoch:2d}/{epochs}] | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model_path = os.path.join(os.path.dirname(__file__), "..", "logs", "models", "best_model_simple.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n[OK] Early stopping at epoch {epoch}")
            break
    
    model.train()

# Load best model and evaluate
print("\n" + "="*70)
print("[4] Evaluating on test set...")
print("="*70)

model_path = os.path.join(os.path.dirname(__file__), "..", "logs", "models", "best_model_simple.pt")
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
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    
    print(f"\n[TEST] Results:")
    print(f"  AUROC: {test_auroc:.4f}")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")

print("\n[OK] Training complete!")
print("="*70)
