#!/usr/bin/env python3
"""
Train Multi-Modal LSTM combining Vital Signs + Clinical Notes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("MULTI-MODAL LSTM TRAINING - VITAL SIGNS + CLINICAL NOTES")
print("="*80)

# LOAD DATA
print("\n1️⃣ LOADING DATA...")

try:
    data = np.load('multimodal_data.npz')
    
    X_vital_train = torch.tensor(data['X_vital_train'], dtype=torch.float32)
    X_text_train = torch.tensor(data['X_text_train'], dtype=torch.float32)
    X_vital_val = torch.tensor(data['X_vital_val'], dtype=torch.float32)
    X_text_val = torch.tensor(data['X_text_val'], dtype=torch.float32)
    X_vital_test = torch.tensor(data['X_vital_test'], dtype=torch.float32)
    X_text_test = torch.tensor(data['X_text_test'], dtype=torch.float32)
    
    print(f"   ✓ Loaded multi-modal data")
    
except FileNotFoundError:
    print(f"   ⚠️  multimodal_data.npz not found!")
    vital_data = np.load('processed_data.npz')
    X_vital_train = torch.tensor(vital_data['X_train'], dtype=torch.float32)
    X_vital_val = torch.tensor(vital_data['X_val'], dtype=torch.float32)
    X_vital_test = torch.tensor(vital_data['X_test'], dtype=torch.float32)
    
    X_text_train = torch.zeros(len(X_vital_train), 384, dtype=torch.float32)
    X_text_val = torch.zeros(len(X_vital_val), 384, dtype=torch.float32)
    X_text_test = torch.zeros(len(X_vital_test), 384, dtype=torch.float32)

y_data = np.load('processed_data.npz')
y_train = torch.tensor(y_data['y_train'], dtype=torch.float32)
y_val = torch.tensor(y_data['y_val'], dtype=torch.float32)
y_test = torch.tensor(y_data['y_test'], dtype=torch.float32)

print(f"   X_vital_train: {X_vital_train.shape}")
print(f"   X_text_train: {X_text_train.shape}")
print(f"   y_train: {y_train.shape}")

# DEFINE MODEL
print("\n2️⃣ DEFINING MULTI-MODAL LSTM ARCHITECTURE...")

class MultiModalLSTM(nn.Module):
    """Multi-modal LSTM combining vital signs + clinical notes"""
    
    def __init__(self, vital_features=6, vital_hidden=128, text_embedding_dim=384,
                 text_hidden=64, fusion_hidden=32, dropout=0.3):
        
        super().__init__()
        
        self.vital_lstm = nn.LSTM(
            input_size=vital_features,
            hidden_size=vital_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.vital_fc1 = nn.Linear(vital_hidden, 64)
        self.vital_fc2 = nn.Linear(64, 32)
        self.vital_relu = nn.ReLU()
        self.vital_dropout = nn.Dropout(dropout)
        
        self.text_fc1 = nn.Linear(text_embedding_dim, text_hidden)
        self.text_fc2 = nn.Linear(text_hidden, 32)
        self.text_relu = nn.ReLU()
        self.text_dropout = nn.Dropout(dropout)
        
        self.fusion_fc1 = nn.Linear(64, fusion_hidden)
        self.fusion_fc2 = nn.Linear(fusion_hidden, 16)
        self.fusion_fc3 = nn.Linear(16, 1)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)
    
    def forward(self, vital_signs, clinical_embedding):
        lstm_out, _ = self.vital_lstm(vital_signs)
        vital_last = lstm_out[:, -1, :]
        
        vital_features = self.vital_relu(self.vital_fc1(vital_last))
        vital_features = self.vital_dropout(vital_features)
        vital_features = self.vital_relu(self.vital_fc2(vital_features))
        vital_features = self.vital_dropout(vital_features)
        
        text_features = self.text_relu(self.text_fc1(clinical_embedding))
        text_features = self.text_dropout(text_features)
        text_features = self.text_relu(self.text_fc2(text_features))
        text_features = self.text_dropout(text_features)
        
        combined = torch.cat([vital_features, text_features], dim=1)
        
        fused = self.fusion_relu(self.fusion_fc1(combined))
        fused = self.fusion_dropout(fused)
        fused = self.fusion_relu(self.fusion_fc2(fused))
        fused = self.fusion_dropout(fused)
        
        output = self.fusion_fc3(fused)
        
        return output

model = MultiModalLSTM()
print(f"   ✓ Model Architecture Defined")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# TRAINING SETUP
print("\n3️⃣ SETTING UP TRAINING...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

model = model.to(device)

pos_weight = (y_train == 0).sum().float() / (y_train == 1).sum().float()
print(f"   Pos weight: {pos_weight:.4f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
train_dataset = TensorDataset(X_vital_train, X_text_train, y_train)
val_dataset = TensorDataset(X_vital_val, X_text_val, y_val)
test_dataset = TensorDataset(X_vital_test, X_text_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# TRAINING LOOP
print("\n4️⃣ TRAINING MULTI-MODAL MODEL...")
print("="*80)

best_val_loss = float('inf')
patience = 15
patience_counter = 0
epochs = 30

train_losses = []
val_losses = []
val_aurocs = []

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    
    for vital_batch, text_batch, label_batch in train_loader:
        vital_batch = vital_batch.to(device)
        text_batch = text_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        
        logits = model(vital_batch, text_batch).squeeze()
        loss = criterion(logits, label_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        val_logits = []
        val_labels = []
        val_loss = 0.0
        
        for vital_batch, text_batch, label_batch in val_loader:
            vital_batch = vital_batch.to(device)
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            
            logits = model(vital_batch, text_batch).squeeze()
            val_logits.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(label_batch.cpu().numpy())
            
            loss = criterion(logits, label_batch)
            val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_logits = np.array(val_logits)
        val_labels = np.array(val_labels)
        val_auroc = roc_auc_score(val_labels, val_logits)
        val_aurocs.append(val_auroc)
    
    print(f"Epoch [{epoch:2d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUROC: {val_auroc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_multimodal_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break
    
    model.train()

print("="*80)

# EVALUATION
print("\n5️⃣ EVALUATING ON TEST SET...")
print("="*80)

model.load_state_dict(torch.load('best_multimodal_model.pt'))
model.eval()

test_logits = []
test_labels = []

with torch.no_grad():
    for vital_batch, text_batch, label_batch in test_loader:
        vital_batch = vital_batch.to(device)
        text_batch = text_batch.to(device)
        
        logits = model(vital_batch, text_batch).squeeze()
        test_logits.extend(torch.sigmoid(logits).cpu().numpy())
        test_labels.extend(label_batch.cpu().numpy())

test_logits = np.array(test_logits)
test_labels = np.array(test_labels)
test_auroc = roc_auc_score(test_labels, test_logits)

test_binary = (test_logits >= 0.5).astype(int)

cm = confusion_matrix(test_labels, test_binary)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = 2 * tp / (2 * tp + fp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\n📊 Multi-Modal Test Results:")
print(f"  AUROC: {test_auroc:.4f}")
print(f"  Sensitivity (TPR): {sensitivity:.4f}")
print(f"  Specificity (TNR): {specificity:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  Accuracy: {accuracy:.4f}")

print(f"\nConfusion Matrix:")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")

vital_only_auroc = 0.9941
improvement = (test_auroc - vital_only_auroc) / vital_only_auroc * 100

print(f"\n✓ MODEL COMPARISON:")
print(f"  Vital-only LSTM:  AUROC {vital_only_auroc:.4f}")
print(f"  Multi-modal LSTM: AUROC {test_auroc:.4f}")
print(f"  Change: {improvement:+.3f}%")

# VISUALIZATION
print("\n6️⃣ SAVING VISUALIZATIONS...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Multi-Modal LSTM Training Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_aurocs, label='Val AUROC', marker='o', linewidth=2, color='green')
axes[1].axhline(y=vital_only_auroc, color='r', linestyle='--', 
                label=f'Vital-only ({vital_only_auroc:.4f})', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUROC')
axes[1].set_title('Validation AUROC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multimodal_training_curves.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: multimodal_training_curves.png")

results = {
    'model': 'MultiModalLSTM',
    'vital_only_auroc': vital_only_auroc,
    'multimodal_auroc': float(test_auroc),
    'improvement_percent': improvement,
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'f1_score': float(f1),
    'accuracy': float(accuracy),
    'test_samples': len(test_labels),
    'true_positives': int(tp),
    'false_positives': int(fp),
    'false_negatives': int(fn),
    'true_negatives': int(tn)
}

with open('multimodal_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("   ✓ Saved: multimodal_results.json")

# SUMMARY
print("\n" + "="*80)
print("✓ MULTI-MODAL TRAINING COMPLETE!")
print("="*80)

print(f"\nResults Summary:")
print(f"  Model: MultiModalLSTM (Vital Signs + Clinical Notes)")
print(f"  Test AUROC: {test_auroc:.4f}")
print(f"  vs Vital-only: {vital_only_auroc:.4f}")
print(f"  Improvement: {improvement:+.3f}%")

print(f"\nNext Steps:")
print(f"  1. Review results in multimodal_results.json")
print(f"  2. Compare with vital-only model performance")
print(f"  3. Update dissertation with multi-modal results")

print("\n" + "="*80)
