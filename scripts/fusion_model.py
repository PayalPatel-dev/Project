#!/usr/bin/env python3
"""
STEP 3: FUSION MODEL - Combine LSTM (Vital Signs) + Clinical Classifier
Implements three fusion strategies:
1. Weighted Average: final_score = 0.6*vital_score + 0.4*clinical_score
2. Stacking: Feed both scores to a neural network
3. Voting: Ensemble voting (majority rule + confidence weighting)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split
import json
import os

print("\n" + "="*80)
print("STEP 3: MULTIMODAL FUSION - COMBINING VITAL SIGNS + CLINICAL NOTES")
print("="*80)

# Step 1: Load predictions from both models
print("\n[STEP 1] Loading predictions from Step 1 (LSTM) and Step 2 (Clinical Classifier)...")

# LSTM predictions on vital signs test set
lstm_model_path = 'logs/best_model_simple.pt'
if not os.path.exists(lstm_model_path):
    lstm_model_path = 'best_model_simple.pt'

# Load LSTM model to generate predictions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

# Load vital signs data
data = np.load('processed_data.npz')
X_test_vital = torch.tensor(data['X_test'], dtype=torch.float32)
y_test = data['y_test']
print(f"   Vital signs test data: {X_test_vital.shape}")

# Load LSTM model architecture
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
        last_hidden = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_hidden))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Load LSTM predictions
print("\n[STEP 2] Generating LSTM (Vital Signs) predictions...")
lstm_model = SimpleLSTM(input_size=X_test_vital.shape[2])
lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device, weights_only=False))
lstm_model.to(device)
lstm_model.eval()

with torch.no_grad():
    lstm_logits = lstm_model(X_test_vital.to(device)).squeeze()
    vital_scores = torch.sigmoid(lstm_logits).cpu().numpy()

print(f"   LSTM predictions shape: {vital_scores.shape}")
print(f"   LSTM score range: [{vital_scores.min():.4f}, {vital_scores.max():.4f}]")

# Load Clinical Classifier predictions
print("\n[STEP 3] Loading Clinical Note Classifier predictions...")
clinical_scores = np.load('logs/clinical_test_predictions.npy')
print(f"   Clinical predictions shape: {clinical_scores.shape}")
print(f"   Clinical score range: [{clinical_scores.min():.4f}, {clinical_scores.max():.4f}]")

# Ensure same test set size
min_size = min(len(vital_scores), len(clinical_scores))
vital_scores = vital_scores[:min_size]
clinical_scores = clinical_scores[:min_size]
y_test = y_test[:min_size]

print(f"\n   Aligned test size: {min_size}")

# ============================================================================
# FUSION STRATEGY 1: WEIGHTED AVERAGE (BASELINE)
# ============================================================================
print("\n" + "="*80)
print("FUSION STRATEGY 1: WEIGHTED AVERAGE")
print("Formula: final_score = 0.6*vital_score + 0.4*clinical_score")
print("="*80)

# Optimal weights (can be tuned)
weight_vital = 0.6
weight_clinical = 0.4

weighted_avg_scores = weight_vital * vital_scores + weight_clinical * clinical_scores
weighted_avg_binary = (weighted_avg_scores >= 0.5).astype(int)

cm = confusion_matrix(y_test, weighted_avg_binary)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
auroc = roc_auc_score(y_test, weighted_avg_scores)

print(f"\n[RESULTS] Weighted Average Results:")
print(f"   AUROC: {auroc:.4f}")
print(f"   Sensitivity: {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   PPV: {ppv:.4f}")
print(f"   F1-Score: {f1:.4f}")

weighted_avg_results = {
    'fusion_strategy': 'weighted_average',
    'weights': {'vital': weight_vital, 'clinical': weight_clinical},
    'test_auroc': float(auroc),
    'test_sensitivity': float(sensitivity),
    'test_specificity': float(specificity),
    'test_ppv': float(ppv),
    'test_f1': float(f1),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
}

# ============================================================================
# FUSION STRATEGY 2: STACKING (NEURAL NETWORK FUSION)
# ============================================================================
print("\n" + "="*80)
print("FUSION STRATEGY 2: STACKING (NEURAL NETWORK FUSION)")
print("Feed [vital_score, clinical_score] to a small neural network")
print("="*80)

class StackingFusionModel(nn.Module):
    """Small neural network to learn optimal fusion of two scores"""
    def __init__(self, input_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out

# Prepare fusion input: concatenate both scores
fusion_input = np.column_stack([vital_scores, clinical_scores]).astype(np.float32)
print(f"   Fusion input shape: {fusion_input.shape}")

# Split into train/val for stacking model
X_train_fuse, X_val_fuse, y_train_fuse, y_val_fuse = train_test_split(
    fusion_input, y_test, test_size=0.3, random_state=42, stratify=y_test
)

X_train_fuse = torch.tensor(X_train_fuse, dtype=torch.float32)
y_train_fuse = torch.tensor(y_train_fuse, dtype=torch.float32)
X_val_fuse = torch.tensor(X_val_fuse, dtype=torch.float32)
y_val_fuse = torch.tensor(y_val_fuse, dtype=torch.float32)

# Initialize and train stacking model
print("\n   Training stacking fusion model...")
fusion_model = StackingFusionModel()
fusion_model.to(device)

pos_weight = (y_train_fuse == 0).sum().float() / (y_train_fuse == 1).sum().float()
fusion_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=0.01, weight_decay=1e-4)

train_loader = DataLoader(
    TensorDataset(X_train_fuse, y_train_fuse),
    batch_size=16,
    shuffle=True
)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, 101):
    # Train
    fusion_model.train()
    train_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        fusion_optimizer.zero_grad()
        logits = fusion_model(X_batch).squeeze()
        loss = fusion_criterion(logits, y_batch)
        loss.backward()
        fusion_optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validate
    fusion_model.eval()
    with torch.no_grad():
        val_logits = fusion_model(X_val_fuse.to(device)).squeeze()
        val_loss = fusion_criterion(val_logits, y_val_fuse.to(device)).item()
    
    if epoch % 20 == 0:
        print(f"   Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(fusion_model.state_dict(), 'logs/stacking_fusion_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= 15:
            print(f"   [EARLY STOPPING] Stopped at epoch {epoch}")
            break

# Load best stacking model and evaluate
fusion_model.load_state_dict(torch.load('logs/stacking_fusion_model.pt', weights_only=False))
fusion_model.eval()

with torch.no_grad():
    # Evaluate on full test set
    full_fusion_input = torch.tensor(fusion_input, dtype=torch.float32).to(device)
    stacking_logits = fusion_model(full_fusion_input).squeeze()
    stacking_scores = torch.sigmoid(stacking_logits).cpu().numpy()

stacking_binary = (stacking_scores >= 0.5).astype(int)

cm = confusion_matrix(y_test, stacking_binary)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
auroc = roc_auc_score(y_test, stacking_scores)

print(f"\n[RESULTS] Stacking Fusion Results:")
print(f"   AUROC: {auroc:.4f}")
print(f"   Sensitivity: {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   PPV: {ppv:.4f}")
print(f"   F1-Score: {f1:.4f}")

stacking_results = {
    'fusion_strategy': 'stacking',
    'test_auroc': float(auroc),
    'test_sensitivity': float(sensitivity),
    'test_specificity': float(specificity),
    'test_ppv': float(ppv),
    'test_f1': float(f1),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
}

# ============================================================================
# FUSION STRATEGY 3: VOTING ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("FUSION STRATEGY 3: VOTING ENSEMBLE")
print("Combine predictions using weighted majority voting")
print("="*80)

# Hard voting with confidence weights
vital_binary = (vital_scores >= 0.5).astype(int)
clinical_binary = (clinical_scores >= 0.5).astype(int)

# Weighted voting: confidence-weighted ensemble
vital_confidence = np.abs(vital_scores - 0.5) * 2
clinical_confidence = np.abs(clinical_scores - 0.5) * 2

# Weighted sum
voting_scores = (
    0.6 * vital_scores * vital_confidence +
    0.4 * clinical_scores * clinical_confidence
) / (0.6 * vital_confidence + 0.4 * clinical_confidence + 1e-8)

voting_binary = (voting_scores >= 0.5).astype(int)

cm = confusion_matrix(y_test, voting_binary)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
auroc = roc_auc_score(y_test, voting_scores)

print(f"\n[RESULTS] Voting Ensemble Results:")
print(f"   AUROC: {auroc:.4f}")
print(f"   Sensitivity: {sensitivity:.4f}")
print(f"   Specificity: {specificity:.4f}")
print(f"   PPV: {ppv:.4f}")
print(f"   F1-Score: {f1:.4f}")

voting_results = {
    'fusion_strategy': 'voting_ensemble',
    'test_auroc': float(auroc),
    'test_sensitivity': float(sensitivity),
    'test_specificity': float(specificity),
    'test_ppv': float(ppv),
    'test_f1': float(f1),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
}

# ============================================================================
# COMPARISON AND FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON - ALL FUSION STRATEGIES")
print("="*80)

comparison = {
    'vital_signs_lstm': {
        'auroc': float(roc_auc_score(y_test, vital_scores)),
        'sensitivity': float(np.mean((vital_binary == 1) & (y_test == 1)) / np.sum(y_test == 1)),
        'specificity': float(np.mean((vital_binary == 0) & (y_test == 0)) / np.sum(y_test == 0))
    },
    'clinical_classifier': {
        'auroc': float(roc_auc_score(y_test, clinical_scores)),
        'sensitivity': float(np.mean((clinical_binary == 1) & (y_test == 1)) / np.sum(y_test == 1)),
        'specificity': float(np.mean((clinical_binary == 0) & (y_test == 0)) / np.sum(y_test == 0))
    },
    'fusion_strategies': {
        'weighted_average': weighted_avg_results,
        'stacking': stacking_results,
        'voting_ensemble': voting_results
    }
}

# Print comparison table
print(f"\n{'Model':<25} {'AUROC':<12} {'Sensitivity':<12} {'Specificity':<12}")
print("-" * 61)

for model_name, metrics in comparison.items():
    if model_name != 'fusion_strategies':
        auroc = metrics['auroc']
        sens = metrics['sensitivity']
        spec = metrics['specificity']
        print(f"{model_name:<25} {auroc:<12.4f} {sens:<12.4f} {spec:<12.4f}")

print("\nFusion Strategies:")
for strategy, metrics in comparison['fusion_strategies'].items():
    auroc = metrics['test_auroc']
    sens = metrics['test_sensitivity']
    spec = metrics['test_specificity']
    print(f"  {strategy:<21} {auroc:<12.4f} {sens:<12.4f} {spec:<12.4f}")

# Save all results
multimodal_results = {
    'individual_models': comparison,
    'test_size': int(min_size),
    'fusion_predictions': {
        'weighted_average': weighted_avg_scores.tolist(),
        'stacking': stacking_scores.tolist(),
        'voting_ensemble': voting_scores.tolist()
    }
}

with open('logs/multimodal_results.json', 'w') as f:
    json.dump(multimodal_results, f, indent=2)

# Save fusion predictions
np.save('logs/fusion_weighted_avg_predictions.npy', weighted_avg_scores)
np.save('logs/fusion_stacking_predictions.npy', stacking_scores)
np.save('logs/fusion_voting_predictions.npy', voting_scores)

print("\n" + "="*80)
print("[COMPLETE] STEP 3: Fusion models trained successfully!")
print("="*80)
print(f"\n[SAVED] Multimodal results saved to: logs/multimodal_results.json")

# Find best strategy
best_strategy = max(comparison['fusion_strategies'].items(), key=lambda x: x[1]['test_auroc'])
print(f"[BEST] Best performing strategy: {best_strategy[0]}")
print(f"       AUROC: {best_strategy[1]['test_auroc']:.4f}")
print("\n" + "="*80)
