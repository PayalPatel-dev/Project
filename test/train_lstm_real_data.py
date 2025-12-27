#!/usr/bin/env python3
"""
Train LSTM model using the ACTUAL dataset used for other models
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import os
# ============================================================================
# Model Architecture
# ============================================================================
class WorkingLSTM(nn.Module):
    """Improved LSTM for vital signs classification"""
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(16)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        out = self.fc1(last_hidden)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out

# ============================================================================
# Dataset
# ============================================================================
class VitalSignsRealDataset(Dataset):
    """Real vital signs dataset from processed_data.npz"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# Load Data
# ============================================================================
print("="*70)
print("TRAINING LSTM ON ACTUAL DATASET")
print("="*70)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "logs", "data", "processed_data.npz")
data = np.load(data_path)

X_train = data['X_train']  # (868, 24, 6)
y_train = data['y_train']  # (868,)
X_val = data['X_val']      # (217, 24, 6)
y_val = data['y_val']      # (217,)
X_test = data['X_test']    # (272, 24, 6)
y_test = data['y_test']    # (272,)

print(f"\n[DATA LOADED]")
print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Validation set: X_val {X_val.shape}, y_val {y_val.shape}")
print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")
print(f"Class distribution (train): {np.bincount(y_train)}")
print(f"Class distribution (test): {np.bincount(y_test)}")

# ============================================================================
# Training Setup
# ============================================================================
device = torch.device('cpu')
batch_size = 32
num_epochs = 100
learning_rate = 0.001

train_dataset = VitalSignsRealDataset(X_train, y_train)
val_dataset = VitalSignsRealDataset(X_val, y_val)
test_dataset = VitalSignsRealDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\n[DATALOADERS CREATED]")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================================================
# Model and Optimizer
# ============================================================================
model = WorkingLSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.3)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

print(f"\n[MODEL CREATED]")
print(f"LSTM(input_size=6, hidden_size=64, num_layers=2)")

# ============================================================================
# Training Loop
# ============================================================================
print("\n[TRAINING]")
print("-"*70)

best_val_loss = float('inf')
best_epoch = 0
patience = 20
patience_counter = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)  # (batch_size, 24, 6)
        batch_y = batch_y.to(device).unsqueeze(1)  # (batch_size, 1)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_predictions.extend(probs.flatten())
            val_targets.extend(batch_y.cpu().numpy().flatten())
    
    val_loss /= len(val_loader)
    val_auc = roc_auc_score(val_targets, val_predictions)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        # Save best model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_save_path = os.path.join(script_dir, "..", "logs", "models", "working_lstm_model.pt")
        torch.save(model.state_dict(), model_save_path)
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
        break

# ============================================================================
# Test Evaluation
# ============================================================================
print("\n[TESTING]")
print("-"*70)

model.load_state_dict(torch.load(os.path.join(script_dir, "..", "logs", "models", "working_lstm_model.pt"), map_location=device, weights_only=False))
model.eval()

test_predictions = []
test_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.sigmoid(outputs).cpu().numpy()
        test_predictions.extend(probs.flatten())
        test_targets.extend(batch_y.cpu().numpy().flatten())

test_predictions = np.array(test_predictions)
test_targets = np.array(test_targets)

# Metrics
test_auc = roc_auc_score(test_targets, test_predictions)
test_acc = accuracy_score(test_targets, (test_predictions > 0.5).astype(int))

print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(test_targets, (test_predictions > 0.5).astype(int))
print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(test_targets, (test_predictions > 0.5).astype(int), 
                          target_names=['Normal', 'Abnormal']))

# Score distribution
normal_scores = test_predictions[test_targets == 0]
abnormal_scores = test_predictions[test_targets == 1]

print(f"\nScore Distribution:")
print(f"Normal patients - Mean: {normal_scores.mean():.4f}, Std: {normal_scores.std():.4f}, Min: {normal_scores.min():.4f}, Max: {normal_scores.max():.4f}")
print(f"Abnormal patients - Mean: {abnormal_scores.mean():.4f}, Std: {abnormal_scores.std():.4f}, Min: {abnormal_scores.min():.4f}, Max: {abnormal_scores.max():.4f}")

print("\n" + "="*70)
print(f"Model saved to: ../logs/working_lstm_model.pt")
print("="*70)
