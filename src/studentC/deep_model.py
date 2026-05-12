import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay)
import os, time

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
MODELS_DIR  = "models"
SEED        = 42
EPOCHS      = 50
BATCH_SIZE  = 1024
LR          = 1e-3
os.makedirs(MODELS_DIR, exist_ok=True)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
X_train = np.load(f"{RESULTS_DIR}/X_train_sm.npy").astype(np.float32)
y_train = np.load(f"{RESULTS_DIR}/y_train_sm.npy").astype(np.float32)
X_val   = np.load(f"{RESULTS_DIR}/X_val.npy").astype(np.float32)
y_val   = np.load(f"{RESULTS_DIR}/y_val.npy").astype(np.float32)
X_test  = np.load(f"{RESULTS_DIR}/X_test.npy").astype(np.float32)
y_test  = np.load(f"{RESULTS_DIR}/y_test.npy").astype(np.float32)

# PyTorch DataLoaders
def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
val_loader   = make_loader(X_val,   y_val,   shuffle=False)

# ─── MLP ARCHITECTURE ────────────────────────────────────────────────────────
class FraudMLP(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),          nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model     = FraudMLP(input_dim=X_train.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Class-weighted loss (fraud weight = ratio)
fraud_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion = nn.BCELoss(reduction='none')

def weighted_bce(pred, target, pos_weight=fraud_weight):
    loss = criterion(pred, target)
    weights = torch.where(target == 1,
        torch.tensor(pos_weight, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32))
    return (loss * weights).mean()

# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
train_losses, val_losses, val_pr_aucs = [], [], []
best_pr_auc, best_epoch = 0, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = weighted_bce(pred, yb)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    # Validation
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        vl = 0
        for Xb, yb in val_loader:
            pred = model(Xb.to(DEVICE))
            vl += weighted_bce(pred, yb.to(DEVICE)).item()
            val_probs.extend(pred.cpu().numpy())
            val_true.extend(yb.numpy())

    epoch_val_loss = vl / len(val_loader)
    pr_auc = average_precision_score(val_true, val_probs)
    scheduler.step(epoch_val_loss)

    train_losses.append(np.mean(batch_losses))
    val_losses.append(epoch_val_loss)
    val_pr_aucs.append(pr_auc)

    if pr_auc > best_pr_auc:
        best_pr_auc, best_epoch = pr_auc, epoch
        torch.save(model.state_dict(), f"{MODELS_DIR}/mlp_best.pt")

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} "
              f"| Val Loss: {epoch_val_loss:.4f} | Val PR-AUC: {pr_auc:.4f}")

print(f"\nBest Val PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch}")

# ─── LEARNING CURVES ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses,   label='Val Loss')
ax1.set_title("MLP Training & Validation Loss")
ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(val_pr_aucs, color='green')
ax2.axvline(best_epoch - 1, linestyle='--', color='red', label=f'Best (E{best_epoch})')
ax2.set_title("Validation PR-AUC"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/mlp_learning_curves.png", dpi=150)
plt.close()

# ─── TEST EVALUATION ─────────────────────────────────────────────────────────
model.load_state_dict(torch.load(f"{MODELS_DIR}/mlp_best.pt"))
model.eval()
with torch.no_grad():
    test_probs = model(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
test_preds = (test_probs >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, test_probs)
pr_auc  = average_precision_score(y_test, test_probs)
print(f"\nTest ROC-AUC: {roc_auc:.4f} | Test PR-AUC: {pr_auc:.4f}")
print(classification_report(y_test, test_preds, target_names=['Legit', 'Fraud']))

# ─── FINAL COMPARISON ACROSS ALL MODELS ──────────────────────────────────────
import joblib
classical = pd.read_csv(f"{RESULTS_DIR}/classical_model_summary.csv")
mlp_row   = pd.DataFrame([{"name": "MLP (Deep)", "roc_auc": roc_auc, "pr_auc": pr_auc}])
all_models = pd.concat([classical[["name","roc_auc","pr_auc"]], mlp_row], ignore_index=True)
all_models.to_csv(f"{RESULTS_DIR}/full_model_comparison.csv", index=False)
print(f"\nFinal Model Comparison:\n{all_models.to_string(index=False)}")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(all_models))
w = 0.35
ax.bar(x - w/2, all_models['roc_auc'], w, label='ROC-AUC', color='steelblue')
ax.bar(x + w/2, all_models['pr_auc'],  w, label='PR-AUC',  color='crimson')
ax.set_xticks(x); ax.set_xticklabels(all_models['name'], rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Model Comparison: ROC-AUC vs PR-AUC")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/full_model_comparison.png", dpi=150)
plt.close()

print("Deep model complete. All results saved.")