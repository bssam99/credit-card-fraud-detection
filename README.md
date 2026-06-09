# Credit Card Fraud Detection: Comparing Deep Learning with Classical Methods

**CISC 867 — Deep Learning | Queen's University | Project 16**  
**Team:** Bssam Osman · Ahmed Nafea · Aya Mohamed  
**Supervisor:** Dr. Hazem Abbas

---

## Overview

This project compares five machine learning models — Logistic Regression, Random Forest, XGBoost, and two MLP configurations — for credit card fraud detection on a highly imbalanced dataset (0.17% fraud rate). We address class imbalance via SMOTE and class-weighted loss functions, and evaluate primarily on PR-AUC, which is more informative than ROC-AUC under extreme imbalance.

### Key results

| Model | ROC-AUC | PR-AUC | F1 (τ=0.5) |
|---|---|---|---|
| Logistic Regression | 0.9731 | 0.7231 | 0.11 |
| Random Forest | 0.9794 | 0.8226 | 0.68 |
| XGBoost | 0.9806 | 0.8625 | 0.68 |
| MLP 4-Layer + Weighted BCE | 0.9673 | 0.8494 | 0.78 |
| MLP 5-Layer + Weighted BCE | 0.9739 | 0.8592 | — |

XGBoost achieves the best PR-AUC among all models. The 5-layer MLP closes to within ~0.003 PR-AUC of XGBoost with ~3× longer training time.

---

## Repository structure

```
credit-card-fraud-detection/
├── data/                        # Not tracked — see Dataset section below
│   └── creditcard.csv
├── notebooks/
│   ├── eda_preprocessing.ipynb  # Student A — EDA, splits, scaling, SMOTE
│   ├── classical_models.ipynb   # Student B — Logistic Regression, Random Forest, XGBoost
│   └── deep_model.ipynb         # Student C — MLP architecture, training, ablation
├── results/                     # Auto-created by notebooks; .npy arrays and CSVs
├── models/                      # Auto-created; saved .pkl / .pt / .json model files
├── report_utils.py              # Shared plotting helpers (ROC/PR curves, confusion matrix)
├── requirements.txt
├── LOG.md
└── README.md
```

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

1. Create a Kaggle account if you don't have one.
2. Download `creditcard.csv` from the link above.
3. Place it at `data/creditcard.csv`.

The dataset is **not** committed to this repository (284 MB; also subject to Kaggle's terms of use).

**Dataset facts:**
- 284,807 transactions over two days in September 2013
- 492 fraudulent transactions (0.172% prevalence)
- 30 features: V1–V28 (PCA-anonymised), Amount, Time
- No missing values

---

## Setup

### Requirements

- Python 3.9+
- ~2 GB free disk space (for `.npy` splits)
- No GPU required — all models were trained on CPU

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
numpy>=1.24
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
imbalanced-learn>=0.10
xgboost>=1.7
torch>=2.0
joblib>=1.2
```

### (Optional) Create a virtual environment first

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## How to run

The three notebooks must be run **in order**. Each notebook reads outputs produced by the previous one.

### Step 1 — EDA and preprocessing

```
notebooks/eda_preprocessing.ipynb
```

- Loads `data/creditcard.csv`
- Produces EDA visualisations saved to `results/`
- Performs a stratified 70/10/20 train/val/test split
- Fits `StandardScaler` **on the training set only**, then applies it to val and test (no leakage)
- Applies SMOTE to the training set only
- Saves all splits as `.npy` files to `results/`

**Outputs written to `results/`:**
`X_train.npy`, `y_train.npy`, `X_train_sm.npy`, `y_train_sm.npy`,
`X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`

### Step 2 — Classical models

```
notebooks/classical_models.ipynb
```

- Reads the `.npy` splits from `results/`
- Trains Logistic Regression, Random Forest, and XGBoost
- Saves trained models to `models/`
- Saves ROC/PR curves and the threshold sensitivity plot to `results/`
- Writes `results/classical_model_summary.csv`

### Step 3 — Deep model and full comparison

```
notebooks/deep_model.ipynb
```

- Reads the `.npy` splits from `results/`
- Trains three MLP configurations (ablation study): 4-layer + weighted BCE, 4-layer + focal loss, 5-layer + weighted BCE
- Saves best checkpoints to `models/mlp_best.pt` and `models/abl_*.pt`
- Writes convergence analysis and ablation results to `results/ablation_summary.csv`
- Reads `results/classical_model_summary.csv` and produces the full cross-model comparison table and bar chart

---

## Reproducibility

All random operations use `SEED = 42`, set consistently across:

- `train_test_split(..., random_state=42)`
- `SMOTE(random_state=42)`
- `LogisticRegression(random_state=42)`, `RandomForestClassifier(random_state=42)`, `XGBClassifier(random_state=42)`
- `torch.manual_seed(42)`

Running the three notebooks in order on the same dataset should reproduce the reported metrics within floating-point tolerance.

---

## Team contributions

| Member | Role | Responsibilities |
|---|---|---|
| Bssam Osman | Student A — EDA & Preprocessing | Data exploration, train/val/test split, StandardScaler (leakage-free), SMOTE, EDA visualisations |
| Ahmed Nafea | Student B — Classical Models | Logistic Regression, Random Forest, XGBoost implementation, threshold sensitivity analysis, feature importance |
| Aya Mohamed | Student C — Deep Model & Reporting | MLP architecture, training loop, convergence analysis, ablation study, full model comparison, report writing |

---

## Methodology summary

**Imbalance handling:**
- SMOTE oversampling applied to the training set only (LR, RF, MLP)
- `scale_pos_weight` used natively in XGBoost on the original imbalanced training set
- Class-weighted BCE loss and focal loss evaluated for the MLP

**Evaluation:**
- **PR-AUC** (primary) — unaffected by the large number of true negatives; random-classifier baseline ≈ 0.172%
- **ROC-AUC** (secondary) — reported for comparability with prior literature
- **F1 at τ=0.5** (tertiary) — interpretable default; threshold optimisation explored separately

**Data leakage prevention:**
- `StandardScaler` is fit only on `X_train`, then applied via `.transform()` to `X_val` and `X_test`
- SMOTE is applied only after the split, and only to the training partition

---

## References

Key references (see full list with synthesis in the report):

1. Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI 2015.
2. Carcillo et al., "SCARFF: A Scalable Framework for Streaming Credit Card Fraud Detection with Spark," IEEE Big Data 2018.
3. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," JAIR 2002.

---

## Use of GenAI tools

Claude (Anthropic) was used for: clarifying concepts during development, grammar and phrasing assistance in the written report, and debugging assistance after team members had first attempted fixes independently. All experimental results, model design decisions, and analysis are the team's own work.
