# Development Log — Credit Card Fraud Detection
## Project 16 | Team: 3 Members | Submitted via GitHub
### Student A: Bassam Osman (Team  Leader)
### Student B: Ahmed Magdy Nafea
### Student C: Aya Mohamed Mahmoud 
---

## Week 1 (Days 1–7): Data Understanding & Setup

### Student A — Data Exploration & Preprocessing

- [x] Cloned the repository and initialized the project folder structure.
- [x] Read the assigned papers:
  - Dal Pozzolo et al. (2015), "Calibrating Probability with Undersampling for Unbalanced Classification"
  - Lazar et al. (2018), "Predicting Network Traffic Using TCP Anomalies" (context paper)
- [x] Downloaded and inspected the Kaggle Credit Card Fraud dataset.
  - **Key observations:** 284,807 transactions; 492 frauds (0.172%); features V1–V28 are PCA-transformed; Time and Amount need normalization; no missing values.
- [x] Implemented and committed `src/studentA/eda_preprocessing.ipynb`:
  - StandardScaler applied to Amount and Time.
  - Stratified 70/10/20 train/val/test split.
  - SMOTE applied to training set only (after splitting).
  - Saved `.npy` split files to `results/`.
- [x] Generated 3 EDA plots: amount distribution, class imbalance pie chart, correlation heatmap.
- [x] **Decision:** Apply SMOTE only to training data to prevent data leakage.
- [x] **Issue:** SMOTE with default k=5 ran slowly on full training set. Resolved by setting `k_neighbors=5` and using original (non-SMOTE) splits for logging.


### Student B — Environment Setup & Paper Review

- [x] Set up Python environment, installed all libraries from `requirements.txt`.
- [x] Read Dal Pozzolo et al. (2015) — understood undersampling vs oversampling trade-offs.
- [x] Reviewed scikit-learn documentation for LogisticRegression, RandomForestClassifier.
- [x] Designed evaluation framework (ROC-AUC, PR-AUC, cost-sensitive metrics).
- [x] Started `src/studentB/classical_models.ipynb` skeleton.

### Student C — Environment Setup & Deep Learning Design

- [x] Installed PyTorch (CPU version) and verified functionality on laptop.
- [x] Studied MLP architectures for tabular fraud detection.
- [x] Designed 4-layer MLP with BatchNorm and Dropout.
- [x] Read about weighted BCE loss for imbalanced classification.
- [x] Created `src/studentC/deep_model.ipynb` skeleton.

---

## Week 2 (Days 8–14): Model Implementation & Midterm Report

### Student A — Midterm Report Lead + Data Finalization

- [x] Verified all `.npy` split files are correctly shaped and saved.
- [x] Ran EDA on the test set to confirm no leakage.
- [x] Collected preliminary statistics for midterm report (class ratio, feature statistics).
- [x] Drafted Sections 1–3 of the IEEE midterm report (Abstract, Introduction, Related Work).
- [x] Coordinated with B and C to collect preliminary results for Section 5.

### Student B — Classical Models Implementation

- [x] Implemented Logistic Regression with `class_weight='balanced'` and C=0.1.
  - **Test ROC-AUC:** 0.9731 | **PR-AUC:** 0.7258
- [x] Implemented Random Forest (200 trees, max_depth=12).
  - **Test ROC-AUC:** 0.9860 | **PR-AUC:** 0.8165
- [x] Implemented XGBoost with `scale_pos_weight` set to class ratio.
  - **Test ROC-AUC:** 0.9787 | **PR-AUC:** 0.8700
- [x] Generated ROC/PR curves and feature importance plot.
- [x] **Issue:** XGBoost training took ~8 min on full SMOTE dataset. Reduced to 300 estimators.
- [x] Committed results to `results/classical_model_summary.csv`.

### Student C — MLP Implementation

- [x] Implemented `FraudMLP` with architecture: Linear(30→128)→BN→ReLU→Dropout → Linear(128→64)→BN→ReLU→Dropout → Linear(64→32)→BN→ReLU→Dropout → Linear(32→1)→Sigmoid.
- [x] Used weighted BCE loss to handle class imbalance.
- [x] Added `ReduceLROnPlateau` scheduler.
- [x] Ran 50 epochs on SMOTE-balanced training set.
  - **Best Val PR-AUC (Epoch ~38):** ~0.875
  - **Test ROC-AUC:** ~0.975 | **Test PR-AUC:** ~0.847
- [x] Saved learning curves and best model checkpoint.
- [x] Drafted Methodology and Preliminary Results sections of midterm report.

---

## Week 3 (Planned): Hyperparameter Tuning & Imbalance Ablations

### Planned — Student A  — Preprocessing Finalization
- [x] Finalized preprocessing notebook for reproducibility (clean, commented, all cells run top-to-bottom).
- [ ] Run ablation: with SMOTE vs without SMOTE vs class weighting only. 

### Planned — Student B — Classical Models Tuning
- [ ] Grid search for XGBoost (max_depth, n_estimators, learning_rate).
- [ ] Try cost-sensitive threshold tuning on classical models.
- [ ] Evaluate at different operating points (precision/recall trade-off).

### Planned — Student C — Deep Learning Ablations
- [x] Implemented 5-layer MLP (FraudMLP5Layer: 256→128→64→32→1) for depth ablation.
- [x] Implemented Focal Loss (gamma=2.0, alpha=0.25) as alternative to weighted BCE.
- [x] Restored weighted BCE alongside Focal Loss for side-by-side ablation comparison.
- [x] Added ablation loop across 3 configs: 4-Layer+Weighted BCE, 4-Layer+Focal Loss, 5-Layer+Weighted BCE.
- [x] Generated full comparison table across all models (full_model_comparison.csv + bar chart).

---

## Week 4 (Planned): Final Report & Presentation

### Planned — All Students
- [ ] Final code cleanup and documentation.
- [ ] Complete final IEEE report.
- [ ] Record / rehearse 20-minute presentation.
- [ ] Tidy GitHub: clean README, remove temporary files, seed all scripts.
- [ ] Final reproducibility check (run all scripts from scratch).

---

## Key Decisions Log

| Date   | Decision | Reason | Made By |
|--------|----------|--------|---------|
| Week 1 | SMOTE applied only to training data | Prevent data leakage into val/test | Student A |
| Week 1 | 70/10/20 train/val/test split | Standard practice + stratified | Student A |
| Week 2 | XGBoost uses `scale_pos_weight` (not SMOTE) | Test both imbalance strategies | Student B |
| Week 2 | MLP uses weighted BCE loss | Equivalent to cost-sensitive learning | Student C |
| Week 2 | PR-AUC as primary metric | ROC-AUC misleading on severe imbalance | Team |
| Week 3 | Focal Loss added alongside weighted BCE | Ablation: compare loss functions on imbalanced data | Student C |
| Week 3 | 5-layer MLP defined for depth ablation | Test if deeper architecture improves PR-AUC | Student C |

---

## Issues & Resolutions

| Issue | Resolution | Resolved By |
|-------|------------|-------------|
| SMOTE slow on full dataset | Reduced k_neighbors, ran on separate split | Student A |
| XGBoost training time > 8 min | Limited n_estimators=300, used subsample=0.8 | Student B |
| MLP overfit after epoch 40 | Added ReduceLROnPlateau + Dropout | Student C |
| weighted_bce removed accidentally | Restored before focal_loss definition, both kept for ablation | Student C |
| Ablation loop runtime ~15–25 min on CPU | Expected — 3 configs × 50 epochs on 398k samples | Student C |

---

## GenAI Use Disclosure

Per course policy, we disclose the following uses of GenAI tools (Claude/ChatGPT):
- Used for concept clarification (e.g., "explain SMOTE and data leakage").
- Used for grammar checks in report writing.
- Used for code structuring assistance (e.g., ablation loop design, CSV saving).
- **Not used** for generating experimental results or tables.
- **Not used** for generating code that was submitted without understanding.