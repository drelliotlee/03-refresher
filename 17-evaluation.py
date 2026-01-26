# ============================================================
# Evaluation with predict_proba() and explicit metrics
# ============================================================
# Assumes: model already fitted, X_test_processed and y_test available

from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, root_mean_squared_error, auc

# BINARY CLASSIFICATION - ROC-AUC (INDUSTRY STANDARD)
# ROC curve plots (FPR, TPR) across all thresholds
# AUC Range: 0.5 (random) to 1.0 (perfect)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]           # y_test_proba is probabilities [0,1], not hard labels {0,1}
roc_auc = roc_auc_score(y_test, y_test_proba)

# BINARY CLASSIFICATION (IMBALANCED) - PRECISION-RECALL AUC
# Better than ROC-AUC for imbalanced data
# PR curve plots (recall, precision) across all thresholds
# Use when positive class is rare (e.g., 1% fraud, 0.1% cancer)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(recall_vals, precision_vals)

# REGRESSION - RMSE (Root Mean Squared Error)
# RMSE = sqrt(average((actual - predicted)²)
# Unlike MAE (mean avg error), penalizes large errors more heavily
# Unlike R², has interpretable units (same as target variable)
y_test_pred = model.predict(X_test_processed)                         # y_test_pred = hard-predicted values, not probabilities
rmse = root_mean_squared_error(y_test, y_test_pred)

# MULTICLASS CLASSIFICATION - F1 SCORE
# F1 = 2 * (precision * recall) / (precision + recall)
# Better for imbalanced classes than accuracy
y_test_pred = model.predict(X_test_processed)                         # y_test_pred = hard-predicted class labels
f1 = f1_score(y_test, y_test_pred, average='weighted')

# MULTICLASS CLASSIFICATION - ROC-AUC (One-vs-Rest)
# Treats multiclass as multiple binary problems: class 0 vs all, class 1 vs all, etc.
# Then averages the ROC-AUC scores
y_test_proba = model.predict_proba(X_test_processed)                  # y_test_proba = n samples x m classes probabilities, not hard labels
multiclass_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')

# MULTICLASS CLASSIFICATION - LOG LOSS (Cross-Entropy)
# Measures how well-calibrated probabilities are
# Lower is better (0 = perfect, higher = worse)
# Heavily penalizes confident wrong predictions
# Common in Kaggle competitions
from sklearn.metrics import log_loss
y_test_proba = model.predict_proba(X_test_processed)
logloss = log_loss(y_test, y_test_proba)

# ============================================================
# Understanding Precision vs Recall
# ============================================================
    
# PRECISION = TP/(TP+FP) = "Of my positive predictions, how many were correct?"
# WHEN TO USE: when FALSE POSITIVES are costly
# Example: Auditing taxpayers for fraud takes time and manpower
#
# RECALL = TP/(TP+FN) = "Of all actual positives, how many did I catch?"
# WHEN TO USE: when FALSE NEGATIVES are catastrophic  
# Example: Missing a bomb at airport security is deadly