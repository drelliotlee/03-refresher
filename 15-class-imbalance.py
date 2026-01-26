import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # ← IMPORTANT: Ensure train/test have same class ratio as original data
)

# Note we used imblearn's Pipeline instead of sklearn's Pipeline!
# Can do everything sklearn's Pipeline can do, but can add/remove rows
preprocess_pipeline = Pipeline([
    # other pre-processing steps
    
    # 1. SMOTE
    # takes a real row, finds k nearest neighbors, and creates a synthetic new row
    # which is a linear interpolation between real row and random 1 of 5 knn neighbors 
    ('smote', SMOTE(
        sampling_strategy='auto',  # ← create new rows until all classes equal # rows
        k_neighbors=5,             # ← synthetic row is interpolation of 5 nearest neighbors
        random_state=42
    )),
    
    # 2. SMOTENC
    # SMOTE can only handle continuous, SMOTENC handles categorical and boolean
    # "neighbor" still only uses continuous features, but
    # "interpolation" of categorical means randomly pick 1 of 5 neighbors values
    ('smotenc', SMOTENC(
        categorical_features=[0, 2, 5],  # ← indices of categorical/boolean columns
        sampling_strategy='auto',
        k_neighbors=5,
        random_state=42
    )),
    
    # 3. Random Over-Sampling
    # Simply duplicates rows from smaller classes until same # rows as largest class
    ('random_oversampler', RandomOverSampler(
        sampling_strategy='auto',  # ← create new rows until all classes equal # rows
        random_state=42
    )),
    
    # 4. Random Under-Sampling
    # Randomly removes samples from majority class until balanced
    ('random_undersampler', RandomUnderSampler(
        sampling_strategy='auto',  # ← remove rows until all classes equal # rows
        random_state=42
    )),
    
    
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train_processed = preprocess_pipeline.transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)

# 10. Class Weights 
# Original loss: L = Σᵢ loss(yᵢ, ŷᵢ)      ←  All errors treated equally
# Weighted loss: L = Σᵢ wᵢ·loss(yᵢ, ŷᵢ)   ←  Penalize mistakes on minority class more heavily
#                       wᵢ = #total_rows / (#classes × #rowsᵢ)
# Example: 90% class 0, 10% class 1, errors on class 1 cost ~9x more
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # ← This option here
)
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← This option here
    random_state=42
)
model.fit(X_train_processed, y_train)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)









# ============================================================================
# METHOD 5: Combination (Under-sample majority + Over-sample minority)
# ============================================================================
# Often works better than either alone

from imblearn.combine import SMOTEENN, SMOTETomek

preprocess_pipeline_combined = ImbPipeline([
    # other pre-processing steps
    
    ('smote_tomek', SMOTETomek(random_state=42)),  # SMOTE + Tomek links cleaning
    
    # other pre-processing steps
])

# ============================================================================
# EVALUATION: Metrics for Imbalanced Datasets
# ============================================================================
# DON'T use accuracy - it's misleading for imbalanced data!
# Example: 95% class 0, 5% class 1 → always predicting 0 gives 95% accuracy!

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    average_precision_score,  # PR-AUC
    confusion_matrix,
    classification_report
)

y_test_pred = model.predict(X_test_processed)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]

# Metric 1: ROC-AUC (good, but can be optimistic on imbalanced data)
roc_auc = roc_auc_score(y_test, y_test_proba)

# Metric 2: PR-AUC (Precision-Recall AUC) - BEST for imbalanced data
pr_auc = average_precision_score(y_test, y_test_proba)

# Metric 3: F1 Score (harmonic mean of precision and recall)
f1 = f1_score(y_test, y_test_pred)

# Metric 4: Confusion Matrix (see actual TP, FP, TN, FN)
cm = confusion_matrix(y_test, y_test_pred)
# [[TN, FP],
#  [FN, TP]]

# Metric 5: Classification Report (precision, recall, F1 per class)
print(classification_report(y_test, y_test_pred))

# ============================================================================
# COMPARING METHODS
# ============================================================================

methods = {
    'No Resampling': (X_train, y_train),
    'SMOTE': (X_train_smote, y_train_smote),
    'Random Over-sampling': (X_train_over, y_train_over),
    'Random Under-sampling': (X_train_under, y_train_under),
    'Class Weights': (X_train, y_train)  # Uses weighted model
}

for method_name, (X_tr, y_tr) in methods.items():
    if method_name == 'Class Weights':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X_tr, y_tr)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    
    print(f"{method_name}: ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")