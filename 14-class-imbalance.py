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
from sklearn.metrics import average_precision_score

# 1. Stratified Train/Test Split
# Ensures that both train and test sets have the same class distribution as the original dataset
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # ← this option here
)

preprocess_pipeline = Pipeline([    # ← Imblearn's Pipeline instead of sklearn's Pipeline (can't add/remove rows)
    # other pre-processing steps
    
    # 2. SMOTE
    # takes a real row, finds k nearest neighbors, and creates a synthetic new row
    # which is a linear interpolation between real row and random 1 of 5 knn neighbors 
    ('smote', SMOTE(
        sampling_strategy='auto',  # ← create new rows until all classes equal # rows
        k_neighbors=5,             # ← synthetic row is interpolation of 5 nearest neighbors
        random_state=42
    )),
    
    # 3. SMOTENC
    # SMOTE can only handle continuous, SMOTENC handles categorical and boolean
    # "neighbor" still only uses continuous features, but
    # "interpolation" of categorical means randomly pick 1 of 5 neighbors values
    ('smotenc', SMOTENC(
        categorical_features=[0, 2, 5],  # ← indices of categorical/boolean columns
        sampling_strategy='auto',
        k_neighbors=5,
        random_state=42
    )),
    
    # 4. Random Over-Sampling
    # Simply duplicates rows from smaller classes until same # rows as largest class
    ('random_oversampler', RandomOverSampler(
        sampling_strategy='auto',  # ← create new rows until all classes equal # rows
        random_state=42
    )),
    
    # 5. Random Under-Sampling
    # Randomly removes samples from majority class until balanced
    ('random_undersampler', RandomUnderSampler(
        sampling_strategy='auto',  # ← remove rows until all classes equal # rows
        random_state=42
    )),
    
    # 6. SMOTE + Tomek links cleaning
    # first, creates synthetic rows via SMOTE
    # then, finds pairs of rows from different classes that are each other's nearest neighbors
    # then removes the row from the majority class in each pair so there is more separation in feature space
    ('smote_tomek', SMOTETomek(random_state=42)), 
    
    # 7. SMOTE + ENN cleaning
    # first, creates synthetic rows via SMOTE
    # then, for every row, find its 3 nearest KNN neighbors
    # if the majority of those neighbors have a different class, drop that row
    ('smote_enn', SMOTEENN(random_state=42))
    
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train_processed = preprocess_pipeline.transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)

# 8. Class Weights 
# Original loss: L = Σᵢ loss(yᵢ, ŷᵢ)      ←  All errors treated equally
# Weighted loss: L = Σᵢ wᵢ·loss(yᵢ, ŷᵢ)   ←  Penalize mistakes on minority class more heavily
#                       wᵢ = #total_rows / (#classes × #rowsᵢ)
# Example: 90% class 0, 10% class 1, errors on class 1 cost ~9x more
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # ← This option here for regressions
)
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← This option here for forests
    random_state=42
)
model.fit(X_train_processed, y_train)

# 9. Evaluation : PR-AUC (Precision-Recall AUC) is best for imbalanced data
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
test_score = average_precision_score(y_test, y_test_proba)