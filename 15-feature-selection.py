import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier

# --- 1. First remove columns of low-variance (univariately) ---
# Rationale: Low-variance columns provide little information for distinguishing between targets.
variance_selector = VarianceThreshold(threshold=0.01)
variance_selector.fit(X_train)
X_train_var = variance_selector.transform(X_train)
X_test_var = variance_selector.transform(X_test)
print(f"Features kept after variance threshold: {X_train.columns[variance_selector.get_support()].tolist()}")

# --- 2. SelectKBest (also univariate) ---
# Rationale:    want to keep columns which have unilateral relationship with target variable
# Drawbacks:    ignoring feature interactions, May select correlated features that provide similar information
# in practice, people rarely optimize k and just pick a number
#
# ┌──────────────────────────────┬────────────────────┬────────────────────┐
# │                              │ Categorical Target │ Continuous Target  │
# ├──────────────────────────────┼────────────────────┼────────────────────┤
# │ Continuous features          │ f_classif          │ f_regression       │
# │ Any/mixed features *         │ mutual_info_classif│ mutual_info_regr.  │
# │ Non-neg categorical features │ chi2               │ N/A                │
# └──────────────────────────────┴────────────────────┴────────────────────┘
# *mutual_info uses KNN first to turn continuous into categorical

k_best_selector = SelectKBest(score_func=f_classif, k=10)  
k_best_selector = SelectKBest(score_func=f_regression, k=10)
k_best_selector = SelectKBest(score_func=chi2, k=10)
k_best_selector = SelectKBest(score_func=mutual_info_classif, k=10)
k_best_selector = SelectKBest(score_func=mutual_info_regression, k=10)

k_best_selector.fit(X_train, y_train)
X_train_kbest = k_best_selector.transform(X_train)
X_test_kbest = k_best_selector.transform(X_test)
print(f"Features kept after SelectKBest: {X_train.columns[k_best_selector.get_support()].tolist()}")
print(f"Feature scores from SelectKBest: {k_best_selector.scores_}")

# ============================================================================
# 2. WRAPPER METHODS - Use model performance to select features
# ============================================================================

# --- Recursive Feature Elimination (RFE) ---
rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_estimator, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
print(f"Features kept after RFE: {X_train.columns[rfe.get_support()].tolist()}")
feature_ranking = rfe.ranking_

# --- RFE with Cross-Validation (more robust) ---
rfecv = RFECV(
    estimator=rf_estimator, 
    step=1, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    min_features_to_select=5
)
rfecv.fit(X_train, y_train)
X_train_rfecv = rfecv.transform(X_train)
X_test_rfecv = rfecv.transform(X_test)
optimal_features = rfecv.n_features_
print(f"Features kept after RFECV: {X_train.columns[rfecv.get_support()].tolist()}")

# ============================================================================
# 3. EMBEDDED METHODS - Feature selection during model training
# ============================================================================

# --- L1 Regularization (Lasso) - drives coefficients to zero ---
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train, y_train)
lasso_selector = SelectFromModel(lasso, prefit=True, threshold='median')
X_train_lasso = lasso_selector.transform(X_train)
X_test_lasso = lasso_selector.transform(X_test)
print(f"Features kept after Lasso: {X_train.columns[lasso_selector.get_support()].tolist()}")

# --- Tree-based Feature Importance ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
importances = rf_model.feature_importances_

# Select features based on importance threshold
importance_selector = SelectFromModel(rf_model, prefit=True, threshold='mean')
X_train_rf = importance_selector.transform(X_train)
X_test_rf = importance_selector.transform(X_test)
print(f"Features kept after Random Forest importance: {X_train.columns[importance_selector.get_support()].tolist()}")

# ============================================================================
# 4. CORRELATION-BASED SELECTION
# ============================================================================

# Remove highly correlated features
correlation_matrix = X_train.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
X_train_uncorrelated = X_train.drop(columns=to_drop)
X_test_uncorrelated = X_test.drop(columns=to_drop)

# ============================================================================
# 5. COMBINING METHODS
# ============================================================================

# Get features selected by multiple methods (ensemble approach)
from collections import Counter

selected_features_kbest = X_train.columns[k_best_selector.get_support()]
selected_features_rfe = X_train.columns[rfe.get_support()]
selected_features_rf = X_train.columns[importance_selector.get_support()]

all_selected = (
    list(selected_features_kbest) + 
    list(selected_features_rfe) + 
    list(selected_features_rf)
)
feature_counts = Counter(all_selected)
# Keep features selected by at least 2 methods
consensus_features = [f for f, count in feature_counts.items() if count >= 2]

X_train_consensus = X_train[consensus_features]
X_test_consensus = X_test[consensus_features]
print(f"Consensus features (selected by 2+ methods): {consensus_features}")

# ============================================================================
# 6. EVALUATION - Compare model performance with different feature sets
# ============================================================================

from sklearn.model_selection import cross_val_score

def evaluate_features(X_train_subset, y_train_subset, method_name):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X_train_subset, y_train_subset, cv=5)
    print(f"{method_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean()

# Compare all methods
results = {
    'Original': evaluate_features(X_train, y_train, 'Original'),
    'SelectKBest': evaluate_features(X_train_kbest, y_train, 'SelectKBest'),
    'RFE': evaluate_features(X_train_rfe, y_train, 'RFE'),
    'RFECV': evaluate_features(X_train_rfecv, y_train, 'RFECV'),
    'Random Forest': evaluate_features(X_train_rf, y_train, 'Random Forest'),
    'Consensus': evaluate_features(X_train_consensus, y_train, 'Consensus')
}

# Select best method based on performance
best_method = max(results, key=results.get)
print(f"\nBest feature selection method: {best_method}")

# ============================================================================
# FINAL STEP: Use selected features for final model training
# ============================================================================
# Choose your best performing feature set and proceed with model training
# X_train_final = X_train_rfecv  # or whichever performed best
# X_test_final = X_test_rfecv
