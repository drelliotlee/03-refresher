import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Assume we have some arbitrary df with features and target
X_train, X_test = ...  # features
y_train, y_test = ...  # target

# ============================================================================
# 1. LINEAR REGRESSION (for continuous target)
# ============================================================================
# Math: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
#       Minimizes: Σ(y - ŷ)²  (ordinary least squares)
# Intuition: Find best-fit straight line/hyperplane through data points
# Pros: Fast, interpretable, works well with linear relationships
# Cons: Assumes linearity, sensitive to outliers, can't model complex patterns
# Use when: Relationship is linear, need interpretability, fast predictions needed

model = LinearRegression(
    fit_intercept=True,  # ← Whether to calculate intercept β₀ (almost always True)
    n_jobs=-1  # ← Use all CPU cores for parallelization
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================================
# 2. LOGISTIC REGRESSION (for binary/multiclass classification)
# ============================================================================
# Math: P(y=1|x) = 1/(1 + e^(-z)) where z = β₀ + β₁x₁ + ... + βₙxₙ  (sigmoid)
#       Minimizes: -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]  (cross-entropy loss)
# Intuition: Linear regression + sigmoid to squeeze output between 0 and 1 for probabilities
# Pros: Fast, interpretable, outputs calibrated probabilities, works well when decision boundary is linear
# Cons: Assumes linear decision boundary, can underfit complex patterns
# Use when: Need probabilities, interpretability important, linear separability exists

model = LogisticRegression(
    penalty='l2',  # ← Regularization type: 'l1' (Lasso), 'l2' (Ridge), 'elasticnet', or None
    C=1.0,  # ← Inverse of regularization strength (smaller = more regularization)
    solver='lbfgs',  # ← Optimization algorithm: 'lbfgs', 'liblinear', 'saga', 'newton-cg'
    max_iter=1000,  # ← Maximum iterations for convergence
    class_weight='balanced',  # ← Auto-adjust weights for imbalanced classes (or dict, or None)
    multi_class='auto',  # ← 'ovr' (one-vs-rest), 'multinomial', or 'auto'
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 3. XGBoost (Extreme Gradient Boosting)
# ============================================================================
# Math: F(x) = f₀ + η·f₁(x) + η·f₂(x) + ... + η·fₘ(x)
#       Each tree fₜ fits residuals of F(x) so far
#       Minimizes: Loss + Ω(f) where Ω = γT + ½λΣwⱼ² (regularization on tree complexity)
# Intuition: Build trees sequentially, each correcting errors of previous trees
# Pros: High accuracy, handles missing values, built-in regularization, feature importance
# Cons: Slow to train, many hyperparameters, can overfit, needs tuning
# Use when: Tabular data, need high accuracy, have time for tuning

model = xgb.XGBClassifier(  # or XGBRegressor for regression
    n_estimators=100,  # ← Number of boosting rounds (trees to build)
    learning_rate=0.1,  # ← η: shrinkage applied to each tree (smaller = more robust but needs more trees)
    max_depth=6,  # ← Maximum depth of each tree (deeper = more complex, higher risk of overfit)
    min_child_weight=1,  # ← Minimum sum of instance weights needed in a child (higher = more conservative)
    subsample=0.8,  # ← Fraction of samples to use for each tree (prevents overfitting)
    colsample_bytree=0.8,  # ← Fraction of features to use for each tree (prevents overfitting)
    gamma=0,  # ← γ: minimum loss reduction required to make split (higher = more conservative)
    reg_alpha=0,  # ← L1 regularization on weights (Lasso)
    reg_lambda=1,  # ← λ: L2 regularization on weights (Ridge)
    scale_pos_weight=1,  # ← Balance of positive/negative weights (for imbalanced data)
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 4. LightGBM (Light Gradient Boosting Machine)
# ============================================================================
# Math: Same boosting framework as XGBoost but uses leaf-wise tree growth
#       Instead of level-wise (all leaves at same depth), grows leaf with max loss reduction
# Intuition: Faster XGBoost via leaf-wise growth + histogram binning of features
# Pros: Very fast training, low memory, handles large datasets, high accuracy
# Cons: Can overfit on small datasets, sensitive to hyperparameters
# Use when: Large datasets, speed is critical, tabular data

model = lgb.LGBMClassifier(  # or LGBMRegressor for regression
    n_estimators=100,  # ← Number of boosting rounds
    learning_rate=0.1,  # ← Shrinkage rate (smaller = more robust)
    max_depth=-1,  # ← Maximum tree depth (-1 = no limit, uses num_leaves instead)
    num_leaves=31,  # ← Max number of leaves in one tree (2^max_depth for balanced tree)
    min_child_samples=20,  # ← Minimum number of samples needed in a leaf (higher = more conservative)
    subsample=0.8,  # ← Fraction of samples for each tree (also called bagging_fraction)
    colsample_bytree=0.8,  # ← Fraction of features for each tree (also called feature_fraction)
    reg_alpha=0,  # ← L1 regularization
    reg_lambda=0,  # ← L2 regularization
    class_weight='balanced',  # ← Handle imbalanced classes
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 5. CatBoost (Categorical Boosting)
# ============================================================================
# Math: Same boosting framework but uses ordered boosting to prevent target leakage
#       For categorical features, uses ordered target statistics instead of one-hot encoding
# Intuition: XGBoost optimized for categorical features with automatic encoding
# Pros: Handles categoricals natively, less tuning needed, robust to overfitting
# Cons: Slower than LightGBM, still slower than sklearn models
# Use when: Many categorical features, want less tuning, tabular data

model = cb.CatBoostClassifier(  # or CatBoostRegressor for regression
    iterations=100,  # ← Number of boosting rounds (same as n_estimators)
    learning_rate=0.1,  # ← Shrinkage rate
    depth=6,  # ← Maximum tree depth
    l2_leaf_reg=3,  # ← L2 regularization coefficient (higher = more regularization)
    border_count=254,  # ← Number of splits for numerical features (higher = finer splits)
    bagging_temperature=1,  # ← Bayesian bootstrap intensity (0 = no bootstrap, higher = more aggressive)
    random_strength=1,  # ← Amount of randomness for scoring splits (higher = more random)
    auto_class_weights='Balanced',  # ← Automatically balance class weights: 'Balanced', 'SqrtBalanced', or None
    cat_features=None,  # ← Indices or names of categorical features (None = auto-detect)
    verbose=False,  # ← Suppress training output
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 6. SVR (Support Vector Regression)
# ============================================================================
# Math: min ½||w||² + C·Σmax(0, |yᵢ - ŷᵢ| - ε)
#       Creates ε-tube around predictions, only penalizes errors outside tube
#       Uses kernel trick: K(x,x') to map to higher dimensions for non-linear patterns
# Intuition: Find hyperplane that fits data within ε margin, support vectors define boundary
# Pros: Effective in high dimensions, memory efficient (only stores support vectors), works with non-linear data
# Cons: Slow on large datasets, sensitive to feature scaling, hard to interpret
# Use when: High-dimensional data, small-to-medium datasets, need non-linear relationships

model = SVR(
    kernel='rbf',  # ← Kernel type: 'linear', 'poly', 'rbf' (radial basis), 'sigmoid'
    C=1.0,  # ← Penalty for errors outside ε-tube (higher = less tolerance for errors)
    epsilon=0.1,  # ← ε: width of tube around predictions where no penalty is given
    gamma='scale',  # ← Kernel coefficient (higher = more complex boundary): 'scale', 'auto', or float
    degree=3  # ← Degree of polynomial kernel (only used if kernel='poly')
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================================
# 7. SVC (Support Vector Classification)
# ============================================================================
# Math: min ½||w||² + C·Σξᵢ  subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
#       Maximizes margin between classes while allowing some misclassification (slack ξ)
#       Uses kernel trick: K(x,x') to map to higher dimensions for non-linear boundaries
# Intuition: Find maximum-margin hyperplane that separates classes, support vectors define boundary
# Pros: Effective in high dimensions, works with non-linear boundaries, memory efficient
# Cons: Very slow on large datasets, no probability estimates by default, sensitive to scaling
# Use when: High-dimensional data, small-to-medium datasets, clear margin of separation exists

model = SVC(
    kernel='rbf',  # ← Kernel type: 'linear', 'poly', 'rbf' (radial basis), 'sigmoid'
    C=1.0,  # ← Penalty for misclassification (higher = less tolerance, may overfit)
    gamma='scale',  # ← Kernel coefficient (higher = more complex boundary): 'scale', 'auto', or float
    degree=3,  # ← Degree of polynomial kernel (only used if kernel='poly')
    class_weight='balanced',  # ← Adjust weights for imbalanced classes
    probability=True,  # ← Enable probability estimates (slower but needed for predict_proba)
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Only available if probability=True
