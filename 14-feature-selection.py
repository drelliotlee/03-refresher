import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()   # Calculate correlation matrix
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 
        self.columns_to_drop_ = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.threshold)]
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_)

# Using Feature Selection within a larger Pipeline

preprocess_pipeline = Pipeline([
    # other pre-processing steps

    # --- 1. Remove columns of low-variance (univariately) ---
    # Rationale: Low-variance columns provide little information for distinguishing between targets.
    ('variance_threshold', VarianceThreshold(threshold=0.01)),
    
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
    ('select_k_best', SelectKBest(score_func=f_classif, k=10)),
    ('select_k_best', SelectKBest(score_func=f_regression, k=10)),
    ('select_k_best', SelectKBest(score_func=chi2, k=10)),
    ('select_k_best', SelectKBest(score_func=mutual_info_classif, k=10)),
    ('select_k_best', SelectKBest(score_func=mutual_info_regression, k=10)),
    
    # --- 3. Correlation-based feature removal ---
    # Calculate Pearson correlation matrix: corr(i,j) = Σ(xᵢ - x̄ᵢ)(xⱼ - x̄ⱼ) / √[Σ(xᵢ - x̄ᵢ)² · Σ(xⱼ - x̄ⱼ)²]
    # 0 = no correlation, -1/+1 = perfect correlation
    # Rationale: Correlated features provide similar information, drop if above 0.95 threshold
    # Cons: x_1 and x_2 may be correlated but the small residual times they differ may be highly predictive of y
    ('correlation_dropper', CorrelationDropper(threshold=0.95)),
    
    # --- 4. L1 Regularization (Lasso) ---
    # Fit a linear regression on (X_train,y_train) but with extra penalty term
    # Normal regression:  minimize (1/100) Σᵢ₌₁¹⁰⁰ (yᵢ - (β₀ + β₁·x₁ᵢ + β₂·x₂ᵢ))²
    # Lasso regression:   minimize (1/100) Σᵢ₌₁¹⁰⁰ (yᵢ - (β₀ + β₁·x₁ᵢ + β₂·x₂ᵢ))²  +  α( |β₁| + |β₂| )
    # Mathematical properties make (βᵢ = exactly 0) far preferred to (βᵢ = very small value)
    # Therefore lasso minimization is quasi-feature selection
    # Intuition: removes features with little LINEAR predictive power, even if have other nonlinear relationships
    ('lasso_selection', SelectFromModel(
        Lasso(alpha=0.01, random_state=42),
        threshold='median'
    )),
    
    # --- 5. Tree-based feature importance  ---
    # Just fit RF using all features (which also returns array feature_importances_ ex. [0.35, 0.28, 0.15, 0.12, 0.10, 0.00])
    # Only keep the features whose importance > mean/median importance
    ('rf_importance', SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold='mean'
    )),
    
    # --- 6. Recursive Feature Elimination (RFE) ---
    # Rationale: considers feature interactions, more robust than univariate methods
    # Intuition: train on all features, remove 'step' least important feature, re-train, repeat until n_features_to_select left
    ('rfe',RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        n_features_to_select=10,
        step=1
    )),
    
    # --- 7. RFE with Cross-Validation (RFECV) ---
    # RFE too simple, used a fixed arbitrary n_features_to_select              -> tries N, N-step, N-2*step, ...
    # RFE overfitted, used all of X_train and didn't validate on unknown data  -> splits X-train into 5 folds: 4 train, 1 val, repeat 5x
    # 
    # for numFeatures n = N, N-step, N-2*step....
    #     run RFE to select n features
    #     for fold f = 1 to 5:
    #         train RF (not RFE) on 4 train folds and score on 1 val fold → score_f
    #     score_n = average(score_f)
    # Best features = features from max{score_n}
    #
    # Cons: suppose 15 starting features, step=1, 5-fold CV, min_features_to_select=5
    #       RFE (15→10) trains 5 RF models (one per iteration)
    #       RFECV trains 66 (RFE: 1+2+3+...+11) + 55 (CV: 11 counts × 5 folds) = 121 RF models
    ('rfecv', RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        step=1,
        cv=5,
        scoring='accuracy',
        min_features_to_select=5
    )),
    
    # --- 8. Ensemble Feature Selection (consensus voting) ---
    # Run multiple feature selectors and keep features selected by >= min_votes methods
    # More robust than single method, reduces chance of dropping important features
    ('ensemble_selector', FeatureSelectionEnsemble(
        selectors=[
            ('lasso', SelectFromModel(Lasso(alpha=0.01, random_state=42), threshold='median')),
            ('rf', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='mean')),
            ('kbest', SelectKBest(score_func=f_classif, k=10))
        ],
        min_votes=2  # Keep features if at least 2 methods selected them
    ))
    
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train_processed = preprocess_pipeline.transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)
