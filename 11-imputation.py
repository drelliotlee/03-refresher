import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import MissingIndicator
from sklearn.compose import ColumnTransformer
from sktime.transformations.series.impute import Imputer  # Time series imputer

# Train-Test Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocess_pipeline = Pipeline([
    # other pre-processing steps
    
    ('imputation', ColumnTransformer([
        # Simple imputation for numeric columns 1-3
        # also demonstrating how to add missing indicator columns
        ('simple_impute_w_indicator', FeatureUnion([
            ('simple_imputer', SimpleImputer(strategy='median')),
            ('missing_indicator', MissingIndicator())
        ]), ['numeric1', 'numeric2', 'numeric3']),
        #
        # Categorical imputation (mode) for categorical columns
        ('categorical_imputer', SimpleImputer(strategy='most_frequent'), ['categorical1', 'categorical2']),
        #
        # KNN imputation for numeric columns 4-6
        ('knn_imputer', KNNImputer(n_neighbors=5, weights='distance'), ['numeric4', 'numeric5', 'numeric6']),
        #
        # MICE imputation for numeric columns 7-9
        ('mice_imputer', IterativeImputer(max_iter=10, random_state=42), ['numeric7', 'numeric8', 'numeric9']),
        #
        # Time series forward fill for time series columns
        ('time_imputer', Imputer(method='ffill'), ['timeseries1', 'timeseries2']),
    ], remainder='passthrough')),
    
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train_processed = preprocess_pipeline.transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)


# ============================================
# MICE ALGORITHM:
# ============================================
# Note: only works on numerical data!
# Initialize all missing values with column-wise mean
#
# For up to max_iter iterations (or until values converge):
#    For each column i with missing values:
#       Predict missing val in col i <- linear regression(all other columns)


# ============================================
# Predictive imputation using other features
# ============================================
# ignore this method. it requires:
# 1. for each column i, filter down to only rows w missing values
# 2. for all other columns not i, fill missing values with simple imputation (mean/mode)
# 3. train a classifier to predict column i using all other columns as features
#    with zero tuning, just arbitrary choice of model family and hyperparameters
# 4. use trained model to predict missing values in column i
# 5. repeat for all categorical columns
# Just seems too too much arbitrary design choices, and too computationally expensive.


# ============================================
# SENSITIVITY ANALYSIS:
# ============================================
# Impute with multiple methods, train model on each, compare performance.
# But very computationally expensive, rarely done in practice.