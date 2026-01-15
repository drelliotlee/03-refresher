import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Train-Test Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# Any dtype: Create missing indicator column, then impute with other method
# ============================================
X_train_w_indicator = X_train.copy()
X_test_w_indicator = X_test.copy()

# Create binary indicator columns for missingness
for col in numeric_cols + categorical_cols:
    X_train_w_indicator[f'{col}_was_missing'] = X_train[col].isnull().astype(int)
    X_test_w_indicator[f'{col}_was_missing'] = X_test[col].isnull().astype(int)

# Then impute the original columns (ex. using simple mean/mode)
for col in numeric_cols:
    mean_value = X_train[col].mean()
    X_train_w_indicator[col].fillna(mean_value, inplace=True)
    X_test_w_indicator[col].fillna(mean_value, inplace=True)
for col in categorical_cols:
    mode_value = X_train[col].mode()[0]
    X_train_w_indicator[col].fillna(mode_value, inplace=True)
    X_test_w_indicator[col].fillna(mode_value, inplace=True)

# ============================================
# Time Series: Forward/Backward Fill 
# ============================================

# Important: Copy and sort by timestamp first!
X_train_imputed = X_train.copy().sort_values('timestamp')
X_test_imputed = X_test.copy().sort_values('timestamp')
# Forward fill - propagates last valid value forward ROW BY ROW
X_train_imputed[['numeric1', 'numeric2', 'numeric3']] = X_train_imputed[['numeric1', 'numeric2', 'numeric3']].fillna(method='ffill')
X_test_imputed[['numeric1', 'numeric2', 'numeric3']] = X_test_imputed[['numeric1', 'numeric2', 'numeric3']].fillna(method='ffill')
# Backward fill - propagates next valid value backward ROW BY ROW
X_train_imputed[['numeric1', 'numeric2', 'numeric3']] = X_train_imputed[['numeric1', 'numeric2', 'numeric3']].fillna(method='bfill')
X_test_imputed[['numeric1', 'numeric2', 'numeric3']] = X_test_imputed[['numeric1', 'numeric2', 'numeric3']].fillna(method='bfill')

# ============================================
# Numerical #1: KNN Imputation
# ============================================
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

# KNN doesn't handle categorical columns, so need to encode them first
numeric_cols = ['numeric1', 'numeric2', 'numeric3']
categorical_cols = ['categorical1', 'categorical2']

# But can't encode NaNs, so fill them with a new 'MISSING' category 
X_train_imputed['categorical1'].fillna('MISSING', inplace=True)
X_test_imputed['categorical1'].fillna('MISSING', inplace=True)

X_train_imputed['categorical2'].fillna('MISSING', inplace=True)
X_test_imputed['categorical2'].fillna('MISSING', inplace=True)

# Encode categorical columns for KNN
encoder1 = LabelEncoder()
X_train_imputed['categorical1_encoded'] = encoder1.fit_transform(X_train_imputed['categorical1'])
X_test_imputed['categorical1_encoded'] = encoder1.transform(X_test_imputed['categorical1'])

encoder2 = LabelEncoder()
X_train_imputed['categorical2_encoded'] = encoder2.fit_transform(X_train_imputed['categorical2'])
X_test_imputed['categorical2_encoded'] = encoder2.transform(X_test_imputed['categorical2'])

# Now ready to use KNN 
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
feature_cols = ['numeric1', 'numeric2', 'numeric3', 'categorical1_encoded', 'categorical2_encoded']
knn_imputer.fit(X_train_imputed[feature_cols])                                        # Fit on training data
X_train_imputed[feature_cols] = knn_imputer.transform(X_train_imputed[feature_cols])  # Transform (AKA impute) training data
X_test_imputed[feature_cols] = knn_imputer.transform(X_test_imputed[feature_cols])    # Transform (AKA impute) test data

# ============================================
# Numerical #2: MICE 
# ============================================
# MICE ALGORITHM:
# Note: only works on numerical data!
# Initialize all missing values with column-wise mean
#
# For up to max_iter iterations (or until values converge):
#    For each column i with missing values:
#       Predict missing val in col i <- linear regression(all other columns)

X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

mice_imputer = IterativeImputer(
    estimator=None,          # None = BayesianRidge a form of linear regression. Could have used RandomForestRegressor()
    max_iter=10, 
    random_state=42,
    initial_strategy='mean'
)

mice_imputer.fit(X_train[numeric_cols])                                        # Fit on training data
X_train_imputed[numeric_cols] = mice_imputer.transform(X_train[numeric_cols])  # Transform training data
X_test_imputed[numeric_cols] = mice_imputer.transform(X_test[numeric_cols])    # Transform test data

# ============================================
# Categorical #1: Simple Mode
# ============================================
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

categorical_cols = ['categorical1', 'categorical2']
for col in categorical_cols:
    mode_value = X_train[col].mode()[0]
    X_train_imputed[col].fillna(mode_value, inplace=True)
    X_test_imputed[col].fillna(mode_value, inplace=True)

# ============================================
# Categorical #2: weighted sampling from observed distribution
# ============================================
X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

for col in categorical_cols:
    histogram = X_train[col].value_counts(normalize=True)
    # histogram looks like:
    # Index     |  Values 
    # ------------------------------
    # 'A'       |  0.40
    # 'B'       |  0.35
    # 'C'       |  0.20
    # 'D'       |  0.05
    
    # create mask of where missing values are
    missing_idx_train = X_train_imputed[X_train_imputed[col].isna()].index
    missing_idx_test = X_test_imputed[X_test_imputed[col].isna()].index
    
    # impute missing values by weighted sampling from histogram
    if len(missing_idx_train) > 0:
        X_train_imputed.loc[missing_idx_train, col] = np.random.choice(histogram.index, size=len(missing_idx_train), p=histogram.values)
    if len(missing_idx_test) > 0:
        X_test_imputed.loc[missing_idx_test, col] = np.random.choice(histogram.index, size=len(missing_idx_test), p=histogram.values)

# ============================================
# Categorical #3: Predictive imputation using other features
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