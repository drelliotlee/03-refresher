import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = df[['height', 'minute', 'income']]
y = df['binary_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess_pipeline = Pipeline([
    # other pre-processing steps
    
    ('scaling', ColumnTransformer([
        ('standard', StandardScaler(), ['height']),  # Normally distributed data → mean=0, std=1
        ('minmax', MinMaxScaler(), ['minute']),      # Uniformly distributed data → [0,1]
        ('robust', RobustScaler(), ['income'])       # Data with outliers → median=0, IQR=1
    ])),
    
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train_processed = preprocess_pipeline.transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_processed, y_train)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)

# MODELS THAT DON'T NEED SCALING:
# Tree-based models (Decision Trees, Random Forest, XGBoost, LightGBM)
# Trees split on thresholds, so scale doesn't matter


