import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Train-Test Split (Key Boundary)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """Clips outliers to IQR bounds instead of removing rows"""
    
    def __init__(self, factor=1.5):
        self.factor = factor
        
    def fit(self, X, y=None):
        """Calculate IQR bounds on training data"""
        X_array = np.array(X)
        Q1 = np.percentile(X_array, 25, axis=0)
        Q3 = np.percentile(X_array, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound_ = Q1 - self.factor * IQR
        self.upper_bound_ = Q3 + self.factor * IQR
        return self
    
    def transform(self, X):
        """Clip values"""
        X_array = np.array(X)
        X_clipped = np.clip(X_array, self.lower_bound_, self.upper_bound_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_clipped, columns=X.columns, index=X.index)
        return X_clipped

class ZScoreOutlierClipper(BaseEstimator, TransformerMixin): 
    def __init__(self, threshold=3):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        """Calculate mean and std on training data"""
        X_array = np.array(X)
        self.mean_ = np.mean(X_array, axis=0)
        self.std_ = np.std(X_array, axis=0)
        self.lower_bound_ = self.mean_ - self.threshold * self.std_
        self.upper_bound_ = self.mean_ + self.threshold * self.std_
        return self
    
    def transform(self, X):
        """Clip values"""
        X_array = np.array(X)
        X_clipped = np.clip(X_array, self.lower_bound_, self.upper_bound_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_clipped, columns=X.columns, index=X.index)
        return X_clipped

preprocess_pipeline = Pipeline([
    # other pre-processing steps
    ('outlier_clipper', IQROutlierClipper(factor=1.5))
    # other pre-processing steps
])

preprocess_pipeline = Pipeline([
    # other pre-processing steps
    ('outlier_clipper', ZScoreOutlierClipper(threshold=3))
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train = preprocess_pipeline.transform(X_train)
X_test = preprocess_pipeline.transform(X_test)