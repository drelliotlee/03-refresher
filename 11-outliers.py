import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

# create a sample dataset
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'feature1': np.random.normal(50, 10, n_samples),
    'feature2': np.random.normal(100, 20, n_samples),
    'feature3': np.random.normal(75, 15, n_samples),
    'target': np.random.randint(0, 2, n_samples)
})

df.loc[np.random.choice(df.index, 20), 'feature1'] = np.random.uniform(100, 150, 20) # Add some outliers
df.loc[np.random.choice(df.index, 20), 'feature2'] = np.random.uniform(200, 300, 20) # Add some outliers

# Train-Test Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# METHOD 1: IQR (Interquartile Range) Method
# ============================================ 
def detect_outliers_iqr(df, columns): 
	#returns a boolean mask where True indicates an outlier.
    outlier_mask = pd.Series(False, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask = outlier_mask | col_outliers 
    return outlier_mask

# Remove outliers from training set
outlier_mask = detect_outliers_iqr(X_train, ['feature1', 'feature2', 'feature3'])
X_train_cleaned = X_train[~outlier_mask] 
y_train_cleaned = y_train[~outlier_mask] 

# Train model on IQR-cleaned data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_cleaned, y_train_cleaned) 

# we do NOT apply outlier mask to test set! 
y_test_predicted = model.predict(X_test)

# ============================================
# METHOD 2: Z-Score Method
# ============================================
def detect_outliers_zscore(df, columns, threshold=3):
    # Returns boolean mask where True indicates an outlier.
    outlier_mask = pd.Series(False, index=df.index)
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        col_outliers = z_scores > threshold
        outlier_mask = outlier_mask | col_outliers
    return outlier_mask

# Remove outliers from training set
outlier_mask = detect_outliers_zscore(X_train, ['feature1', 'feature2', 'feature3'], threshold=3)
X_train_cleaned = X_train[~outlier_mask]
y_train_cleaned = y_train[~outlier_mask]

# Train model on Z-score-cleaned data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_cleaned, y_train_cleaned)

# we do NOT apply outlier removal to test set! 
y_test_predicted = model.predict(X_test)

