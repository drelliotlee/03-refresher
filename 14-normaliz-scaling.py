import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

data = {
    'height': np.random.normal(170,10,1000),
    'minute': np.random.uniform(0,60,1000),
    'income': np.random.lognormal(11.5,0.8,1000),
    'binary_outcome': np.random.randint(0,2,1000)
}

df = pd.DataFrame(data)

X = df[['height', 'minute', 'income']]
y = df['binary_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('standard', StandardScaler(), ['height']),  # Normally distributed data → mean=0, std=1
    ('minmax', MinMaxScaler(), ['minute']),      # Uniformly distributed data → [0,1]
    ('robust', RobustScaler(), ['income'])       # Data with outliers → median=0, IQR=1
])

pipeline = Pipeline([
    ('scaler', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train) #scale X_train, then fit model
score = pipeline.score(X_test, y_test) # transforms X_test with fitted scaler, then scores

# MODELS THAT DON'T NEED SCALING:
# Tree-based models (Decision Trees, Random Forest, XGBoost, LightGBM)
# Trees split on thresholds, so scale doesn't matter


