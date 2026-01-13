import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder, LeaveOneOutEncoder, BinaryEncoder, HashingEncoder

np.random.seed(42)

# CREATE SAMPLE DATA
data = {
    'color': np.random.choice(['red', 'blue', 'green', 'yellow'], 1000),  # Low cardinality
    'size': np.random.choice(['small', 'medium', 'large'], 1000),  # Ordered categories
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'] + [f'City_{i}' for i in range(50)], 1000),  # High cardinality
    'user_id': [f'user_{i}' for i in np.random.randint(0, 10000, 1000)],  # Very high cardinality, streaming
    'income': np.random.randint(30000, 120000, 1000),
    'age': np.random.randint(18, 80, 1000),
    'purchased': np.random.randint(0, 2, 1000)  # Binary target
}
df = pd.DataFrame(data)

# HARD BOUNDARY - Split features and target
X = df.drop(columns='purchased')
y = df['purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================================
# 1. ONE-HOT ENCODING
# Use for: Low cardinality (<50), no natural order, need interpretability
# Pros: No false relationships, works with all models | Cons: Dimension explosion
# ============================================================================
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_train[['color']])
X_train_ohe = ohe.transform(X_train[['color']])
X_test_ohe = ohe.transform(X_test[['color']])
# 'color' (1 column) → ['color_red', 'color_blue', 'color_green', 'color_yellow'] (4 columns)

# ============================================================================
# 2. ORDINAL ENCODING
# Use for: Ordered categories (small<medium<large) OR tree-based models
# Pros: only 1 column created
# ============================================================================
ord_enc = OrdinalEncoder(categories=[['small', 'medium', 'large']], handle_unknown='use_encoded_value', unknown_value=-1)
ord_enc.fit(X_train[['size']])
X_train_ord = ord_enc.transform(X_train[['size']])
X_test_ord = ord_enc.transform(X_test[['size']])
# 'size' values: ['small', 'medium', 'large'] → [0, 1, 2] (preserves order)

# ============================================================================
# 3. TARGET ENCODING
# Use for: when one-hot would cause dimension explosion (>50 categories)
# Cons: technically leakage, but accepted practice
# ============================================================================
target_enc = TargetEncoder(cols=['city'])
target_enc.fit(X_train, y_train)
X_train_target = target_enc.transform(X_train)
X_test_target = target_enc.transform(X_test)
# 'city' values: ['NYC', 'LA', 'Chicago'] → [0.47, 0.53, 0.45] (mean target per city)

# ============================================================================
# 4. LEAVE-ONE-OUT ENCODING
# Technically more correct than target encoding
# Cons: unstable for rare categories with only a few rows
# ============================================================================
loo_enc = LeaveOneOutEncoder(cols=['city'])
loo_enc.fit(X_train, y_train)
X_train_loo = loo_enc.transform(X_train)
X_test_loo = loo_enc.transform(X_test)
# 'city': each row gets mean of other rows from same city (NYC row1→0.48, NYC row2→0.46)

# ============================================================================
# 5. BINARY ENCODING
# Use for: High cardinality (100-10K categories), need dimension reduction
# Pros: far less columns than one-hot | Cons: Creates artificial relationships
# ============================================================================
binary_enc = BinaryEncoder(cols=['city'])
binary_enc.fit(X_train)
X_train_binary = binary_enc.transform(X_train)
X_test_binary = binary_enc.transform(X_test)
# 'city' (55 categories) → 6 binary columns (log2(55)≈6): NYC→[0,0,1,1,0,1]

# ============================================================================
# 6. HASHING ENCODING
# Use for: >10K categories or streaming data, ex. product SKUs
# Pros: even less columns than binary | Cons: collisions because multiple categories map to same hash
# ============================================================================
hash_enc = HashingEncoder(cols=['user_id'], n_components=8)
hash_enc.fit(X_train)
X_train_hash = hash_enc.transform(X_train)
X_test_hash = hash_enc.transform(X_test)
# 'user_id' (10K unique) → 8 hash columns (user_123→col3=1, user_456→col7=1, collisions occur)

# ============================================================================
# EXAMPLE: Full pipeline with mixed encoding
# ============================================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['color']),
    ('ord', OrdinalEncoder(categories=[['small', 'medium', 'large']], handle_unknown='use_encoded_value', unknown_value=-1), ['size']),
    ('target', TargetEncoder(), ['city']),
    ('hash', HashingEncoder(n_components=8), ['user_id'])
], remainder='passthrough')  # passthrough keeps income, age as-is

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train) # fits preprocessor, then fits model
score = pipeline.score(X_test, y_test)  # transforms X_test, then scores
