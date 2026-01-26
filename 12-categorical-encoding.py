import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder, LeaveOneOutEncoder, BinaryEncoder, HashingEncoder
from sklearn.metrics import roc_auc_score

X = df.drop(columns='purchased') 
y = df['purchased'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================================
# 1. ONE-HOT ENCODING
# Use for: Low cardinality (<50) + no ordinal order
# Cons: Dimension explosion
# ============================================================================
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_train[['color']])
X_train_ohe = ohe.transform(X_train[['color']])
X_test_ohe = ohe.transform(X_test[['color']])
#   │ COLOR  │  →   │ RED │ BLUE │ GREEN │ YELLOW │
#   │ red    │      │  1  │  0   │   0   │   0    │
#   │ blue   │      │  0  │  1   │   0   │   0    │
#   │ green  │      │  0  │  0   │   1   │   0    │
#   │ yellow │      │  0  │  0   │   0   │   1    │

# ============================================================================
# 2. ORDINAL ENCODING
# Use for: Ordered categories (small<medium<large) OR tree-based models
# Pros: only 1 column created
# ============================================================================
ord_enc = OrdinalEncoder(categories=[['small', 'medium', 'large']], handle_unknown='use_encoded_value', unknown_value=-1)
ord_enc.fit(X_train[['size']])
X_train_ord = ord_enc.transform(X_train[['size']])
X_test_ord = ord_enc.transform(X_test[['size']])
#   │ SIZE   │  →    │ SIZE │
#   │ small  │       │  0   │
#   │ medium │       │  1   │
#   │ large  │       │  2   │

# ============================================================================
# 3. TARGET ENCODING
# Use for: when one-hot would cause dimension explosion (>50 categories)
# Cons: technically leakage, but accepted practice
# ============================================================================
target_enc = TargetEncoder(cols=['city'])
target_enc.fit(X_train, y_train)
X_train_target = target_enc.transform(X_train)
X_test_target = target_enc.transform(X_test)
#   │ CITY    │  →    │ CITY │
#   │ NYC     │       │ 0.47 │
#   │ LA      │       │ 0.53 │
#   │ Chicago │       │ 0.45 │

# ============================================================================
# 4. LEAVE-ONE-OUT ENCODING
# Technically more correct than target encoding
# Cons: unstable for rare categories with only a few rows
# ============================================================================
loo_enc = LeaveOneOutEncoder(cols=['city'])
loo_enc.fit(X_train, y_train)
X_train_loo = loo_enc.transform(X_train)
X_test_loo = loo_enc.transform(X_test)

# ============================================================================
# 5. BINARY ENCODING
# Use for: High cardinality (100-10K categories), need dimension reduction
# Pros: far less columns than one-hot
# Cons: Creates artificial relationships
# ============================================================================
binary_enc = BinaryEncoder(cols=['city'])
binary_enc.fit(X_train)
X_train_binary = binary_enc.transform(X_train)
X_test_binary = binary_enc.transform(X_test)
#   │ CITY    │  →    │ BIN_0 │ BIN_1 │ BIN_2 │ BIN_3 │ BIN_4 │ BIN_5 │
#   │ NYC     │       │   0   │   0   │   1   │   1   │   0   │   1   │
#   │ LA      │       │   0   │   1   │   0   │   0   │   1   │   0   │
#   │ Chicago │       │   0   │   1   │   1   │   0   │   0   │   0   │
# 'city' (55 categories) → 6 binary columns needed (2^6=64 > 55 categories)

# ============================================================================
# 6. HASHING ENCODING
# Use for: >10K categories or streaming data, ex. product SKUs
# Pros: even less columns than binary
# Cons: collisions because multiple categories map to same hash
# ============================================================================
hash_enc = HashingEncoder(cols=['user_id'], n_components=8)
hash_enc.fit(X_train)
X_train_hash = hash_enc.transform(X_train)
X_test_hash = hash_enc.transform(X_test)
#   │ USER_ID │  →    │ COL_0 │ COL_1 │ COL_2 │ COL_3 │ COL_4 │ COL_5 │ COL_6 │ COL_7 │
#   │ user_123│       │   0   │   0   │   0   │   1   │   0   │   0   │   0   │   0   │
#   │ user_456│       │   0   │   0   │   0   │   0   │   0   │   0   │   1   │   0   │
#   │ user_789│       │   0   │   0   │   0   │   1   │   0   │   0   │   0   │   0   │  
# only 8 hash columns ==> collisions bc 2^8=256 << 10K

# ============================================================================
# EXAMPLE: Full pipeline with mixed encoding
# ============================================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    # other pre-processing steps
    
    ('encoding', ColumnTransformer([
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['color']),
        ('ord', OrdinalEncoder(categories=[['small', 'medium', 'large']], handle_unknown='use_encoded_value', unknown_value=-1), ['size']),
        ('target', TargetEncoder(), ['city']),
        ('hash', HashingEncoder(n_components=8), ['user_id'])
    ], remainder='passthrough')),  # passthrough keeps income, age as-is
    
    # other pre-processing steps
])

preprocess_pipeline.fit(X_train, y_train)
X_train_processed = preprocess_pipeline.transform(X_train)
X_test_processed = preprocess_pipeline.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_processed, y_train)
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)