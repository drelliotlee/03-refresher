
# Train ENSEMBLE model on selected features
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Simple Voting Ensemble
model = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42))
], voting='soft')  # 'soft' uses predict_proba, 'hard' uses majority vote

# Or Stacking Ensemble (meta-learner combines base models)
# model = StackingClassifier(estimators=[
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#     ('xgb', XGBClassifier(random_state=42)),
#     ('lgbm', LGBMClassifier(random_state=42))
# ], final_estimator=LogisticRegression())

model.fit(X_train_processed, y_train)

# Evaluate on test set using predict_proba
y_test_proba = model.predict_proba(X_test_processed)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)
print(f"Test ROC-AUC: {test_score:.4f}")

# Option 2: Stacking with Preprocessing Pipeline

from sklearn.ensemble import StackingClassifier

# Create full pipeline: preprocessing + ensemble
full_pipeline = Pipeline([
    ('preprocessing', preprocess_pipeline),  # Your feature selection pipeline
    ('stacking', StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(random_state=42)),
            ('lgbm', LGBMClassifier(random_state=42))
        ],
        final_estimator=LogisticRegression(),
        cv=5  # Use CV to generate meta-features
    ))
])

full_pipeline.fit(X_train, y_train)
y_test_proba = full_pipeline.predict_proba(X_test)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)
print(f"Test ROC-AUC: {test_score:.4f}")
