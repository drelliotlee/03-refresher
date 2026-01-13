# ============================================================
# What is this generic evaluate() function?
# ============================================================

from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, root_mean_squared_error, auc

def evaluate(model, test_data, problem_type):
    if problem_type == 'binary_classification':
        # ROC-AUC (Area Under ROC Curve) - INDUSTRY STANDARD
        # Models predict probabilities [0,1], not hard labels {0,1}
        # ROC curve plots (FPR, TPR) across all thresholds
        # AUC Range: 0.5 (random) to 1.0 (perfect)
        y_true = test_data['target']
        y_pred_proba = model.predict_proba(test_data.drop('target', axis=1))[:, 1] 
        return roc_auc_score(y_true, y_pred_proba)  
    
    elif problem_type == 'binary_classification_imbalanced':
        # PRECISION-RECALL AUC - better than ROC-AUC for imbalanced data
        # PR curve plots (recall, precision) across all thresholds
        # Use when positive class is rare (e.g., 1% fraud, 0.1% cancer)
        y_true = test_data['target']
        y_pred_proba = model.predict_proba(test_data.drop('target', axis=1))[:, 1]
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall_vals, precision_vals)
    
    elif problem_type == 'regression':
        # RMSE (Root Mean Squared Error) = sqrt(average((actual - predicted)²)
        # Unlike MAE (mean avg error), penalizes large errors more heavily
        # Unlike R², has interpretable units (same as target variable)
        y_true = test_data['target']
        y_pred = model.predict(test_data.drop('target', axis=1))
        return root_mean_squared_error(y_true, y_pred)  
    
    elif problem_type == 'multiclass_classification':
        # F1 SCORE = 2 * (precision * recall) / (precision + recall)
        y_true = test_data['target']
        y_pred = model.predict(test_data.drop('target', axis=1))
        return f1_score(y_true, y_pred, average='weighted')  

# ============================================================
# Understanding Precision vs Recall
# ============================================================
    
# PRECISION = TP/(TP+FP) = "Of my positive predictions, how many were correct?"
# WHEN TO USE: when FALSE POSITIVES are costly
# Example: Auditing taxpayers for fraud takes time and manpower
#
# RECALL = TP/(TP+FN) = "Of all actual positives, how many did I catch?"
# WHEN TO USE: when FALSE NEGATIVES are catastrophic  
# Example: Missing a bomb at airport security is deadly