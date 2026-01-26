# train-validate-test
# AKA train-tune-evaluate
# train (parameters) - tune (model/hyperparameters) - evaluate (final test)

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

## KEY BOUNDARY BETWEEN 2 PIPES ##
X = df.drop('target_column', axis=1)  
y = df['target_column']                
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
## KEY BOUNDARY BETWEEN 2 PIPES ##

# OPTION 1: in practice, pre-processing is done here (scaling, encoding, imputation, etc.)
preprocess_pipeline.fit(X_train, y_train)
X_train = preprocess_pipeline.transform(X_train)
X_test = preprocess_pipeline.transform(X_test)
# end pre-processing

all_scores_df = pd.DataFrame(columns=['model_family', 'hyperparams', 'mean_score', 'std_score'])
for model_family in MODEL_FAMILIES:
    for hyperparams in HYPERPARAMETER_GRID[model_family]:

        cv_scores = []
        
        # VERSION 1
        # for continuous Ys, split X_train/y_train into 4 train folds + 1 val fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for i, j in kf.split(X_train): 
            X_train_fold, y_train_fold = X_train.iloc[i], y_train.iloc[i]
            X_val_fold, y_val_fold = X_train.iloc[j], y_train.iloc[j]
            
            # OPTION 2: most rigorous/correct way: pre-process INSIDE each fold to avoid data leakage
            preprocess_pipeline.fit(X_train_fold, y_train_fold)
            X_train_fold = preprocess_pipeline.transform(X_train_fold)
            X_val_fold = preprocess_pipeline.transform(X_val_fold)
            # end pre-process
            
            model = model_family(hyperparams)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate using predict_proba and explicit metric
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_val_proba)
            cv_scores.append(score)
            
        # VERSION 2    
        # for categorical Ys, use StratifiedKFold
        # Problem: what if your target distribution is imbalanced? (90% normal, 10% fraud)
        # Stratified K-Fold: ensures every fold will have ~90% normal, ~10% fraud
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for i, j in skf.split(X_train, y_train): 
            X_train_fold, y_train_fold = X_train.iloc[i], y_train.iloc[i]
            X_val_fold, y_val_fold = X_train.iloc[j], y_train.iloc[j]
            
            # OPTION 2: most rigorous/correct way: pre-process INSIDE each fold to avoid data leakage
            preprocess_pipeline.fit(X_train_fold, y_train_fold)
            X_train_fold = preprocess_pipeline.transform(X_train_fold)
            X_val_fold = preprocess_pipeline.transform(X_val_fold)
            # end pre-process
            
            model = model_family(hyperparams)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate using predict_proba and explicit metric
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_val_proba)
            cv_scores.append(score)
            
        # VERSION 3
        # for data with GROUPS that should not be split across train/val
        # Problem: what if you have multiple rows per user/patient/store?
        # Group K-Fold: ensures ALL rows from same group stay together in same fold
        #   - ex. Medical: all scans from same patient in one fold 
        #   - ex. Finance: all transactions from same user in one fold 
        gkf = GroupKFold(n_splits=5)
        groups = X_train['group_col']  # Assuming group_col is in X_train
        for i, j in gkf.split(X_train, y_train, groups=groups): 
            X_train_fold, y_train_fold = X_train.iloc[i], y_train.iloc[i]
            X_val_fold, y_val_fold = X_train.iloc[j], y_train.iloc[j]
            
            # OPTION 2: most rigorous/correct way: pre-process INSIDE each fold to avoid data leakage
            preprocess_pipeline.fit(X_train_fold, y_train_fold)
            X_train_fold = preprocess_pipeline.transform(X_train_fold)
            X_val_fold = preprocess_pipeline.transform(X_val_fold)
            # end pre-process
            
            model = model_family(hyperparams)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate using predict_proba and explicit metric
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_val_proba)
            cv_scores.append(score)
            
        # VERSION 4
        # for TIME SERIES data always trains on past, validates on future
        # Example: if you have 1000 days of data, k=5:
        #   Fold 1: train on days 1-200,   validate on days 201-400
        #   Fold 2: train on days 1-400,   validate on days 401-600
        #   Fold 3: train on days 1-600,   validate on days 601-800
        #   Fold 4: train on days 1-800,   validate on days 801-1000
        # Notice: training set GROWS each fold (expanding window)
        tscv = TimeSeriesSplit(n_splits=5)
        for i, j in tscv.split(X_train):
            X_train_fold, y_train_fold = X_train.iloc[i], y_train.iloc[i]
            X_val_fold, y_val_fold = X_train.iloc[j], y_train.iloc[j]
            
            # OPTION 2: most rigorous/correct way: pre-process INSIDE each fold to avoid data leakage
            preprocess_pipeline.fit(X_train_fold, y_train_fold)
            X_train_fold = preprocess_pipeline.transform(X_train_fold)
            X_val_fold = preprocess_pipeline.transform(X_val_fold)
            # end pre-process
            
            model = model_family(hyperparams)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate using predict_proba and explicit metric
            y_val_proba = model.predict_proba(X_val_fold)[:, 1]
            score = roc_auc_score(y_val_fold, y_val_proba)
            cv_scores.append(score)
            
        # add 1 more row to all_scores_df
        all_scores_df = pd.concat([all_scores_df, pd.DataFrame([{
            'model_family': model_family.__name__,
            'hyperparams': hyperparams,
            'mean_score': average(cv_scores),
            'std_score': std(cv_scores)
        }])], ignore_index=True)

best_row = all_scores_df.sort_values('mean_score', ascending=False).iloc[0]
best_model = (MODEL_FAMILIES[best_row['model_family']], best_row['hyperparams'])
best_model.fit(X_train, y_train) # retrain on full train data with best hyperparameters

y_test_proba = best_model.predict_proba(X_test)[:, 1]
test_score = roc_auc_score(y_test, y_test_proba)

