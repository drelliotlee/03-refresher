# train-validate-test
# AKA train-tune-evaluate
# train (parameters) - tune (model/hyperparameters) - evaluate (final test)

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

## KEY BOUNDARY BETWEEN 2 PIPES ##
train_val_data, test_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
## KEY BOUNDARY BETWEEN 2 PIPES ##

results_df = pd.DataFrame(columns=['model_family', 'hyperparams', 'mean_score', 'std_score', 'cv_scores'])
for model_family in MODEL_FAMILIES:
    for hyperparams in HYPERPARAMETER_GRID[model_family]:

        cv_scores = []
        
        # VERSION 1
        # for continuous Ys, split train_val_data into 4 train folds + 1 val fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(train_val_data): 
            train_fold = train_val_data.iloc[train_idx]
            val_fold = train_val_data.iloc[val_idx]
            model = model_family(hyperparams)
            model.fit(train_fold)
            score = evaluate(model, val_fold)
            cv_scores.append(score)
            
        # VERSION 2    
        # for categorical Ys, use StratifiedKFold
        # Problem: what if your target distribution is imbalanced? (90% normal, 10% fraud)
        # Stratified K-Fold: ensures every fold will have ~90% normal, ~10% fraud
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X = train_val_data.drop('target_column', axis=1) 
        y = train_val_data['target_column']               
        for train_idx, val_idx in skf.split(X, y): 
            train_fold = train_val_data.iloc[train_idx]
            val_fold = train_val_data.iloc[val_idx]
            model = model_family(hyperparams)
            model.fit(train_fold)
            score = evaluate(model, val_fold)
            cv_scores.append(score)
            
        # VERSION 3
        # for data with GROUPS that should not be split across train/val
        # Problem: what if you have multiple rows per user/patient/store?
        # Group K-Fold: ensures ALL rows from same group stay together in same fold
        #   - ex. Medical: all scans from same patient in one fold 
        #   - ex. Finance: all transactions from same user in one fold 
        gkf = GroupKFold(n_splits=5)
        X = train_val_data.drop('target_column', axis=1)
        y = train_val_data['target_column']
        for train_idx, val_idx in gkf.split(X, y, groups=train_val_data['group_col']): # key line here. 
            train_fold = train_val_data.iloc[train_idx]
            val_fold = train_val_data.iloc[val_idx]
            model = model_family(hyperparams)
            model.fit(train_fold)
            score = evaluate(model, val_fold)
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
        for train_idx, val_idx in tscv.split(train_val_data):
            train_fold = train_val_data.iloc[train_idx]
            val_fold = train_val_data.iloc[val_idx]
            model = model_family(hyperparams)
            model.fit(train_fold)
            score = evaluate(model, val_fold)
            cv_scores.append(score)
            
        # now for all 4 versions, aggregate results
        mean_cv_score = average(cv_scores)
        std_cv_score = std(cv_scores)  
        results_df = pd.concat([results_df, pd.DataFrame([{
            'model_family': model_family.__name__,
            'hyperparams': hyperparams,
            'mean_score': mean_cv_score,
            'std_score': std_cv_score,
            'cv_scores': cv_scores,
        }])], ignore_index=True)

best_row = results_df.sort_values('mean_score', ascending=False).iloc[0]
best_model = (MODEL_FAMILIES[best_row['model_family']], best_row['hyperparams'])
best_model.fit(train_val_data) # retrain on full train+val data with best hyperparameters

test_score = evaluate(best_model, test_data) # final evaluation on UNTOUCHED test set

