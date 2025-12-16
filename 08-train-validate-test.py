# should really be called train-tune-evaluate
# train (parameters) - tune (model/hyperparameters) - evaluate (final test)

data = load_full_dataset()
train_val_data, test_data = split(data)
best_score = -infinity
best_model_config = None

for model_family in MODEL_FAMILIES:
    for hyperparams in HYPERPARAMETER_GRID[model_family]:

        cv_scores = []

        for train_fold, val_fold in K_FOLDS(train_val_data, k=5):
            # split train_val_data into k=5 equal folds
            # each iteration: 4 folds = train, 1 fold = validation
            model = model_family(hyperparams)
            model.fit(train_fold)
            score = evaluate(model, val_fold)
            cv_scores.append(score)

        mean_cv_score = average(cv_scores) # average performance across folds for these fixed hyperparameters
        if mean_cv_score > best_score:     # if best hyperparameters so far, keep them
            best_score = mean_cv_score
            best_model = (model_family, hyperparams)

best_model.fit(train_val_data) # retrain on full train+val data with best hyperparameters

test_score = evaluate(best_model, test_data) # final evaluation on untouched test set
# evaluate functions examples:
# regressions -> mean squared error or r2
# binary classification -> AUC-ROC
# multi-class classification -> F1-score or accuracy