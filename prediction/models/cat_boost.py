import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostError
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from tqdm import tqdm

def train_catboost(X_train, y_train, class_weights=None):
    """
    Train a CatBoostClassifier on the training data.
    """
    Xt = X_train.copy()
    cat_cols = Xt.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        Xt[c] = Xt[c].astype('category')

    params = {
        'iterations': 600,
        'depth': 8,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3,
        'bootstrap_type': 'Bernoulli',
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'class_weights': [0.7342, 7.0377, 0.6685],
        'random_seed': 42,
        'verbose': False,
    }
    if class_weights is not None:
        params['class_weights'] = [class_weights[i] for i in sorted(class_weights)]

    model = CatBoostClassifier(**params)
    model.fit(Xt, y_train, cat_features=cat_cols)
    return model

def evaluate_catboost(model, X_test, y_test):
    """
    Evaluate the CatBoost model and return accuracy and classification report.
    """
    Xt = X_test.copy()
    cat_cols = Xt.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        Xt[c] = Xt[c].astype('category')

    y_pred = model.predict(Xt)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def save_model(model, path):
    """
    Save the trained CatBoost model to disk.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load a trained CatBoost model from disk.
    """
    return joblib.load(path)

def tune_catboost(X_train, y_train,
                  class_weights=None,
                  scoring='f1_macro',
                  cv_folds=5,
                  n_iter=30,
                  random_state=42):
    """
    Randomized tuning of CatBoost hyperparams with tqdm.
    """
    # Prepare data and cat feature list
    Xt = X_train.copy()
    cat_cols = Xt.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        Xt[c] = Xt[c].astype('category')

    # Prepare class_weights list
    weights_list = None
    if class_weights is not None:
        weights_list = [class_weights[i] for i in sorted(class_weights)]

    # Parameter distributions (no Poisson bootstrap on CPU)
    param_dist = {
        'iterations':    [200, 400, 600],
        'depth':         [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg':   [1, 3, 5],
        'bootstrap_type':['Bayesian','Bernoulli'],
    }
    sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = None

    for params in tqdm(sampler, desc="Randomized CatBoost tuning", unit="trial"):
        # merge in fixed settings
        params = {
            **params,
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'random_seed': random_state,
            'verbose': False
        }
        if weights_list is not None:
            params['class_weights'] = weights_list

        try:
            scores = []
            for train_idx, val_idx in cv.split(Xt, y_train):
                X_tr, y_tr = Xt.iloc[train_idx], y_train.iloc[train_idx]
                X_val, y_val = Xt.iloc[val_idx], y_train.iloc[val_idx]
                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, cat_features=cat_cols)
                preds = model.predict(X_val)
                scores.append(f1_score(y_val, preds, average=scoring.split('_')[-1]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
        except CatBoostError:
            continue

    print(f"Best {scoring}: {best_score:.4f}")
    print("Best params:", best_params)

    # Refit on full data
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(Xt, y_train, cat_features=cat_cols)
    return best_model
