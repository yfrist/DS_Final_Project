import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_xgboost(X_train, y_train, sample_weight=None, **fit_kwargs):
    """
    Train the XGBoost model with the best known hyperparameters.

    Accepts optional sample weights to handle class imbalance.
    """
    # Convert categorical columns to category dtype if needed
    X_train = X_train.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        enable_categorical=True,
        subsample=1.0,
        reg_lambda=1.0,
        reg_alpha=0,
        n_estimators=300,
        min_child_weight=1,
        max_depth=3,
        learning_rate=0.5,
        gamma=0,
        colsample_bytree=1.0
    )

    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)
    else:
        model.fit(X_train, y_train, **fit_kwargs)

    return model


def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate the XGBoost model and return accuracy and text report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


def tune_xgboost(X_train, y_train, sample_weight=None, scoring='accuracy', cv_folds=5, n_jobs=-1):
    """
    Perform grid search over key XGBoost hyperparameters to maximize the given scoring metric.
    Returns the best estimator found.

    Parameters:
    - X_train, y_train: training data
    - sample_weight: optional sample weights for imbalance handling
    - scoring: metric to optimize ('accuracy', 'f1_macro', etc.)
    - cv_folds: number of stratified folds for cross-validation
    - n_jobs: parallel jobs
    """
    # Base estimator with minimal defaults
    base = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        enable_categorical=True,
        verbosity=0
    )

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.3, 0.5],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'min_child_weight': [1, 3, 5],
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )

    if sample_weight is not None:
        grid.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        grid.fit(X_train, y_train)

    print(f"Best {scoring}: {grid.best_score_:.4f}")
    print("Best params:", grid.best_params_)
    return grid.best_estimator_


def save_model(model, path):
    """
    Save the trained XGBoost model to disk.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Load a trained XGBoost model from disk.
    """
    return joblib.load(path)
