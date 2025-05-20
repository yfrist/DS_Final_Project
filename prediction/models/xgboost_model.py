from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """
    Train the XGBoost model with the best known hyperparameters.
    """
    # Convert categorical columns to category dtype
    X_train = X_train.apply(lambda col: col.astype('category') if col.dtype == 'object' else col)

    # Best hyperparameters found from tuning
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
        min_child_weight=3,
        max_depth=5,
        learning_rate=0.3,
        gamma=0,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)
    return model


def evaluate_xgboost(model, X_test, y_test):
    """
    Evaluate the XGBoost model.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


def save_model(model, path):
    """
    Save the trained model.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    """
    Load a trained model.
    """
    return joblib.load(path)
