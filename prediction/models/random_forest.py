from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_random_forest(X_train, y_train, class_weight=None, **fit_kwargs):
    """
    Train the Random Forest model using optimal hyperparameters.

    Accepts an optional class_weight dict to handle imbalance.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features=None,
        max_depth=None,
        random_state=42,
        class_weight=class_weight
    )
    model.fit(X_train_scaled, y_train, **fit_kwargs)

    # Save the scaler to apply to test data
    joblib.dump(scaler, "models/random_forest_scaler.pkl")
    return model


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate the Random Forest model.
    """
    scaler = joblib.load("models/random_forest_scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
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
