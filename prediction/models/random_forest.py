from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


def train_random_forest(X_train, y_train):
    """
    Train the Random Forest model using optimal hyperparameters.
    """
    # Optional: Standardize features for consistency with other models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features=None,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Save the scaler to apply to test data
    joblib.dump(scaler, "models/random_forest_scaler.pkl")
    return model


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate the Random Forest model.
    """
    # Apply saved scaler to test data
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
