from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_logistic(X_train, y_train):
    """
    Train the Logistic Regression model using optimal hyperparameters.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=0.1,                # Best regularization strength
        penalty='l2',         # L2 regularization
        solver='lbfgs',       # Best solver
        max_iter=5000
    )
    model.fit(X_train_scaled, y_train)

    # Save the scaler for use during evaluation
    joblib.dump(scaler, "models/logistic_regression_scaler.pkl")
    return model


def evaluate_logistic(model, X_test, y_test):
    """
    Evaluate the Logistic Regression model.
    """
    # Load the scaler and scale the test data
    scaler = joblib.load("models/logistic_regression_scaler.pkl")
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
