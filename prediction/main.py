import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

# Import model functions
from models.xgboost_model import train_xgboost, evaluate_xgboost, save_model
from models.logistic_regression import train_logistic, evaluate_logistic
from models.random_forest import train_random_forest, evaluate_random_forest

def plot_all_metrics(y_test, X_test, model_outputs):
    """
    Generate plots for accuracy, confusion matrix, class metrics, ROC curves,
    weighted comparison, and feature importance.
    """
    accuracies = {}
    weighted_scores = {}

    # ROC setup
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    for name, (model, y_pred, y_proba, X_input) in model_outputs.items():
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        # Confusion Matrix
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["White", "Black", "Draw"]).plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"results/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

        # Per-Class Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().iloc[:3][["precision", "recall", "f1-score"]]
        df.plot(kind="bar", ylim=(0, 1), title=f"Per-Class Metrics - {name}", figsize=(8, 6))
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"results/class_metrics_{name.replace(' ', '_')}.png")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        # Weighted avg for grouped bar plot
        weighted = report["weighted avg"]
        weighted_scores[name] = [weighted["precision"], weighted["recall"], weighted["f1-score"]]

    # Plot Accuracy Comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
    plt.grid(axis='y')
    plt.savefig("results/accuracy_comparison.png")
    plt.close()

    # Plot ROC Comparison
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-Averaged ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("results/roc_comparison.png")
    plt.close()

    # Grouped Weighted Score Comparison
    df_scores = pd.DataFrame(weighted_scores, index=["Precision", "Recall", "F1-score"])
    df_scores.plot(kind="bar", ylim=(0, 1), figsize=(8, 6))
    plt.title("Weighted Avg Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/weighted_metrics_comparison.png")
    plt.close()

    # Feature Importances
    for name in ["XGBoost", "Random Forest"]:
        model, _, _, X_input = model_outputs[name]
        feat_imp = pd.Series(model.feature_importances_, index=X_input.columns)
        feat_imp.nlargest(10).plot(kind="barh", title=f"Top 10 Feature Importances - {name}", figsize=(8, 6))
        plt.tight_layout()
        plt.savefig(f"results/{name.replace(' ', '_').lower()}_feature_importance.png")
        plt.close()

def main():
    # Load the cleaned data
    data_path = "data/cleaned_chess_games_prediction.csv"
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    # Prepare features and labels
    X = data.drop(columns=['winner', 'game_id'])
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
    y = data['winner'].astype('category').cat.codes

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_outputs = {}

    # XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train)
    acc, report = evaluate_xgboost(xgb_model, X_test, y_test)
    print(f"XGBoost Accuracy: {acc:.4f}")
    save_model(xgb_model, "models/xgboost_model.pkl")
    model_outputs["XGBoost"] = (xgb_model, xgb_model.predict(X_test), xgb_model.predict_proba(X_test), X_test)

    # Logistic Regression
    print("\nTraining Logistic Regression model...")
    log_model = train_logistic(X_train, y_train)
    acc, report = evaluate_logistic(log_model, X_test, y_test)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    save_model(log_model, "models/logistic_regression_model.pkl")
    scaler_log = joblib.load("models/logistic_regression_scaler.pkl")
    X_test_scaled_log = scaler_log.transform(X_test)
    model_outputs["Logistic Regression"] = (log_model, log_model.predict(X_test_scaled_log), log_model.predict_proba(X_test_scaled_log), X_test)

    # Random Forest
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    acc, report = evaluate_random_forest(rf_model, X_test, y_test)
    print(f"Random Forest Accuracy: {acc:.4f}")
    save_model(rf_model, "models/random_forest_model.pkl")
    scaler_rf = joblib.load("models/random_forest_scaler.pkl")
    X_test_scaled_rf = scaler_rf.transform(X_test)
    model_outputs["Random Forest"] = (rf_model, rf_model.predict(X_test_scaled_rf), rf_model.predict_proba(X_test_scaled_rf), X_test)

    # Visual Comparison
    print("\nGenerating comparison plots...")
    plot_all_metrics(y_test, X_test, model_outputs)
    print("All models trained, evaluated, and visualized successfully!")

if __name__ == "__main__":
    main()
