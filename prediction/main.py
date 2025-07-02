import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Import model functions
from models.xgboost_model import train_xgboost, evaluate_xgboost, save_model, tune_xgboost
from models.logistic_regression import train_logistic, evaluate_logistic
from models.random_forest import train_random_forest, evaluate_random_forest
from models.cat_boost import train_catboost, evaluate_catboost, save_model as save_catboost, tune_catboost


def _style_ax(ax):
    """Apply minimalist styling: remove spines, add light horizontal grid."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def plot_all_metrics(y_test, X_test, model_outputs, ordered_labels=None):
    """
    Generate plots for accuracy, confusion matrix, class metrics, ROC curves,
    weighted comparison, and feature importance.
    """
    accuracies = {}
    weighted_scores = {}

    # ROC setup
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    for name, (model, y_pred, y_proba, X_input) in model_outputs.items():
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        # Confusion Matrix
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=ordered_labels).plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"results/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

        # Per-Class Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose().iloc[:3][["precision", "recall", "f1-score"]]

        ax = df.plot(
            kind="bar",
            ylim=(0, 1),
            title=f"Per-Class Metrics - {name}",
            figsize=(8, 6),
        )
        ax.set_ylabel("Score")
        # Use class names instead of numeric labels
        ax.set_xticklabels(ordered_labels, rotation=0)
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
    ax = sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    # Remove grid lines
    ax.grid(False)
    # Annotate bar values on top
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison")
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
        feat_imp.nlargest(10).plot(
            kind="barh",
            title=f"Top 10 Feature Importances - {name}",
            figsize=(8, 6)
        )
        plt.tight_layout()
        plt.savefig(f"results/{name.replace(' ', '_').lower()}_feature_importance.png")
        plt.close()


def main():
    # Load the cleaned data
    data_path = "data/cleaned_chess_games_prediction.csv"
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    # Prepare features and labels
    X = data.drop(columns=['winner', 'game_id', 'victory_status'])
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)
    y = data['winner'].astype('category').cat.codes

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # confirm your mapping
    cats = data['winner'].astype('category').cat.categories
    print("Category → code:", dict(enumerate(cats)))
    code2label = dict(enumerate(cats))  # {0:'Black',1:'Draw',2:'White'}
    ordered_labels = [code2label[i] for i in sorted(code2label)]

    # confirm test counts
    print("y_test counts:", y_test.value_counts().sort_index().to_dict())
    print(f"Class distribution in training set: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Class distribution in test set: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Compute class weights on the original distribution
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print("Using class weights:", class_weights)
    sample_weight = y_train.map(class_weights)

    model_outputs = {}

    # CatBoost
    print("\nTraining CatBoost with class weights...")
    cat_model = train_catboost(X_train, y_train, class_weights=class_weights)
    acc, report = evaluate_catboost(cat_model, X_test, y_test)
    print(f"CatBoost Accuracy: {acc:.4f}")

    with open("results/cat_boost_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    save_catboost(cat_model, "models/catboost_model.pkl")
    # register for plotting and ensembling
    proba_cat = cat_model.predict_proba(X_test)
    model_outputs["CatBoost"] = (
        cat_model,
        cat_model.predict(X_test),
        proba_cat,
        X_test
    )


    # XGBoost
    print("\nTraining XGBoost model")
    # xgb_model = tune_xgboost(X_train, y_train, scoring='accuracy', cv_folds=5, n_jobs=-1)
    xgb_model = train_xgboost(X_train, y_train, sample_weight=sample_weight)
    acc, report = evaluate_xgboost(xgb_model, X_test, y_test)
    print(f"XGBoost Accuracy: {acc:.4f}")

    with open("results/xgboost_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    save_model(xgb_model, "models/xgboost_model.pkl")
    model_outputs["XGBoost"] = (
        xgb_model,
        xgb_model.predict(X_test),
        xgb_model.predict_proba(X_test),
        X_test
    )

    # Logistic Regression
    print("\nTraining Logistic Regression model with class weights...")
    log_model = train_logistic(X_train, y_train, class_weight=class_weights)
    acc, report = evaluate_logistic(log_model, X_test, y_test)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    with open("results/logistic_regression_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    save_model(log_model, "models/logistic_regression_model.pkl")
    scaler_log = joblib.load("models/logistic_regression_scaler.pkl")
    X_test_scaled_log = scaler_log.transform(X_test)
    model_outputs["Logistic Regression"] = (
        log_model,
        log_model.predict(X_test_scaled_log),
        log_model.predict_proba(X_test_scaled_log),
        X_test
    )

    # Random Forest
    print("\nTraining Random Forest model with class weights...")
    rf_model = train_random_forest(X_train, y_train, class_weight=class_weights)
    acc, report = evaluate_random_forest(rf_model, X_test, y_test)
    print(f"Random Forest Accuracy: {acc:.4f}")
    with open("results/random_forest_results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    save_model(rf_model, "models/random_forest_model.pkl")
    scaler_rf = joblib.load("models/random_forest_scaler.pkl")
    X_test_scaled_rf = scaler_rf.transform(X_test)
    model_outputs["Random Forest"] = (
        rf_model,
        rf_model.predict(X_test_scaled_rf),
        rf_model.predict_proba(X_test_scaled_rf),
        X_test
    )

    # Build ensemble by averaging probabilities…
    print("\nBuilding ensemble by averaging probabilities…")
    proba_cat = cat_model.predict_proba(X_test)
    proba_xgb = xgb_model.predict_proba(X_test)
    proba_lr = log_model.predict_proba(X_test_scaled_log)
    proba_rf = rf_model.predict_proba(X_test_scaled_rf)

    # weights found via Dirichlet sampling
    w_xgb, w_rf, w_lr, w_cat = 0.661, 0.143, 0.134, 0.062

    # weighted average of probabilities
    proba_ensemble = (
            w_xgb * proba_xgb +
            w_rf * proba_rf +
            w_lr * proba_lr +
            w_cat * proba_cat
    )
    proba_ensemble /= (w_xgb + w_rf + w_lr + w_cat)
    base_preds = np.argmax(proba_ensemble, axis=1)

    # apply fixed threshold for Draw (code 1)
    draw_idx = ordered_labels.index("Draw")
    threshold = 0.45
    y_pred_ensemble = base_preds.copy()
    y_pred_ensemble[proba_ensemble[:, draw_idx] >= threshold] = draw_idx

    # evaluate and save report
    acc_ens = accuracy_score(y_test, y_pred_ensemble)
    report_ens = classification_report(
        y_test, y_pred_ensemble,
        target_names=ordered_labels
    )
    print(f"Ensemble Accuracy: {acc_ens:.4f}")
    with open("results/ensemble_report.txt", "w") as f:
        f.write(f"Ensemble Accuracy: {acc_ens:.4f}\n\n")
        f.write(report_ens)

    # register for plotting
    model_outputs["Ensemble"] = (
        None,  # no underlying model object needed
        y_pred_ensemble,
        proba_ensemble,
        X_test
    )

    # Visual Comparison
    print("\nGenerating comparison plots...")
    plot_all_metrics(y_test, X_test, model_outputs, ordered_labels)
    print("All models trained, evaluated, and visualized successfully!")


if __name__ == "__main__":
    main()