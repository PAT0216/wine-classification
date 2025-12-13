# scripts/model_analysis.py
# author: Harrison Li
# date: 2025-12-05

import sys
from pathlib import Path
# Ensure project root is on sys.path so `import src...` works when running:
#   python scripts/model_analysis.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import click
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from src.model_creation import create_classification_models, fit_models
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import cross_validate, cross_val_predict


@click.command()
@click.option("--train-features", type=str, help="Path to scaled training features CSV")
@click.option("--train-target", type=str, help="Path to training target CSV")
@click.option("--test-features", type=str, help="Path to scaled testing features CSV")
@click.option("--test-target", type=str, help="Path to testing target CSV")
@click.option("--results-dir", type=str, help="Directory to save results")
@click.option("--seed", type=int, help="Random seed", default=123)
def main(train_features, train_target, test_features, test_target, results_dir, seed):
    """
    Performs model analysis:

    1. Trains multiple models using cross-validation and saves a table
       of mean ± std scores for each metric (fit_time, score_time,
       test_score, train_score) for all models.
    2. Evaluates all models on the test set, saving accuracy, recall,
       precision and F1 in a table.
    3. Plots a confusion matrix for the best model (RBF SVM) using
       5-fold cross-validation on the training data.
    4. Saves tables under <results_dir>/tables and figures under
       <results_dir>/figures.
    """

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    tables_dir = os.path.join(results_dir, "tables")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X_train = pd.read_csv(train_features)
    y_train = pd.read_csv(train_target).squeeze()  # Ensure it's a Series
    X_test = pd.read_csv(test_features)
    y_test = pd.read_csv(test_target).squeeze()

    # ------------------------------------------------------------------
    # Define models (using modular function from src.model_creation)
    # ------------------------------------------------------------------
    models = create_classification_models(seed=seed)

    # ------------------------------------------------------------------
    # 1. Cross-validation results (full table, like in IPYNB)
    # ------------------------------------------------------------------
    from src.model_evaluation import compute_cv_results
    print("Performing cross-validation for all models ...")
    
    cv_results_df = compute_cv_results(models, X_train, y_train, cv=5)

    cv_results_path = os.path.join(tables_dir, "cross_val_results.csv")
    cv_results_df.to_csv(cv_results_path)
    print(f"Cross-validation results saved to {cv_results_path}")

    # ------------------------------------------------------------------
    # 2. Test set metrics for all models (accuracy, recall, precision, f1)
    # ------------------------------------------------------------------
    print("Evaluating all models on the test set ...")
    test_metrics_dict = {}

    # Fit all models on training data (using modular function)
    models = fit_models(models, X_train, y_train)

    for model_name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        test_metrics_dict[model_name] = {
            "accuracy": acc,
            "recall": rec,
            "precision": prec,
            "f1": f1,
        }

    test_metrics_df = pd.DataFrame(test_metrics_dict).T
    test_metrics_path = os.path.join(tables_dir, "test_metrics.csv")
    test_metrics_df.to_csv(test_metrics_path)
    print(f"Test metrics saved to {test_metrics_path}")

    # ------------------------------------------------------------------
    # 3. Confusion matrix for the best model (RBF SVM) using 5-fold CV
    #    on the training data, like in the notebook
    # ------------------------------------------------------------------
    best_model_name = "RBF SVM"
    best_model = models[best_model_name]

    print(f"Creating confusion matrix for best model: {best_model_name} ...")

    # 5-fold cross-validated predictions on the training data
    y_train_pred_cv = cross_val_predict(best_model, X_train, y_train, cv=5)

    cm = confusion_matrix(y_train, y_train_pred_cv)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot(cmap="Blues")

    plt.title(
        f"Confusion Matrix (5-fold CV): {best_model_name}\n"
        "Labels: 0 = red, 1 = white"
    )

    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
