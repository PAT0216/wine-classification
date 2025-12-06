# scripts/model_analysis.py
# author: Harrison Li
# date: 2025-12-05

import click
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.model_selection import cross_validate, cross_val_predict

@click.command()
@click.option('--train-features', type=str, help="Path to scaled training features CSV")
@click.option('--train-target', type=str, help="Path to training target CSV")
@click.option('--test-features', type=str, help="Path to scaled testing features CSV")
@click.option('--test-target', type=str, help="Path to testing target CSV")
@click.option('--results-dir', type=str, help="Directory to save results")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(train_features, train_target, test_features, test_target, results_dir, seed):
    """
    Performs model analysis:
    1. Trains multiple models using cross-validation.
    2. Evaluates the best model (RBF SVM) on the test set.
    3. Saves results (CV scores, confusion matrix, test metrics).
    """
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load data
    X_train = pd.read_csv(train_features)
    y_train = pd.read_csv(train_target).squeeze() # Ensure it's a Series
    X_test = pd.read_csv(test_features)
    y_test = pd.read_csv(test_target).squeeze()

    # Define models
    models = {
        "dummy": DummyClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(),
        "RBF SVM": SVC(random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
    }

    # Cross-validation
    results_dict = {}
    print("Performing Cross-Validation...")
    for model_name, model in models.items():
        # Note: Data is already scaled, so we don't need a pipeline with StandardScaler here
        scores = cross_validate(model, X_train, y_train, cv=5, return_train_score=True)
        
        mean_test_score = scores['test_score'].mean()
        std_test_score = scores['test_score'].std()
        
        results_dict[model_name] = f"{mean_test_score:.4f} (+/- {std_test_score:.4f})"

    # Save CV results
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['CV Score (Mean +/- Std)'])
    results_path = os.path.join(results_dir, "cross_val_results.csv")
    results_df.to_csv(results_path)
    print(f"Cross-validation results saved to {results_path}")

    # Best Model Evaluation (RBF SVM)
    best_model_name = "RBF SVM"
    best_model = models[best_model_name]
    
    print(f"Evaluating Best Model: {best_model_name}...")
    best_model.fit(X_train, y_train)
    prediction = best_model.predict(X_test)

    # Calculate Test Metrics
    accuracy = accuracy_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    f1 = f1_score(y_test, prediction) # sklearn's f1_score default is binary

    test_metrics = {
        "model": [best_model_name],
        "accuracy": [accuracy],
        "recall": [recall],
        "precision": [precision],
        "f1": [f1]
    }
    
    # Save Test Metrics
    test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_path = os.path.join(results_dir, "test_metrics.csv")
    test_metrics_df.to_csv(test_metrics_path, index=False)
    print(f"Test metrics saved to {test_metrics_path}")

    # Confusion Matrix
    cm_display = ConfusionMatrixDisplay.from_predictions(y_test, prediction, cmap='Blues')
    plt.title(f"Confusion Matrix: {best_model_name}\nLabels: 0 = red, 1 = white")
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == '__main__':
    main()
