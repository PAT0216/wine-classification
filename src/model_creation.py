# src/model_creation.py
# author: Prabuddha Tamhane
# date: 2025-12-12

"""Module for creating classification models for wine classification."""

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def create_classification_models(seed: int = 123) -> dict:
    """
    Create a dictionary of classification models for wine classification.

    This function provides a standard set of classifiers used for comparing
    model performance in the wine classification analysis. All models that
    support random_state are initialized with the provided seed for
    reproducibility.

    Parameters
    ----------
    seed : int, default=123
        Random seed for reproducibility. Must be a non-negative integer.

    Returns
    -------
    dict
        Dictionary mapping model names (str) to sklearn estimator instances.
        Keys: "dummy", "Decision Tree", "KNN", "RBF SVM", "Logistic Regression"

    Raises
    ------
    TypeError
        If `seed` is not an integer.
    ValueError
        If `seed` is negative.

    Examples
    --------
    >>> models = create_classification_models(seed=42)
    >>> list(models.keys())
    ['dummy', 'Decision Tree', 'KNN', 'RBF SVM', 'Logistic Regression']
    """
    if not isinstance(seed, int):
        raise TypeError("`seed` must be an integer.")
    if seed < 0:
        raise ValueError("`seed` must be a non-negative integer.")

    models = {
        "dummy": DummyClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(),
        "RBF SVM": SVC(random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
    }

    return models


def fit_models(models: dict, X_train, y_train) -> dict:
    """
    Fit all models in a dictionary on the provided training data.

    This function iterates through a dictionary of sklearn estimators and
    fits each one on the provided training data. The models are fitted
    in-place and the same dictionary is returned with fitted models.

    Parameters
    ----------
    models : dict
        Dictionary mapping model names (str) to sklearn estimator instances.
    X_train : pd.DataFrame
        Training features. Must be a pandas DataFrame.
    y_train : array-like
        Training target vector. Length must match number of rows in X_train.

    Returns
    -------
    dict
        The same dictionary with all models now fitted on the training data.

    Raises
    ------
    TypeError
        If `models` is not a dictionary or `X_train` is not a DataFrame.
    ValueError
        If `models` is empty or `X_train` and `y_train` have different lengths.

    Examples
    --------
    >>> from sklearn.dummy import DummyClassifier
    >>> import pandas as pd
    >>> models = {"dummy": DummyClassifier()}
    >>> X = pd.DataFrame({"x1": [0, 1, 0, 1]})
    >>> y = [0, 1, 0, 1]
    >>> fitted = fit_models(models, X, y)
    >>> hasattr(fitted["dummy"], "classes_")
    True
    """
    import pandas as pd

    if not isinstance(models, dict):
        raise TypeError("`models` must be a dictionary of estimators.")
    if len(models) == 0:
        raise ValueError("`models` must be a non-empty dictionary.")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("`X_train` must be a pandas DataFrame.")
    if len(X_train) != len(y_train):
        raise ValueError("`X_train` and `y_train` must have the same length.")

    for model_name, model in models.items():
        model.fit(X_train, y_train)

    return models
