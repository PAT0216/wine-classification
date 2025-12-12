import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def compute_cv_results(models: dict, X_train: pd.DataFrame, y_train, cv: int=5) -> pd.DataFrame:
    """
    Compute cross-validation results for multiple models,
    show the results as a dataframe.
    
    This function modulizes the "Cross-validation results" part from the 
    original `scripts/model_analysis.py`. It runs `cross_validate` from 
    `sklearn.model_selection` for each model, and returns the mean and
    standard deviation of each metric in this format: "mean (+/- std)".

    Parameters
    ----------
    models: dict
        Dictionary of multiple models for mapping each model name (str)
        to sklearn estimators.
    X_train: pd.DataFrame
        DataFrame including all features in training set.
    y_train: array-like
        Target vector in training set. The length must be same as `X_train`.
    cv: int, default=5
        Number of cross-validation folds. Number of folds must >=2.
    :return: Description

    Returns
    ----------
    pd.DataFrame
        Return the cv results in a dataframe, where each row corresponds to
        a model, each column is a metric returned by `cross_validate` (e.g.,
        fit_time, score_time, test_score, train_score).
        Values are in this format: "mean (+/- std)".

    Raises
    ----------
    TypeError
        If `model` is not dict, or `X_train` is not a pd.DataFrame.
    ValueError
        If `model` is empty, or `cv`<2, or `X_train` and `y_train` have
        different lengths.
    """