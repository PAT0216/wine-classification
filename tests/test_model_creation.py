# tests/test_model_creation.py

import pytest
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.model_creation import create_classification_models, fit_models


# ============================================================================
# Test Data for fit_models
# ============================================================================

X_SMALL = pd.DataFrame({"x1": [0, 1, 0, 1, 0, 1], "x2": [1, 0, 1, 0, 1, 0]})
y_SMALL = pd.Series([0, 1, 0, 1, 0, 1])

MODELS_SMALL = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=0)
}


# ============================================================================
# create_classification_models() - Normal / Edge Case Tests
# ============================================================================


def test_create_classification_models_returns_dict():
    """Normal case: returns a dictionary with expected model names."""
    models = create_classification_models(seed=123)

    assert isinstance(models, dict)
    assert len(models) == 5

    expected_keys = ["dummy", "Decision Tree", "KNN", "RBF SVM", "Logistic Regression"]
    assert list(models.keys()) == expected_keys


def test_create_classification_models_correct_types():
    """Normal case: each model is the correct sklearn estimator type."""
    models = create_classification_models(seed=42)

    assert isinstance(models["dummy"], DummyClassifier)
    assert isinstance(models["Decision Tree"], DecisionTreeClassifier)
    assert isinstance(models["KNN"], KNeighborsClassifier)
    assert isinstance(models["RBF SVM"], SVC)
    assert isinstance(models["Logistic Regression"], LogisticRegression)


def test_create_classification_models_seed_applied():
    """Normal case: random_state is correctly set for applicable models."""
    seed = 99
    models = create_classification_models(seed=seed)

    assert models["dummy"].random_state == seed
    assert models["Decision Tree"].random_state == seed
    assert models["RBF SVM"].random_state == seed
    assert models["Logistic Regression"].random_state == seed


def test_create_classification_models_logistic_max_iter():
    """Edge case: Logistic Regression has max_iter=2000."""
    models = create_classification_models(seed=123)

    assert models["Logistic Regression"].max_iter == 2000


def test_create_classification_models_default_seed():
    """Edge case: default seed is 123."""
    models = create_classification_models()

    assert models["dummy"].random_state == 123


# ============================================================================
# create_classification_models() - Error Case Tests
# ============================================================================


def test_create_classification_models_seed_not_int():
    """Error case: seed must be an integer."""
    with pytest.raises(TypeError, match="`seed` must be an integer"):
        create_classification_models(seed="not_an_int")

    with pytest.raises(TypeError, match="`seed` must be an integer"):
        create_classification_models(seed=3.14)


def test_create_classification_models_seed_negative():
    """Error case: seed must be non-negative."""
    with pytest.raises(ValueError, match="non-negative"):
        create_classification_models(seed=-1)


# ============================================================================
# fit_models() - Normal / Edge Case Tests
# ============================================================================


def test_fit_models_returns_fitted_dict():
    """Normal case: returns a dictionary with fitted models."""
    models = {"dummy": DummyClassifier(strategy="most_frequent", random_state=0)}
    fitted = fit_models(models, X_SMALL, y_SMALL)

    assert isinstance(fitted, dict)
    assert "dummy" in fitted
    # Check that the model is fitted (has classes_ attribute)
    assert hasattr(fitted["dummy"], "classes_")


def test_fit_models_can_predict():
    """Normal case: fitted models can make predictions."""
    models = {"dummy": DummyClassifier(strategy="most_frequent", random_state=0)}
    fitted = fit_models(models, X_SMALL, y_SMALL)

    predictions = fitted["dummy"].predict(X_SMALL)
    assert len(predictions) == len(y_SMALL)


def test_fit_models_multiple_models():
    """Normal case: fits multiple models correctly."""
    models = {
        "dummy": DummyClassifier(strategy="most_frequent", random_state=0),
        "tree": DecisionTreeClassifier(random_state=0)
    }
    fitted = fit_models(models, X_SMALL, y_SMALL)

    assert hasattr(fitted["dummy"], "classes_")
    assert hasattr(fitted["tree"], "classes_")


def test_fit_models_returns_same_dict():
    """Edge case: returns the same dictionary object (models fitted in-place)."""
    models = {"dummy": DummyClassifier(strategy="most_frequent", random_state=0)}
    fitted = fit_models(models, X_SMALL, y_SMALL)

    assert fitted is models


# ============================================================================
# fit_models() - Error Case Tests
# ============================================================================


def test_fit_models_models_not_dict():
    """Error case: models must be a dictionary."""
    with pytest.raises(TypeError, match="`models` must be a dictionary"):
        fit_models(["not", "a", "dict"], X_SMALL, y_SMALL)


def test_fit_models_models_empty():
    """Error case: models must be non-empty."""
    with pytest.raises(ValueError, match="non-empty"):
        fit_models({}, X_SMALL, y_SMALL)


def test_fit_models_X_not_dataframe():
    """Error case: X_train must be a DataFrame."""
    models = {"dummy": DummyClassifier()}
    with pytest.raises(TypeError, match="`X_train` must be a pandas DataFrame"):
        fit_models(models, [[0, 1], [1, 0]], y_SMALL)


def test_fit_models_length_mismatch():
    """Error case: X_train and y_train must have same length."""
    models = {"dummy": DummyClassifier()}
    with pytest.raises(ValueError, match="same length"):
        fit_models(models, X_SMALL, y_SMALL[:2])
