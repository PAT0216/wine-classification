
import pandas as pd
from sklearn.dummy import DummyClassifier
from src.model_evaluation import compute_cv_results
import pytest

# create simple test data
X_SMALL = pd.DataFrame({"x1": [0, 1, 0, 1]})
y_SMALL = pd.Series([0, 1, 0, 1])

MODELS_SMALL = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=0)
}

# function: compute_cv_results() - normal and edge case test


def test_compute_cv_results_success():
    """Normal and edge case: returns a pd.Dataframe with expected columns."""
    df = compute_cv_results(MODELS_SMALL, X_SMALL, y_SMALL, cv=2)

    assert isinstance(df, pd.DataFrame)
    assert "dummy" in df.index
    assert "test_score" in df.columns
    assert "(+/-" in df.loc["dummy", "test_score"]

# compute_cv_results() - error case test


def test_compute_cv_results_error():
    """Error case: input checks."""
    with pytest.raises(TypeError, match="`models` must be a dictionary"):
        compute_cv_results(["not", "a", "dict"], X_SMALL, y_SMALL, cv=2)

    with pytest.raises(ValueError, match="non-empty"):
        compute_cv_results({}, X_SMALL, y_SMALL, cv=2)

    with pytest.raises(TypeError, match="`X_train` must be a dataframe"):
        compute_cv_results(MODELS_SMALL, [0, 1, 0, 1], y_SMALL, cv=2)

    with pytest.raises(ValueError, match="same lengths"):
        compute_cv_results(MODELS_SMALL, X_SMALL, y_SMALL[:2], cv=2)

    with pytest.raises(ValueError, match="cv.*>= 2"):
        compute_cv_results(MODELS_SMALL, X_SMALL, y_SMALL, cv=1)
