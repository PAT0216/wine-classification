
import pandas as pd
from sklearn.dummy import DummyClassifier

# create simple test data
X_SMALL = pd.DataFrame({"x1": [0, 1, 0, 1]})
y_SMALL = pd.Series([0, 1, 0, 1])

MODELS_SMALL = {
    "dummy": DummyClassifier(strategy="most_frequent", random_state=0)
}