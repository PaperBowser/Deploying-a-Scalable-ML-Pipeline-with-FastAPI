import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from ml.model import train_model, compute_model_metrics
from ml.data import process_data


@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "workclass": ["Private", "Self-emp"],
        "education": ["Bachelors", "Masters"],
        "marital-status": ["Never-married", "Married"],
        "occupation": ["Tech-support", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "India"],
        "salary": [">50K", "<=50K"]
    })
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return df, cat_features


def test_train_model(sample_data):
    """
    checks that train_model() returns a random forest model
    """
    df, cat_features = sample_data
    X, y, _, _ = process_data(df, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics():
    """
    checks that compute_model_metrics() returns precision, recall, and fbeta scores from 0 to 1
    """
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_process_data_training(sample_data):
    """
    checks that process_data() (with training=True) returns:
    X as np.array of the same size as the data
    y as np.array of the same size as the data
    encoder as OneHotEncoder
    lb as LabelBinarizer
    """
    df, cat_features = sample_data
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
