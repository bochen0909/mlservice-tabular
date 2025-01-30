"""
Tests for tabml models.
"""
import pytest
import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from mlservice_tabular.sklearn import RidgeModel, LogisticRegressionModel
from mlservice.core.tabml import TabModel, TabClassification
from mlservice.main import setup_routes, app

def read_prediction_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@pytest.fixture
def ridge_model():
    return RidgeModel(params={"hyperparameters": {"alpha": 1.0}, "columns": {"target": "target"}})

@pytest.fixture
def classification_model():
    class TestClassification(TabClassification):
        def __init__(self, params=None):
            self._fitted = False
            self.return_proba = True
            super().__init__(params)
            self.model = LogisticRegression()
            
        def _train(self, data):
            """Implementation of training logic."""
            if isinstance(data, str):
                data = pd.read_csv(data)
            features = self._infer_features_columns(data.columns)
            X = data[features]
            y = data[self.target_column]
            self.model.fit(X, y)
            self.fitted_ = True
            
            return {
                "train_path": str(data) if isinstance(data, str) else None,
                "timestamp": str(datetime.now()),
                "model_path": None
            }
            
        def _predict(self, data):
            """Implementation of prediction logic."""
            if isinstance(data, str):
                data = pd.read_csv(data)
            features = self._infer_features_columns(data.columns)
            X = data[features]
            predictions = pd.DataFrame()
            predictions[self.prediction_column] = self.model.predict(X)
            if self.return_proba:
                predictions[self.predict_proba_column] = self.model.predict_proba(X)[:, 1]
            return predictions

        @property
        def fitted_(self):
            """Get the fitted state."""
            return self._fitted

        @fitted_.setter
        def fitted_(self, value):
            """Set the fitted state."""
            self._fitted = value
    
    return TestClassification(params={
        "columns": {
            "target": "target",
            "prediction": "pred",
            "predict_proba": "prob"
        }
    })

@pytest.fixture
def sample_data():
    # Create synthetic regression data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['target'] = y
    return df

@pytest.fixture
def sample_classification_data():
    # Create synthetic classification data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    df['target'] = y
    return df

@pytest.fixture
def client():
    setup_routes(['external_routes'])
    return TestClient(app)

# Test decorator functionality
def test_model_endpoints_decorator():
    assert hasattr(RidgeModel, 'router'), "Decorator should add router attribute to model class"
    assert RidgeModel.router.prefix == "/model/sklearn/ridge", "Router should have correct prefix"

# Model functionality tests
def test_model_initialization():
    model = RidgeModel(params={"hyperparameters": {"alpha": 2.0}})
    assert model.hyperparameters["alpha"] == 2.0
    
    # Test default alpha
    model = RidgeModel()
    assert model.model.alpha == 1.0

def test_train(ridge_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    
    metadata = ridge_model.train(str(train_path))
    
    assert hasattr(ridge_model.model, 'coef_'), "Model should be fitted"
    assert 'timestamp' in metadata
    assert 'metrics' in metadata
    assert metadata['train_path'] == str(train_path)

def test_predict(ridge_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train the model first
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    ridge_model.train(str(train_path))
    
    # Test prediction
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    prediction_file_path = ridge_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert ridge_model.prediction_column in prediction.columns
    assert len(prediction) == len(predict_data)

def test_load_model(ridge_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train and save the model
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    metadata = ridge_model.train(str(train_path))
    assert hasattr(ridge_model.model, 'coef_')
    
    # Load the model and verify
    model_path = metadata['model_path']
    model_path = os.path.join(model_path, "model.joblib")
    loaded_model = joblib.load(model_path)
    
    assert hasattr(loaded_model.model, 'coef_')
    assert hasattr(loaded_model.__class__, 'router')
    assert isinstance(loaded_model, RidgeModel)
    
    # Test prediction with loaded model
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    prediction_file_path = loaded_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert loaded_model.prediction_column in prediction.columns

# API endpoint tests
def test_train_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    
    params = json.dumps({
        "hyperparameters": {"alpha": 1.0},
        "columns": {"target": "target"}
    })
    response = client.post(
        "/model/sklearn/ridge/train",
        json={
            "train_path": str(train_path),
            "params": params
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'timestamp' in data
    assert 'metrics' in data
    assert data['train_path'] == str(train_path)

def test_predict_endpoint(client, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train the model first
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    
    params = json.dumps({
        "hyperparameters": {"alpha": 1.0},
        "columns": {"target": "target"}
    })
    train_response = client.post(
        "/model/sklearn/ridge/train",
        json={
            "train_path": str(train_path),
            "params": params
        }
    )
    assert train_response.status_code == 200
    model_path = train_response.json()['model_path']
    
    # Test prediction
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    response = client.post(
        "/model/sklearn/ridge/predict",
        json={
            "data_path": str(predict_path),
            "model_path": model_path
        }
    )
    
    assert response.status_code == 200, response.text
    prediction_file_path = response.json()
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert len(prediction) > 0

def test_error_handling(client, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Test prediction with nonexistent files
    response = client.post(
        "/model/sklearn/ridge/predict",
        json={
            "data_path": "nonexistent.csv",
            "model_path": "nonexistent_model"
        }
    )
    assert response.status_code == 500

def test_feature_column_inference(ridge_model, sample_data):
    # Test that feature columns are correctly inferred
    features = ridge_model._infer_features_columns(sample_data.columns)
    assert 'feature1' in features
    assert 'feature2' in features
    assert 'target' not in features

    # Test inference with no columns
    features = ridge_model._infer_features_columns(None)
    assert features == []

def test_target_column_handling(ridge_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    
    metadata = ridge_model.train(str(train_path))
    assert hasattr(ridge_model.model, 'coef_')
    
    # Verify prediction contains correct column
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    prediction_file_path = ridge_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert ridge_model.prediction_column in prediction.columns

def test_classification_model_properties(classification_model):
    # Test predict_proba_column property
    assert classification_model.predict_proba_column == "prob"
    assert classification_model.prediction_column == "pred"
    assert classification_model.target_column == "target"

def test_classification_model_training(classification_model, sample_classification_data):
    # Test model training
    metadata = classification_model._train(sample_classification_data)
    assert "timestamp" in metadata
    assert classification_model.fitted_

def test_classification_model_prediction(classification_model, sample_classification_data):
    # First train the model
    classification_model._train(sample_classification_data)
    
    # Test prediction with both prediction and probability
    classification_model.return_proba = True
    predictions = classification_model._predict(sample_classification_data)
    assert classification_model.prediction_column in predictions.columns
    assert classification_model.predict_proba_column in predictions.columns

def test_classification_model_evaluation(classification_model, sample_classification_data):
    # First train the model
    classification_model._train(sample_classification_data)
    
    # Test evaluation with both columns
    classification_model.return_proba = True
    predictions = classification_model._predict(sample_classification_data)
    test_data = pd.concat([sample_classification_data, predictions], axis=1)
    metrics = classification_model._evaluate(test_data)
    assert metrics['accuracy'] is not None
    assert metrics['f1'] is not None
    assert metrics['precision'] is not None
    assert metrics['recall'] is not None
    assert metrics['auc_score'] is not None

    # Test evaluation with only prediction column
    classification_model.return_proba = False
    predictions = classification_model._predict(sample_classification_data)
    test_data = pd.concat([sample_classification_data, predictions], axis=1)
    metrics = classification_model._evaluate(test_data)
    assert metrics['accuracy'] is not None
    assert metrics['f1'] is None
    assert metrics['precision'] is None
    assert metrics['recall'] is None
    assert metrics['auc_score'] is None

@pytest.fixture
def logistic_model():
    return LogisticRegressionModel(params={"columns": {"target": "target"}})

def test_logistic_model_initialization():
    model = LogisticRegressionModel(params={"hyperparameters": {"C": 2.0}})
    assert model.hyperparameters["C"] == 2.0
    
    # Test default parameters
    model = LogisticRegressionModel()
    assert isinstance(model.model, LogisticRegression)

def test_logistic_model_train(logistic_model, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    
    metadata = logistic_model.train(str(train_path))
    
    assert hasattr(logistic_model.model, 'coef_'), "Model should be fitted"
    assert 'timestamp' in metadata
    assert 'metrics' in metadata
    assert metadata['train_path'] == str(train_path)
    
    # Test metrics
    assert 'train' in metadata['metrics']
    train_metrics = metadata['metrics']['train']
    assert 'accuracy' in train_metrics
    assert 'f1' in train_metrics
    assert 'precision' in train_metrics
    assert 'recall' in train_metrics
    assert 'auc_score' in train_metrics

def test_logistic_model_predict(logistic_model, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train the model first
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    logistic_model.train(str(train_path))
    
    # Test prediction
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_classification_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    prediction_file_path = logistic_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert logistic_model.prediction_column in prediction.columns
    assert logistic_model.predict_proba_column in prediction.columns
    assert len(prediction) == len(predict_data)
    assert ((prediction[logistic_model.prediction_column] == 0) | 
            (prediction[logistic_model.prediction_column] == 1)).all()
    assert ((prediction[logistic_model.predict_proba_column] >= 0) & 
            (prediction[logistic_model.predict_proba_column] <= 1)).all()

def test_logistic_model_load(logistic_model, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train and save the model
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    metadata = logistic_model.train(str(train_path))
    assert hasattr(logistic_model.model, 'coef_')
    
    # Load the model and verify
    model_path = metadata['model_path']
    model_path = os.path.join(model_path, "model.joblib")
    loaded_model = joblib.load(model_path)
    
    assert hasattr(loaded_model.model, 'coef_')
    assert hasattr(loaded_model.__class__, 'router')
    assert isinstance(loaded_model, LogisticRegressionModel)
    
    # Test prediction with loaded model
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_classification_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    prediction_file_path = loaded_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert loaded_model.prediction_column in prediction.columns
    assert loaded_model.predict_proba_column in prediction.columns

def test_logistic_model_endpoints(client, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    
    # Test training endpoint
    params = json.dumps({
        "hyperparameters": {},
        "columns": {"target": "target"}
    })
    response = client.post(
        "/model/sklearn/logistic/train",
        json={
            "train_path": str(train_path),
            "params": params
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'timestamp' in data
    assert 'metrics' in data
    assert data['train_path'] == str(train_path)
    assert 'train' in data['metrics']
    train_metrics = data['metrics']['train']
    assert 'accuracy' in train_metrics
    
    model_path = data['model_path']
    
    # Test prediction endpoint
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_classification_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    response = client.post(
        "/model/sklearn/logistic/predict",
        json={
            "data_path": str(predict_path),
            "model_path": model_path
        }
    )
    
    assert response.status_code == 200, response.text
    prediction_file_path = response.json()
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert len(prediction) > 0
    assert 'prediction' in prediction.columns
    assert 'predict_proba' in prediction.columns
