"""Tests for RandomForest model."""
import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
import pytest
from fastapi.testclient import TestClient
from mlservice_tabular.sklearn_model import RandomForestModel
from mlservice.main import setup_routes, app

def read_prediction_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@pytest.fixture
def random_forest_model():
    return RandomForestModel(params={"hyperparameters": {"n_estimators": 100}, "columns": {"target": "target"}})

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
def client():
    setup_routes(['external_routes'])
    return TestClient(app)

# Test decorator functionality
def test_model_endpoints_decorator():
    assert hasattr(RandomForestModel, 'router'), "Decorator should add router attribute to model class"
    assert RandomForestModel.router.prefix == "/model/sklearn/random_forest", "Router should have correct prefix"

# Model functionality tests
def test_model_initialization():
    model = RandomForestModel(params={"hyperparameters": {"n_estimators": 50}})
    assert model.hyperparameters["n_estimators"] == 50
    
    # Test default params
    model = RandomForestModel()
    assert isinstance(model.model.n_estimators, int)

def test_train(random_forest_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    
    metadata = random_forest_model.train(str(train_path))
    
    assert hasattr(random_forest_model.model, 'estimators_'), "Model should be fitted"
    assert 'timestamp' in metadata
    assert 'metrics' in metadata
    assert metadata['train_path'] == str(train_path)

def test_predict(random_forest_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train the model first
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    random_forest_model.train(str(train_path))
    
    # Test prediction
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    prediction_file_path = random_forest_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert random_forest_model.prediction_column in prediction.columns
    assert len(prediction) == len(predict_data)

def test_load_model(random_forest_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train and save the model
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    metadata = random_forest_model.train(str(train_path))
    assert hasattr(random_forest_model.model, 'estimators_')
    
    # Load the model and verify
    model_path = metadata['model_path']
    model_path = os.path.join(model_path, "model.joblib")
    loaded_model = joblib.load(model_path)
    
    assert hasattr(loaded_model.model, 'estimators_')
    assert hasattr(loaded_model.__class__, 'router')
    assert isinstance(loaded_model, RandomForestModel)
    
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
        "hyperparameters": {"n_estimators": 100},
        "columns": {"target": "target"}
    })
    response = client.post(
        "/model/sklearn/random_forest/train",
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
        "hyperparameters": {"n_estimators": 100},
        "columns": {"target": "target"}
    })
    train_response = client.post(
        "/model/sklearn/random_forest/train",
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
        "/model/sklearn/random_forest/predict",
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
        "/model/sklearn/random_forest/predict",
        json={
            "data_path": "nonexistent.csv",
            "model_path": "nonexistent_model"
        }
    )
    assert response.status_code == 500

def test_feature_column_inference(random_forest_model, sample_data):
    # Test that feature columns are correctly inferred
    features = random_forest_model._infer_features_columns(sample_data.columns)
    assert 'feature1' in features
    assert 'feature2' in features
    assert 'target' not in features

    # Test inference with no columns
    features = random_forest_model._infer_features_columns(None)
    assert features == []

def test_target_column_handling(random_forest_model, sample_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_data.to_csv(train_path, index=False)
    
    metadata = random_forest_model.train(str(train_path))
    assert hasattr(random_forest_model.model, 'estimators_')
    
    # Verify prediction contains correct column
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    prediction_file_path = random_forest_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert random_forest_model.prediction_column in prediction.columns
