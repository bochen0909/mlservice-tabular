"""Tests for XGBoost Classification model."""
import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
import pytest
from fastapi.testclient import TestClient
import xgboost as xgb
from mlservice_tabular.gbm import XGBoostClassificationModel
from mlservice.main import setup_routes, app

def read_prediction_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@pytest.fixture
def xgboost_model():
    return XGBoostClassificationModel(params={"hyperparameters": {"n_estimators": 100}, "columns": {"target": "target"}})

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

def test_model_initialization():
    model = XGBoostClassificationModel(params={"hyperparameters": {"n_estimators": 50}})
    assert model.hyperparameters["n_estimators"] == 50
    
    # Test default parameters
    model = XGBoostClassificationModel()
    assert isinstance(model.model, xgb.XGBClassifier)
    assert model.hyperparameters['objective'] == 'binary:logistic'

def test_train(xgboost_model, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    
    metadata = xgboost_model.train(str(train_path))
    
    assert hasattr(xgboost_model.model, 'get_booster'), "Model should be fitted"
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

def test_predict(xgboost_model, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train the model first
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    xgboost_model.train(str(train_path))
    
    # Test prediction
    predict_path = tmp_path / "predict.csv"
    predict_data = sample_classification_data.drop('target', axis=1)
    predict_data.to_csv(predict_path, index=False)
    
    prediction_file_path = xgboost_model.predict(str(predict_path))
    assert isinstance(prediction_file_path, str)
    assert os.path.exists(prediction_file_path)
    
    prediction = read_prediction_file(prediction_file_path)
    assert isinstance(prediction, pd.DataFrame)
    assert xgboost_model.prediction_column in prediction.columns
    assert xgboost_model.predict_proba_column in prediction.columns
    assert len(prediction) == len(predict_data)
    assert ((prediction[xgboost_model.prediction_column] == 0) | 
            (prediction[xgboost_model.prediction_column] == 1)).all()
    assert ((prediction[xgboost_model.predict_proba_column] >= 0) & 
            (prediction[xgboost_model.predict_proba_column] <= 1)).all()

def test_load_model(xgboost_model, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    # Train and save the model
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    metadata = xgboost_model.train(str(train_path))
    assert hasattr(xgboost_model.model, 'get_booster')
    
    # Load the model and verify
    model_path = metadata['model_path']
    model_path = os.path.join(model_path, "model.joblib")
    loaded_model = joblib.load(model_path)
    
    assert hasattr(loaded_model.model, 'get_booster')
    assert hasattr(loaded_model.__class__, 'router')
    assert isinstance(loaded_model, XGBoostClassificationModel)
    
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

def test_model_endpoints(client, sample_classification_data, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    train_path = tmp_path / "train.csv"
    sample_classification_data.to_csv(train_path, index=False)
    
    # Test training endpoint
    params = {
        "hyperparameters": {"n_estimators": 100},
        "columns": {"target": "target"}
    }
    response = client.post(
        "/model/gbm/xgboost_classification/train",
        json={
            "train_path": str(train_path),
            "params": json.dumps(params)
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
        "/model/gbm/xgboost_classification/predict",
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

def test_error_handling(client, tmp_path):
    os.environ['ML_HOME'] = str(tmp_path)
    
    response = client.post(
        "/model/gbm/xgboost_classification/predict",
        json={
            "data_path": "nonexistent.csv",
            "model_path": "nonexistent_model"
        }
    )
    assert response.status_code == 500
