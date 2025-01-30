# Developer Guide: Adding New Models to MLService-Tabular

This guide explains how to add new scikit-learn models to the MLService-Tabular project.

## Project Architecture

MLService-Tabular uses a hierarchy of base classes:
- `TabRegression`: Base class for regression models
- `TabClassification`: Base class for classification models

These base classes handle:
- Feature column inference
- Data validation
- API endpoint generation
- Prediction column management
- Common model operations

## Adding a New Model

### 1. Choose the Right Base Class

- For regression problems: Inherit from `TabRegression`
- For classification problems: Inherit from `TabClassification`

### 2. Create the Model File

Create a new Python file in `mlservice_tabular/{type}_model/` named after your model:

(Type is used to organize the modules, for example if the model is derived from `sklearn`, make it as `mlservice_tabular/sklearn_model/`)

```python
from typing import Any, Optional
import pandas as pd
from {type}.some_module import YourModel
from mlservice.core.tabml import TabRegression  # or TabClassification
from mlservice.core.ml import model_endpoints

@model_endpoints("{type}/your_model_name")
class YourModelClass(TabRegression):  # or TabClassification
    def __init__(self, params=None):
        super().__init__(params)
        self.model = YourModel(**self.hyperparameters)

    def _predict(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_columns = self._infer_features_columns(data.columns)
        X = data[feature_columns].values
        y_pred = self.model.predict(X)
        data[self.prediction_column] = y_pred
        return data
        
    def _train(self, train_data: Any, eval_data: Optional[Any] = None):
        feature_columns = self._infer_features_columns(train_data.columns)
        self._set_feature_columns(feature_columns)
        X = train_data[feature_columns].values
        y = train_data[self.target_column].values
        self.model.fit(X, y)
        return self
```

### 3. Required Methods

#### `__init__(self, params=None)`
- Initialize your scikit-learn model with hyperparameters
- Access hyperparameters through `self.hyperparameters`
- Always call `super().__init__(params)`

#### `_train(self, train_data: Any, eval_data: Optional[Any] = None)`
- Extract feature columns using `_infer_features_columns()`
- Store feature columns using `_set_feature_columns()`
- Convert data to numpy arrays for scikit-learn
- Train the model using `fit()`
- Return `self`

#### `_predict(self, data: pd.DataFrame) -> pd.DataFrame`
- Get stored feature columns
- Convert input data to numpy arrays
- Make predictions
- Add predictions to the input DataFrame
- Return the modified DataFrame

### 4. Model Registration

1. Add `@model_endpoints` decorator with your model's endpoint path:
```python
@model_endpoints("{type}/your_model_name")
```

2. Add to `mlservice_tabular/{type}_model/__init__.py`:
```python
from mlservice_tabular.{type}_model.your_model import YourModelClass

__all__ = [..., 'YourModelClass']
```

### 5. Testing

Create a test file in `tests/mlservice_tabular/{type}_model/test_your_model.py`:

```python
import numpy as np
import pandas as pd
import pytest
from mlservice_tabular.{type}_model.your_model import YourModelClass

def test_model_training():
    # Create sample data
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    
    # Initialize model
    model = YourModelClass()
    
    # Train model
    model.train(X, target_column='target')
    
    # Verify features were stored
    assert model.feature_columns == ['feature1', 'feature2']
    
    # Verify model was trained
    assert model.model is not None

def test_model_prediction():
    # Create and train model
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    
    model = YourModelClass()
    model.train(X_train, target_column='target')
    
    # Create test data
    X_test = pd.DataFrame({
        'feature1': [10, 11],
        'feature2': [12, 13]
    })
    
    # Make predictions
    results = model.predict(X_test)
    
    # Verify predictions
    assert model.prediction_column in results.columns
    assert len(results) == len(X_test)
```

## Special Considerations

### For Classification Models

- Override `_predict()` to include probability scores:
```python
def _predict(self, data: pd.DataFrame) -> pd.DataFrame:
    feature_columns = self._infer_features_columns(data.columns)
    X = data[feature_columns].values
    y_pred = self.model.predict(X)
    y_proba = self.model.predict_proba(X)[:, 1]  # For binary classification
    data[self.prediction_column] = y_pred
    data[self.predict_proba_column] = y_proba
    return data
```

### Best Practices

1. **Hyperparameters**
   - Access through `self.hyperparameters`
   - Provide sensible defaults
   - Document required and optional parameters

2. **Feature Handling**
   - Use `_infer_features_columns()` to get feature columns
   - Store feature columns after training
   - Handle missing values appropriately

3. **Error Handling**
   - Validate input data shapes and types
   - Provide meaningful error messages
   - Check for required parameters

4. **Testing**
   - Test with different hyperparameters
   - Test edge cases (empty data, missing values)
   - Test model persistence if implemented

## Example Usage

After adding your model, it can be used through the API:

```bash
# Training
curl -X POST "http://localhost:8000/{type}/your_model_name/train" \
     -H "Content-Type: application/json" \
     -d '{
           "data": [...],
           "target_column": "target",
           "hyperparameters": {"param1": "value1"}
         }'

# Prediction
curl -X POST "http://localhost:8000/{type}/your_model_name/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "data": [...]
         }'
```

## Troubleshooting

1. **Model Not Found Error**
   - Check if model is imported in `__init__.py`
   - Verify endpoint path in `@model_endpoints` decorator

2. **Training Errors**
   - Verify data format matches requirements
   - Check hyperparameter types and values
   - Ensure target column exists in training data

3. **Prediction Errors**
   - Ensure model is trained before prediction
   - Check feature columns match training data
   - Verify input data format
