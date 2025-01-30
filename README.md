# MLService-Tabular

A FastAPI-based machine learning service for tabular data providing REST API endpoints for training and inference using scikit-learn models.

## Description

MLService-Tabular is an extension package for pymlservice that implements various scikit-learn models for tabular data. It provides a standardized API interface for training and making predictions with different regression and classification models.

Currently supported models:
- Ridge Regression
- Lasso Regression
- Logistic Regression

## Requirements

- Python >3.10
- pymlservice ^0.1.1
- Additional dependencies will be installed through Poetry

## Installation

1. Make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install the package:
```bash
poetry install
```

## Usage

### Running the Server

Start the server using Poetry:

```bash
poetry run python -m mlservice_tabular.main --host 0.0.0.0 --port 8000
```

Command line options:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--external-routines`: List of additional external routine modules to import

### API Documentation

Once the server is running, you can access the Swagger UI documentation at:
```
http://localhost:8000/docs
```

### Available Models

#### Ridge Regression
- Endpoint: `/sklearn/ridge`
- Parameters:
  - `alpha`: Regularization strength (default: 1.0)

#### Lasso Regression
- Endpoint: `/sklearn/lasso`
- Parameters:
  - `alpha`: Regularization strength (default: 1.0)

#### Logistic Regression
- Endpoint: `/sklearn/logistic`
- Parameters:
  - Default scikit-learn logistic regression parameters

### Example Usage

Training a model:
```bash
curl -X POST "http://localhost:8000/sklearn/ridge/train" \
     -H "Content-Type: application/json" \
     -d '{
           "data": [...],
           "target_column": "target",
           "hyperparameters": {"alpha": 0.5}
         }'
```

Making predictions:
```bash
curl -X POST "http://localhost:8000/sklearn/ridge/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "data": [...]
         }'
```

## Development

### Running Tests

Run the test suite using Poetry:

```bash
poetry run pytest
```

### Adding New Models

1. Create a new model class in `mlservice_tabular/sklearn_model/`
2. Inherit from appropriate base class (TabRegression or TabClassification)
3. Implement required methods: `_train()` and `_predict()`
4. Decorate class with `@model_endpoints("sklearn/your_model_name")`

## License

[License information not provided]

## Authors

- Bo Chen <bochen0909@gmail.com>
