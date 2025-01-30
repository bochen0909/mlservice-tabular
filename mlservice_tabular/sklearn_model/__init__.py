"""External routes package for ML Service."""
from mlservice_tabular.sklearn_model.ridge import RidgeModel
from mlservice_tabular.sklearn_model.logistic import LogisticRegressionModel
from mlservice_tabular.sklearn_model.lasso import LassoModel
from mlservice_tabular.sklearn_model.random_forest_regression import RandomForestRegressionModel

__all__ = ['RidgeModel', 'LogisticRegressionModel', 'LassoModel', 'RandomForestRegressionModel']
