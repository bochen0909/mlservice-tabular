"""External routes package for ML Service."""
from mlservice_tabular.sklearn.ridge import RidgeModel
from mlservice_tabular.sklearn.logistic import LogisticRegressionModel

__all__ = ['RidgeModel', 'LogisticRegressionModel']
