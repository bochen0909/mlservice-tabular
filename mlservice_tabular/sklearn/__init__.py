"""External routes package for ML Service."""
from mlservice_tabular.sklearn.ridge import RidgeModel
from mlservice_tabular.sklearn.logistic import LogisticRegressionModel
from mlservice_tabular.sklearn.lasso import LassoModel

__all__ = ['RidgeModel', 'LogisticRegressionModel', 'LassoModel']
