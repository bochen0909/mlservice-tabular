from typing import Any, Optional
import pandas as pd
from sklearn.linear_model import Lasso
from mlservice.core.tabml import TabRegression
from mlservice.core.ml import model_endpoints

@model_endpoints("sklearn/lasso")
class LassoModel(TabRegression):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = Lasso(alpha=self.hyperparameters.get("alpha", 1.0))

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
