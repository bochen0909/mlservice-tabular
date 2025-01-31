from typing import Any, Optional
import json
import pandas as pd
import lightgbm as lgb
from mlservice.core.tabml import TabClassification
from mlservice.core.ml import model_endpoints

@model_endpoints("gbm/lightgbm_classification")
class LightGBMClassificationModel(TabClassification):
    def __init__(self, params=None):
        if params is None:
            params = {"hyperparameters": {}}
        elif isinstance(params, str):
            params = json.loads(params)
        if "hyperparameters" not in params:
            params["hyperparameters"] = {}
        params["hyperparameters"].setdefault('objective', 'binary')
        super().__init__(params)
        self.model = lgb.LGBMClassifier(**self.hyperparameters)

    def _predict(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_columns = self._infer_features_columns(data.columns)
        X = data[feature_columns].values
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]  # Probability of class 1
        data[self.prediction_column] = y_pred
        data[self.predict_proba_column] = y_proba
        return data
    
    def _train(self, train_data: Any, eval_data: Optional[Any] = None):
        feature_columns = self._infer_features_columns(train_data.columns)
        self._set_feature_columns(feature_columns)
        X = train_data[feature_columns].values
        y = train_data[self.target_column].values
        self.model.fit(X, y)
        return self
