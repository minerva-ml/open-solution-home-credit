import lightgbm as lgb
import xgboost as xgb
import numpy as np
from steppy.adapters import to_numpy_label_inputs
from toolkit.misc import LightGBM


class LightGBMLowMemory(LightGBM):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, categorical_features=None, **kwargs):
        y = to_numpy_label_inputs([y])
        y_valid = to_numpy_label_inputs([y_valid])

        X = X[feature_names].values.astype(np.float32)
        y = y.astype(np.float32)

        X_valid = X_valid[feature_names].values.astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        train = lgb.Dataset(X, label=y)
        valid = lgb.Dataset(X_valid, label=y_valid)

        self.evaluation_results = {}
        self.estimator = lgb.train(self.model_config,
                                   train, valid_sets=[valid], valid_names=['valid'],
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   evals_result=self.evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self


from attrdict import AttrDict
from sklearn.externals import joblib
from steppy.base import BaseTransformer


class XGBoost(BaseTransformer):
    def __init__(self, **params):
        super().__init__()
        self.params = params
        self.training_params = ['nrounds', 'early_stopping_rounds']
        self.evaluation_function = None

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
            X, y,
            X_valid, y_valid,
            feature_names=None,
            feature_types=None,
            **kwargs):
        train = xgb.DMatrix(X,
                            label=y,
                            feature_names=feature_names,
                            feature_types=feature_types)
        valid = xgb.DMatrix(X_valid,
                            label=y_valid,
                            feature_names=feature_names,
                            feature_types=feature_types)

        evaluation_results = {}
        self.estimator = xgb.train(self.model_config,
                                   train,
                                   evals=[(train, 'train'), (valid, 'valid')],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.nrounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self

    def transform(self, X, y=None, feature_names=None, feature_types=None, **kwargs):
        X_DMatrix = xgb.DMatrix(X,
                                label=y,
                                feature_names=feature_names,
                                feature_types=feature_types)
        prediction = self.estimator.predict(X_DMatrix)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.estimator, filepath)


class XGBoostLowMemory(XGBoost):
    def fit(self, X, y, X_valid, y_valid, feature_names=None, feature_types=None, **kwargs):
        y = to_numpy_label_inputs([y])
        y_valid = to_numpy_label_inputs([y_valid])

        X = X[feature_names].values.astype(np.float32)
        y = y.astype(np.float32)

        X_valid = X_valid[feature_names].values.astype(np.float32)
        y_valid = y_valid.astype(np.float32)

        train = xgb.DMatrix(X,
                            label=y,
                            feature_names=feature_names,
                            feature_types=feature_types)
        valid = xgb.DMatrix(X_valid,
                            label=y_valid,
                            feature_names=feature_names,
                            feature_types=feature_types)

        self.evaluation_results = {}
        self.estimator = xgb.train(self.model_config,
                                   train,
                                   evals=[(valid, 'valid')],
                                   evals_result=self.evaluation_results,
                                   num_boost_round=self.training_config.nrounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function)
        return self
