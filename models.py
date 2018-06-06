import lightgbm as lgb
import numpy as np
from steppy.adapters import to_numpy_label_inputs
from toolkit.misc import LightGBM
from steppy.base import make_transformer
from toolkit.sklearn_recipes.models import SklearnClassifier


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


def sklearn_preprocess():
    def sklearn_preprocessing(X, X_valid=None):
        if X_valid is None:
            return {'X': X.fillna(0)}
        return {'X': X.fillna(0), 'X_valid': X_valid.fillna(0)}
    return make_transformer(sklearn_preprocessing)


class SklearnBinaryClassifier(SklearnClassifier):
    def transform(self, X, y=None, target=1, **kwargs):
        prediction = self.estimator.predict_proba(X)[:, target]
        return {SklearnClassifier.RESULT_KEY: prediction}


def get_sklearn_classifier(ClassifierClass, **kwargs):
    return SklearnBinaryClassifier(ClassifierClass(**kwargs))