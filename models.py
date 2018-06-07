import lightgbm as lgb
import numpy as np
from steppy.adapters import to_numpy_label_inputs
from toolkit.misc import LightGBM
from toolkit.sklearn_recipes.models import SklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def get_sklearn_classifier(ClassifierClass, normalize=False, **kwargs):

    class SklearnBinaryClassifier(SklearnClassifier):
        def transform(self, X, y=None, target=1, **kwargs):
            prediction = self.estimator.predict_proba(X)[:, target]
            return {SklearnClassifier.RESULT_KEY: prediction}

    if normalize:
        return SklearnBinaryClassifier(Pipeline([('standarizer', StandardScaler()),
                                                 ('classifier', ClassifierClass(**kwargs))]))

    return SklearnBinaryClassifier(ClassifierClass(**kwargs))