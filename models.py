import xgboost as xgb
from attrdict import AttrDict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from toolkit.sklearn_transformers.models import SklearnClassifier

logger = get_logger()


class XGBoost(BaseTransformer):
    def __init__(self, **params):
        super().__init__()
        logger.info('initializing XGBoost...')
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
        self.estimator = xgb.train(params=self.model_config,
                                   dtrain=train,
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
        self.estimator = xgb.Booster(params=self.model_config)
        self.estimator.load_model(filepath)
        return self

    def persist(self, filepath):
        self.estimator.save_model(filepath)


def get_sklearn_classifier(ClassifierClass, normalize=False, **kwargs):

    class SklearnBinaryClassifier(SklearnClassifier):
        def transform(self, X, y=None, target=1, **kwargs):
            prediction = self.estimator.predict_proba(X)[:, target]
            return {SklearnClassifier.RESULT_KEY: prediction}

    if normalize:
        return SklearnBinaryClassifier(Pipeline([('standarizer', StandardScaler()),
                                                 ('classifier', ClassifierClass(**kwargs))]))

    return SklearnBinaryClassifier(ClassifierClass(**kwargs))
