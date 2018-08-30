from attrdict import AttrDict
from deepsense import neptune

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from toolkit.keras_transformers.models import ClassifierXY
from toolkit.sklearn_transformers.models import SklearnClassifier
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.regularizers import l1_l2
import xgboost as xgb
import catboost as ctb

from .callbacks import neptune_monitor_lgbm, NeptuneMonitor
from .utils import get_logger

logger = get_logger()
ctx = neptune.Context()


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


class LightGBM(BaseTransformer):
    def __init__(self, name=None, **params):
        super().__init__()
        logger.info('initializing LightGBM...')
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_function = None
        if params['callbacks_on']:
            self.callbacks = callbacks(channel_prefix=name)
        else:
            self.callbacks = []

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            feature_names='auto',
            categorical_features='auto',
            **kwargs):
        evaluation_results = {}

        self._check_target_shape_and_type(y, 'y')
        self._check_target_shape_and_type(y_valid, 'y_valid')
        y = self._format_target(y)
        y_valid = self._format_target(y_valid)

        logger.info('LightGBM, train data shape        {}'.format(X.shape))
        logger.info('LightGBM, validation data shape   {}'.format(X_valid.shape))
        logger.info('LightGBM, train labels shape      {}'.format(y.shape))
        logger.info('LightGBM, validation labels shape {}'.format(y_valid.shape))

        data_train = lgb.Dataset(data=X,
                                 label=y,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)
        data_valid = lgb.Dataset(X_valid,
                                 label=y_valid,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)

        self.estimator = lgb.train(self.model_config,
                                   data_train,
                                   feature_name=feature_names,
                                   categorical_feature=categorical_features,
                                   valid_sets=[data_train, data_valid],
                                   valid_names=['data_train', 'data_valid'],
                                   evals_result=evaluation_results,
                                   num_boost_round=self.training_config.number_boosting_rounds,
                                   early_stopping_rounds=self.training_config.early_stopping_rounds,
                                   verbose_eval=self.model_config.verbose,
                                   feval=self.evaluation_function,
                                   callbacks=self.callbacks,
                                   **kwargs)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict(X)
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def _check_target_shape_and_type(self, target, name):
        if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target)))
        try:
            assert len(target.shape) == 1, '"{}" must be 1-D. It is {}-D instead.'.format(name,
                                                                                          len(target.shape))
        except AttributeError:
            print('Cannot determine shape of the {}. '
                  'Type must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead'.format(name,
                                                                                                     type(target)))

    def _format_target(self, target):

        if isinstance(target, pd.Series):
            return target.values
        elif isinstance(target, np.ndarray):
            return target
        elif isinstance(target, list):
            return np.array(target)
        else:
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target)))


class CatBoost(BaseTransformer):
    def __init__(self, **kwargs):
        self.estimator = ctb.CatBoostClassifier(**kwargs)

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            feature_names=None,
            categorical_features=None,
            **kwargs):

        logger.info('Catboost, train data shape        {}'.format(X.shape))
        logger.info('Catboost, validation data shape   {}'.format(X_valid.shape))
        logger.info('Catboost, train labels shape      {}'.format(y.shape))
        logger.info('Catboost, validation labels shape {}'.format(y_valid.shape))

        categorical_indeces = self._get_categorical_indeces(feature_names, categorical_features)
        self.estimator.fit(X, y,
                           eval_set=(X_valid, y_valid),
                           cat_features=categorical_indeces)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict_proba(X)[:, 1]
        return {'prediction': prediction}

    def load(self, filepath):
        self.estimator.load_model(filepath)
        return self

    def persist(self, filepath):
        self.estimator.save_model(filepath)

    def _get_categorical_indeces(self, feature_names, categorical_features):
        if categorical_features:
            return [feature_names.index(feature) for feature in categorical_features]
        else:
            return None


def get_sklearn_classifier(ClassifierClass, **kwargs):
    class SklearnBinaryClassifier(SklearnClassifier):

        def transform(self, X, y=None, target=1, **kwargs):
            prediction = self.estimator.predict_proba(X)[:, target]
            return {'prediction': prediction}

    return SklearnBinaryClassifier(ClassifierClass(**kwargs))


def callbacks(channel_prefix):
    neptune_monitor = neptune_monitor_lgbm(channel_prefix)
    return [neptune_monitor]


class NeuralNetwork(ClassifierXY):
    def __init__(self, architecture_config, training_config, callbacks_config, **kwargs):
        super().__init__(architecture_config, training_config, callbacks_config)
        logger.info('initializing Neural Network...')
        self.params = kwargs
        self.name = 'NeuralNetwork{}'.format(kwargs['suffix'])
        self.model_params = architecture_config['model_params']
        self.optimizer_params = architecture_config['optimizer_params']

    def _build_optimizer(self, **kwargs):
        return Adam(**self.optimizer_params)

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _build_model(self, input_shape, **kwargs):
        K.clear_session()
        model = Sequential()
        for layer in range(self.model_params['layers']):
            config = {key: val[layer] for key, val in self.model_params.items() if key != 'layers'}
            if layer == 0:
                model.add(Dense(config['neurons'],
                                kernel_regularizer=l1_l2(l1=config['l1'], l2=config['l2']),
                                input_shape=input_shape))
            else:
                model.add(Dense(config['neurons'],
                                kernel_regularizer=l1_l2(l1=config['l1'], l2=config['l2'])))
            if config['batch_norm']:
                model.add(BatchNormalization())
            model.add(Activation(config['activation']))
            model.add(Dropout(config['dropout']))

        return model

    def _compile_model(self, input_shape):
        model = self._build_model(input_shape)
        optimizer = self._build_optimizer()
        loss = self._build_loss()
        model.compile(optimizer=optimizer, loss=loss)

        return model

    def fit(self, X, y, validation_data, *args, **kwargs):
        self.callbacks = self._create_callbacks()
        self.model = self._compile_model(input_shape=(X.shape[1], ))
        self.model.fit(X, y,
                       validation_data=validation_data,
                       verbose=1,
                       callbacks=self.callbacks,
                       **self.training_config)
        return self

    def _create_callbacks(self, **kwargs):
        neptune = NeptuneMonitor(self.name)
        return [neptune]

    def transform(self, X, y=None, validation_data=None, *args, **kwargs):
        predictions = self.model.predict(X, verbose=1)
        return {'predicted': np.array([x[0] for x in predictions])}
