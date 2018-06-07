from functools import partial

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer

import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from models import LightGBMLowMemory as LightGBM
from models import get_sklearn_classifier
from postprocessing import Clipper


def main(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      save_output=True,
                                                      cache_output=True,
                                                      load_saved_output=True)
        light_gbm = classifier_lgbm((features, features_valid),
                                    config,
                                    train_mode)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      cache_output=True)
        light_gbm = classifier_lgbm(features,
                                    config,
                                    train_mode)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[light_gbm],
                   adapter=Adapter({'prediction': E(light_gbm.name, 'prediction')}),
                   cache_dirpath=config.env.cache_dirpath)

    return clipper

def sklearn_main(config, ClassifierClass, clf_name, train_mode, normalize=False):
    model_params = getattr(config, clf_name)
    random_search_config = getattr(config.random_search, clf_name)
    full_config = (config, model_params, random_search_config)
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      save_output=True,
                                                      cache_output=True,
                                                      load_saved_output=True)

        sklearn_preproc = sklearn_preprocessing((features, features_valid), config, train_mode)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      cache_output=True)
        sklearn_preproc = sklearn_preprocessing(features, config, train_mode)

    sklearn_clf = sklearn_classifier(sklearn_preproc, ClassifierClass,
                                    full_config, clf_name,
                                    train_mode, normalize)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[sklearn_clf],
                   adapter=Adapter({'prediction': E(sklearn_clf.name, 'predicted')}),
                   cache_dirpath=config.env.cache_dirpath)

    return clipper

def feature_extraction(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)

        target_encoder, target_encoder_valid = _target_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                                config, train_mode,
                                                                **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[feature_by_type_split],
                                                                  numerical_features_valid=[feature_by_type_split_valid],
                                                                  categorical_features=[target_encoder],
                                                                  categorical_features_valid=[target_encoder_valid],
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  **kwargs)

        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)

        target_encoder = _target_encoders(feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[feature_by_type_split],
                                          numerical_features_valid=[],
                                          categorical_features=[target_encoder],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          **kwargs)

        return feature_combiner


def _feature_by_type_splits(config, train_mode):
    if train_mode:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter=Adapter({'X': E('input', 'X')}),
                                     cache_dirpath=config.env.cache_dirpath)

        feature_by_type_split_valid = Step(name='feature_by_type_split_valid',
                                           transformer=feature_by_type_split,
                                           input_data=['input'],
                                           adapter=Adapter({'X': E('input', 'X_valid')}),
                                           cache_dirpath=config.env.cache_dirpath)

        return feature_by_type_split, feature_by_type_split_valid

    else:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter=Adapter({'X': E('input', 'X')}),
                                     cache_dirpath=config.env.cache_dirpath)

    return feature_by_type_split


def _join_features(numerical_features,
                   numerical_features_valid,
                   categorical_features,
                   categorical_features_valid,
                   config, train_mode,
                   **kwargs):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter=Adapter({
                                  'numerical_feature_list': [
                                      E(feature.name, 'numerical_features') for feature in numerical_features],
                                  'categorical_feature_list': [
                                      E(feature.name, 'categorical_features') for feature in categorical_features],
                              }),
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter=Adapter({
                                        'numerical_feature_list': [
                                            E(feature.name,
                                              'numerical_features') for feature in numerical_features_valid],
                                        'categorical_feature_list': [
                                            E(feature.name,
                                              'categorical_features') for feature in categorical_features_valid],
                                    }),
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter=Adapter({
                                  'numerical_feature_list': [
                                      E(feature.name, 'numerical_features') for feature in numerical_features],
                                  'categorical_feature_list': [
                                      E(feature.name, 'categorical_features') for feature in categorical_features],
                              }),
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

    return feature_joiner


def sklearn_preprocessing(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features
        fillna = Step(name='fillna',
                        transformer=_fillna(-1),
                        input_data=['input'],
                        input_steps=[features_train, features_valid],
                        adapter=Adapter({'X': E(features_train.name, 'features'),
                              'X_valid': E(features_valid.name, 'features'),
                              }),
                        cache_dirpath=config.env.cache_dirpath,
                        **kwargs
                        )
    else:
        fillna = Step(name='fillna',
                        transformer=_fillna(-1),
                        input_data = ['input'],
                        input_steps = [features],
                        adapter = Adapter({'X': E(features.name, 'features')}),
                        cache_dirpath = config.env.cache_dirpath,
                        ** kwargs
                        )
    return fillna


def _fillna(value=-1):
    def _inner_fillna(X, X_valid=None):
        if X_valid is None:
            return {'X': X.fillna(value)}
        return {'X': X.fillna(value), 'X_valid': X_valid.fillna(value)}
    return make_transformer(_inner_fillna)


def classifier_lgbm(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features
        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(LightGBM, config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[NeptuneMonitor(
                                                    **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    SaveResults(
                                                        **config.random_search.light_gbm.callbacks.save_results)
                                                ])
        else:
            transformer = LightGBM(**config.light_gbm)

        light_gbm = Step(name='light_gbm',
                         transformer=transformer,
                         input_data=['input'],
                         input_steps=[features_train, features_valid],
                         adapter=Adapter({'X': E(features_train.name, 'features'),
                                          'y': E('input', 'y'),
                                          'feature_names': E(features_train.name, 'feature_names'),
                                          'categorical_features': E(features_train.name, 'categorical_features'),
                                          'X_valid': E(features_valid.name, 'features'),
                                          'y_valid': E('input', 'y_valid'),
                                          }),
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    else:
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter=Adapter({'X': E(features.name, 'features')}),
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    return light_gbm


def sklearn_classifier(sklearn_features, ClassifierClass, full_config, clf_name, train_mode, normalize, **kwargs):
    config, model_params, rs_config = full_config
    if train_mode:
        if config.random_search.random_forest.n_runs:
            transformer = RandomSearchOptimizer(partial(get_sklearn_classifier, ClassifierClass=ClassifierClass, normalize=normalize),
                                                model_params,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=rs_config.n_runs,
                                                callbacks=[NeptuneMonitor(
                                                    **rs_config.callbacks.neptune_monitor),
                                                    SaveResults(
                                                        **rs_config.callbacks.save_results)
                                                ])
        else:
            transformer = get_sklearn_classifier(ClassifierClass, normalize, **model_params)

        sklearn_clf = Step(name=clf_name,
                         transformer=transformer,
                         input_data=['input'],
                         input_steps=[sklearn_features],
                         adapter=Adapter({'X': E(sklearn_features.name, 'X'),
                                          'y': E('input', 'y'),
                                          'X_valid': E(sklearn_features.name, 'X_valid'),
                                          'y_valid': E('input', 'y_valid'),
                                          }),
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    else:
        sklearn_clf = Step(name=clf_name,
                         transformer = get_sklearn_classifier(ClassifierClass, normalize, **model_params),
                         input_steps=[sklearn_features],
                         adapter=Adapter({'X': E(sklearn_features.name, 'X')}),
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    return sklearn_clf


def _target_encoders(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        numpy_label, numpy_label_valid = _to_numpy_label(config, **kwargs)
        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoder(),
                              input_data=['input'],
                              input_steps=[feature_by_type_split, numpy_label],
                              adapter=Adapter({'X': E(feature_by_type_split.name, 'categorical_features'),
                                               'y': E(numpy_label.name, 'y'),
                                               }),
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        target_encoder_valid = Step(name='target_encoder_valid',
                                    transformer=target_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid, numpy_label_valid],
                                    adapter=Adapter({'X': E(feature_by_type_split_valid.name, 'categorical_features'),
                                                     'y': E(numpy_label_valid.name, 'y'),
                                                     }),
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return target_encoder, target_encoder_valid

    else:
        feature_by_type_split = dispatchers

        target_encoder = Step(name='target_encoder',
                              transformer=fe.TargetEncoder(),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter=Adapter({'X': E(feature_by_type_split.name, 'categorical_features')}),
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return target_encoder


def _to_numpy_label(config, **kwargs):
    to_numpy_label = Step(name='to_numpy_label',
                          transformer=fe.ToNumpyLabel(),
                          input_data=['input'],
                          adapter=Adapter({'y': [E('input', 'y')]}),
                          cache_dirpath=config.env.cache_dirpath,
                          **kwargs)

    to_numpy_label_valid = Step(name='to_numpy_label_valid',
                                transformer=to_numpy_label,
                                input_data=['input'],
                                adapter=Adapter({'y': [E('input', 'y_valid')]}),
                                cache_dirpath=config.env.cache_dirpath,
                                **kwargs)

    return to_numpy_label, to_numpy_label_valid


PIPELINES = {'main': {'train': partial(main, train_mode=True),
                      'inference': partial(main, train_mode=False)},

             'random_forest': {'train': partial(sklearn_main, ClassifierClass=RandomForestClassifier, clf_name='random_forest', train_mode=True),
                               'inference': partial(sklearn_main, ClassifierClass=RandomForestClassifier, clf_name='random_forest', train_mode=False)},
             'log_reg': {'train': partial(sklearn_main, ClassifierClass=LogisticRegression, clf_name='logistic_regression', train_mode=True, normalize=True),
                         'inference': partial(sklearn_main, ClassifierClass=LogisticRegression, clf_name='logistic_regression', train_mode=False, normalize=True)},
             #'SVC': {'train': partial(sklearn_main, ClassifierClass=SVC, clf_name='SVC', train_mode=True, normalize=True),
             #            'inference': partial(sklearn_main, ClassifierClass=SVC, clf_name='SVC', train_mode=False, normalize=True)}
             }
