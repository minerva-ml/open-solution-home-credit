from functools import partial

from sklearn.metrics import roc_auc_score
from steppy.adapter import Adapter, E
from steppy.base import Step
from utils import ToNumpyLabel

import feature_extraction as fe
from hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, SaveResults
from models import LightGBMLowMemory as LightGBM
from postprocessing import Clipper


def lightGBM(config, train_mode):
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


def feature_extraction(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)

        bureau, bureau_valid = _bureau(config, train_mode, **kwargs)

        categorical_encoder, categorical_encoder_valid = _categorical_encoders((feature_by_type_split, feature_by_type_split_valid),
                                                                config, train_mode,
                                                                **kwargs)

        groupby_aggregation, groupby_aggregation_valid = _groupby_aggregations(
            (feature_by_type_split, feature_by_type_split_valid), config, train_mode, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[feature_by_type_split,
                                                                                      groupby_aggregation,
                                                                                      bureau],
                                                                  numerical_features_valid=[feature_by_type_split_valid,
                                                                                            groupby_aggregation_valid,
                                                                                            bureau_valid],
                                                                  categorical_features=[categorical_encoder],
                                                                  categorical_features_valid=[categorical_encoder_valid],
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  **kwargs)

        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)

        bureau = _bureau(config, train_mode, **kwargs)

        categorical_encoder = _categorical_encoders(feature_by_type_split, config, train_mode, **kwargs)

        groupby_aggregation = _groupby_aggregations(feature_by_type_split, config, train_mode, **kwargs)

        feature_combiner = _join_features(numerical_features=[feature_by_type_split, groupby_aggregation, bureau],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_encoder],
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


def _categorical_encoders(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        numpy_label, numpy_label_valid = _to_numpy_label(config, **kwargs)
        categorical_encoder = Step(name='categorcial_encoder',
                              transformer=fe.CategoricalEncoder(),
                              input_data=['input'],
                              input_steps=[feature_by_type_split, numpy_label],
                              adapter=Adapter({'X': E(feature_by_type_split.name, 'categorical_features'),
                                               'y': E(numpy_label.name, 'y'),
                                               }),
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        categorical_encoder_valid = Step(name='categorical_encoder_valid',
                                    transformer=categorical_encoder,
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split_valid, numpy_label_valid],
                                    adapter=Adapter({'X': E(feature_by_type_split_valid.name, 'categorical_features'),
                                                     'y': E(numpy_label_valid.name, 'y'),
                                                     }),
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return categorical_encoder, categorical_encoder_valid

    else:
        feature_by_type_split = dispatchers

        categorical_encoder = Step(name='categorical_encoder',
                              transformer=fe.CategoricalEncoder(),
                              input_data=['input'],
                              input_steps=[feature_by_type_split],
                              adapter=Adapter({'X': E(feature_by_type_split.name, 'categorical_features')}),
                              cache_dirpath=config.env.cache_dirpath,
                              **kwargs)

        return categorical_encoder


def _groupby_aggregations(dispatchers, config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split],
                                    adapter=Adapter({'categorical_features': E(feature_by_type_split.name,
                                                                               'categorical_features'),
                                                     'numerical_features': E(feature_by_type_split.name,
                                                                             'numerical_features')
                                                     }),
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        groupby_aggregations_valid = Step(name='groupby_aggregations_valid',
                                          transformer=groupby_aggregations,
                                          input_data=['input'],
                                          input_steps=[feature_by_type_split_valid],
                                          adapter=Adapter({'categorical_features': E(feature_by_type_split_valid.name,
                                                                                     'categorical_features'),
                                                           'numerical_features': E(feature_by_type_split_valid.name,
                                                                                   'numerical_features')
                                                           }),
                                          cache_dirpath=config.env.cache_dirpath,
                                          **kwargs)

        return groupby_aggregations, groupby_aggregations_valid

    else:
        feature_by_type_split = dispatchers
        groupby_aggregations = Step(name='groupby_aggregations',
                                    transformer=fe.GroupbyAggregations(**config.groupby_aggregation),
                                    input_data=['input'],
                                    input_steps=[feature_by_type_split],
                                    adapter=Adapter({'categorical_features': E(feature_by_type_split.name,
                                                                               'categorical_features'),
                                                     'numerical_features': E(feature_by_type_split.name,
                                                                             'numerical_features')
                                                     }),
                                    cache_dirpath=config.env.cache_dirpath,
                                    **kwargs)

        return groupby_aggregations


def _bureau(config, train_mode, **kwargs):
    if train_mode:
        bureau = Step(name='bureau',
                      transformer=fe.GroupbyAggregationFromFile(**config.bureau),
                      input_data=['input'],
                      adapter=Adapter({'X': E('input', 'X')}),
                      cache_dirpath=config.env.cache_dirpath,
                      **kwargs)

        bureau_valid = Step(name='bureau_valid',
                            transformer=bureau,
                            input_data=['input'],
                            adapter=Adapter({'X': E('input', 'X_valid')}),
                            cache_dirpath=config.env.cache_dirpath,
                            **kwargs)

        return bureau, bureau_valid

    else:
        bureau = Step(name='bureau',
                      transformer=fe.GroupbyAggregationFromFile(**config.bureau),
                      input_data=['input'],
                      adapter=Adapter({'X': E('input', 'X')}),
                      cache_dirpath=config.env.cache_dirpath,
                      **kwargs)

        return bureau


def _to_numpy_label(config, **kwargs):
    to_numpy_label = Step(name='to_numpy_label',
                          transformer=ToNumpyLabel(),
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


PIPELINES = {'lightGBM': {'train': partial(lightGBM, train_mode=True),
                          'inference': partial(lightGBM, train_mode=False)},
             }
