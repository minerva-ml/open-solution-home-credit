from functools import partial

import feature_extraction as fe
from models import LightGBMLowMemory as LightGBM
from steps.adapters import to_numpy_label_inputs, identity_inputs
from steps.base import Step, Dummy
from postprocessing import Clipper


def main(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction(config, train_mode,
                                                      save_output=True, cache_output=True, load_saved_output=True)
        light_gbm = classifier_lgbm((features, features_valid), config, train_mode)
    else:
        features = feature_extraction(config, train_mode, cache_output=True)
        light_gbm = classifier_lgbm(features, config, train_mode)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[light_gbm],
                   adapter={'prediction': ([(light_gbm.name, 'prediction')]),
                            },
                   cache_dirpath=config.env.cache_dirpath)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[clipper],
                  adapter={'y_pred': ([(clipper.name, 'clipped_prediction')]),
                           },
                  cache_dirpath=config.env.cache_dirpath)
    return output


def feature_extraction(config, train_mode, **kwargs):
    if train_mode:
        feature_by_type_split, feature_by_type_split_valid = _feature_by_type_splits(config, train_mode)

        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[feature_by_type_split],
                                                                  numerical_features_valid=[feature_by_type_split_valid],
                                                                  categorical_features=[],
                                                                  categorical_features_valid=[],
                                                                  config=config, train_mode=train_mode, **kwargs)

        return feature_combiner, feature_combiner_valid
    else:
        feature_by_type_split = _feature_by_type_splits(config, train_mode)

        feature_combiner = _join_features(numerical_features=[feature_by_type_split],
                                          numerical_features_valid=[],
                                          categorical_features=[],
                                          categorical_features_valid=[],
                                          config=config, train_mode=train_mode, **kwargs)

        return feature_combiner


def _feature_by_type_splits(config, train_mode):
    if train_mode:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter={'X': ([('input', 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

        feature_by_type_split_valid = Step(name='feature_by_type_split_valid',
                                           transformer=feature_by_type_split,
                                           input_data=['input'],
                                           adapter={'X': ([('input', 'X_valid')]),
                                                    },
                                           cache_dirpath=config.env.cache_dirpath)

        return feature_by_type_split, feature_by_type_split_valid

    else:
        feature_by_type_split = Step(name='feature_by_type_split',
                                     transformer=fe.DataFrameByTypeSplitter(**config.dataframe_by_type_splitter),
                                     input_data=['input'],
                                     adapter={'X': ([('input', 'X')]),
                                              },
                                     cache_dirpath=config.env.cache_dirpath)

    return feature_by_type_split


def _join_features(numerical_features, numerical_features_valid,
                   categorical_features, categorical_features_valid,
                   config, train_mode, **kwargs):
    if train_mode:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

        feature_joiner_valid = Step(name='feature_joiner_valid',
                                    transformer=feature_joiner,
                                    input_steps=numerical_features_valid + categorical_features_valid,
                                    adapter={'numerical_feature_list': (
                                        [(feature.name, 'numerical_features') for feature in numerical_features_valid],
                                        identity_inputs),
                                        'categorical_feature_list': (
                                            [(feature.name, 'categorical_features') for feature in
                                             categorical_features_valid],
                                            identity_inputs),
                                    },
                                    cache_dirpath=config.env.cache_dirpath, **kwargs)

        return feature_joiner, feature_joiner_valid

    else:
        feature_joiner = Step(name='feature_joiner',
                              transformer=fe.FeatureJoiner(),
                              input_steps=numerical_features + categorical_features,
                              adapter={
                                  'numerical_feature_list': (
                                      [(feature.name, 'numerical_features') for feature in numerical_features],
                                      identity_inputs),
                                  'categorical_feature_list': (
                                      [(feature.name, 'categorical_features') for feature in categorical_features],
                                      identity_inputs),
                              },
                              cache_dirpath=config.env.cache_dirpath, **kwargs)

    return feature_joiner


def classifier_lgbm(features, config, train_mode, **kwargs):
    if train_mode:
        features_train, features_valid = features

        transformer = LightGBM(**config.light_gbm)

        light_gbm = Step(name='light_gbm',
                         transformer=transformer,
                         input_data=['input'],
                         input_steps=[features_train, features_valid],
                         adapter={'X': ([(features_train.name, 'features')]),
                                  'y': ([('input', 'y')], to_numpy_label_inputs),
                                  'feature_names': ([(features_train.name, 'feature_names')]),
                                  'categorical_features': ([(features_train.name, 'categorical_features')]),
                                  'X_valid': ([(features_valid.name, 'features')]),
                                  'y_valid': ([('input', 'y_valid')], to_numpy_label_inputs),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    else:
        light_gbm = Step(name='light_gbm',
                         transformer=LightGBM(**config.light_gbm),
                         input_steps=[features],
                         adapter={'X': ([(features.name, 'features')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath,
                         **kwargs)
    return light_gbm


PIPELINES = {'main': {'train': partial(main, train_mode=True),
                      'inference': partial(main, train_mode=False)},
             }
