from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from . import pipeline_blocks as blocks


def lightGBM(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=False,
                                                             cache_output=False,
                                                             load_persisted_output=False)
        light_gbm = blocks.classifier_light_gbm((features, features_valid),
                                                config,
                                                train_mode, suffix)
    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=False)
        light_gbm = blocks.classifier_light_gbm(features,
                                                config,
                                                train_mode, suffix)

    return light_gbm


def lightGBM_stacking(config, train_mode, suffix=''):
    features = blocks.stacking_features(config, train_mode, suffix,
                                        persist_output=False,
                                        cache_output=False,
                                        load_persisted_output=False)

    light_gbm = blocks.classifier_light_gbm_stacking(features, config, train_mode, suffix,
                                                     cache_output=False)
    return light_gbm


def xgboost(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=True,
                                                             cache_output=True,
                                                             load_persisted_output=True)
        xgb = blocks.classifier_xgb((features, features_valid),
                                    config,
                                    train_mode,
                                    suffix)
    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=True)
        xgb = blocks.classifier_xgb(features,
                                    config,
                                    train_mode,
                                    suffix)

    return xgb


def sklearn_main(config, ClassifierClass, clf_name, train_mode, suffix='', normalize=False):
    model_params = getattr(config, clf_name)
    random_search_config = getattr(config.random_search, clf_name)
    full_config = (config, model_params, random_search_config)
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=True,
                                                             cache_output=True,
                                                             load_persisted_output=True)

        sklearn_preproc = blocks.preprocessing_fillna((features, features_valid), config, train_mode, suffix)
    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=True)
        sklearn_preproc = blocks.preprocessing_fillna(features, config, train_mode, suffix)

    sklearn_clf = blocks.classifier_sklearn(sklearn_preproc,
                                            ClassifierClass,
                                            full_config,
                                            clf_name,
                                            train_mode,
                                            suffix,
                                            normalize)
    return sklearn_clf


PIPELINES = {'lightGBM': lightGBM,
             'lightGBM_stacking': lightGBM_stacking,
             'XGBoost': xgboost,
             'random_forest': {'train': partial(sklearn_main,
                                                ClassifierClass=RandomForestClassifier,
                                                clf_name='random_forest',
                                                train_mode=True),
                               'inference': partial(sklearn_main,
                                                    ClassifierClass=RandomForestClassifier,
                                                    clf_name='random_forest',
                                                    train_mode=False)
                               },
             'log_reg': {'train': partial(sklearn_main,
                                          ClassifierClass=LogisticRegression,
                                          clf_name='logistic_regression',
                                          train_mode=True,
                                          normalize=True),
                         'inference': partial(sklearn_main,
                                              ClassifierClass=LogisticRegression,
                                              clf_name='logistic_regression',
                                              train_mode=False,
                                              normalize=True)
                         },
             'svc': {'train': partial(sklearn_main,
                                      ClassifierClass=SVC,
                                      clf_name='svc',
                                      train_mode=True,
                                      normalize=True),
                     'inference': partial(sklearn_main,
                                          ClassifierClass=SVC,
                                          clf_name='svc',
                                          train_mode=False,
                                          normalize=True)
                     }
             }
