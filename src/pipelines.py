from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from . import pipeline_blocks as blocks


def light_gbm(config, train_mode, suffix=''):
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


def catboost(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=False,
                                                             cache_output=False,
                                                             load_persisted_output=False)
        catboost = blocks.classifier_catboost((features, features_valid),
                                              config,
                                              train_mode, suffix)
    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=False)
        catboost = blocks.classifier_catboost(features,
                                              config,
                                              train_mode, suffix)

    return catboost


def lightGBM_stacking(config, train_mode, suffix=''):
    features = blocks.stacking_features(config, train_mode, suffix,
                                        persist_output=False,
                                        cache_output=False,
                                        load_persisted_output=False)

    light_gbm = blocks.classifier_light_gbm_stacking(features, config, train_mode, suffix,
                                                     cache_output=False)
    return light_gbm


def log_reg_stacking(config, train_mode, suffix=''):
    features = blocks.stacking_features(config, train_mode, suffix,
                                        persist_output=False,
                                        cache_output=False,
                                        load_persisted_output=False)

    normalized_features = blocks.stacking_normalization(features, config, train_mode, suffix,
                                                        persist_output=False,
                                                        cache_output=False,
                                                        load_persisted_output=False
                                                        )
    log_reg = blocks.classifier_log_reg_stacking(normalized_features, config, train_mode, suffix,
                                                 cache_output=False)
    return log_reg


def xgboost(config, train_mode, suffix=''):
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=True,
                                                             cache_output=True,
                                                             load_persisted_output=True)

        xgb_features, xgb_features_valid = blocks.xgb_preprocessing((features, features_valid),
                                                                    config,
                                                                    train_mode,
                                                                    suffix,
                                                                    persist_output=True,
                                                                    cache_output=True,
                                                                    load_persisted_output=True)

        xgb = blocks.classifier_xgb((xgb_features, xgb_features_valid),
                                    config,
                                    train_mode,
                                    suffix)
    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=True)
        xgb_features = blocks.xgb_preprocessing(features,
                                                config,
                                                train_mode,
                                                suffix,
                                                cache_output=True)
        xgb = blocks.classifier_xgb(xgb_features,
                                    config,
                                    train_mode,
                                    suffix)

    return xgb


def sklearn_pipeline(config, ClassifierClass, clf_name, train_mode, suffix='', normalize=False):
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=True,
                                                             cache_output=True,
                                                             load_persisted_output=True)

        sklearn_features = blocks.sklearn_preprocessing((features, features_valid),
                                                        config,
                                                        train_mode,
                                                        suffix,
                                                        normalize,
                                                        persist_output=True,
                                                        cache_output=True,
                                                        load_persisted_output=True)
    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=True)

        sklearn_features = blocks.sklearn_preprocessing(features,
                                                        config,
                                                        train_mode,
                                                        suffix,
                                                        normalize,
                                                        cache_output=True)

    sklearn_clf = blocks.classifier_sklearn(sklearn_features,
                                            ClassifierClass,
                                            config,
                                            clf_name,
                                            train_mode,
                                            suffix)
    return sklearn_clf


def xgboost_stacking(config, train_mode, suffix=''):
    features = blocks.stacking_features(config, train_mode, suffix,
                                        persist_output=False,
                                        cache_output=False,
                                        load_persisted_output=False)

    xgboost = blocks.classifier_xgboost_stacking(features, config, train_mode, suffix,
                                                 cache_output=False)
    return xgboost


def log_reg_stacking(config, train_mode, suffix=''):
    features = blocks.stacking_features(config, train_mode, suffix,
                                        persist_output=False,
                                        cache_output=False,
                                        load_persisted_output=False)

    normalized_features = blocks.stacking_normalization(features, config, train_mode, suffix,
                                                        persist_output=False,
                                                        cache_output=False,
                                                        load_persisted_output=False
                                                        )
    log_reg = blocks.classifier_log_reg_stacking(normalized_features, config, train_mode, suffix,
                                                 cache_output=False)
    return log_reg


def light_gbm_stacking(config, train_mode, suffix=''):
    features = blocks.stacking_features(config, train_mode, suffix,
                                        persist_output=False,
                                        cache_output=False,
                                        load_persisted_output=False)

    light_gbm = blocks.classifier_light_gbm_stacking(features, config, train_mode, suffix,
                                                     cache_output=False)
    return light_gbm


PIPELINES = {'lightGBM': light_gbm,
             'catboost': catboost,
             'XGBoost': xgboost,
             'random_forest': partial(sklearn_pipeline,
                                      ClassifierClass=RandomForestClassifier,
                                      clf_name='random_forest'),
             'log_reg': partial(sklearn_pipeline,
                                ClassifierClass=LogisticRegression,
                                clf_name='logistic_regression',
                                normalize=True),
             'svc': partial(sklearn_pipeline,
                            ClassifierClass=SVC,
                            clf_name='svc',
                            normalize=True),
             'XGBoost_stacking': xgboost_stacking,
             'lightGBM_stacking': light_gbm_stacking,
             'log_reg_stacking': log_reg_stacking,
             }
