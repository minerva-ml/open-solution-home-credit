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
    if train_mode:
        features, features_valid = blocks.feature_extraction(config,
                                                             train_mode,
                                                             suffix,
                                                             persist_output=True,
                                                             cache_output=True,
                                                             load_persisted_output=True)

        sklearn_preproc, sklearn_preproc_valid = blocks.sklearn_preprocessing(features,
                                                                              features_valid,
                                                                              config,
                                                                              train_mode,
                                                                              normalize,
                                                                              suffix)
        sklearn_clf = blocks.classifier_sklearn((sklearn_preproc, sklearn_preproc_valid),
                                                ClassifierClass,
                                                config,
                                                clf_name,
                                                train_mode,
                                                suffix)

    else:
        features = blocks.feature_extraction(config,
                                             train_mode,
                                             suffix,
                                             cache_output=True)
        sklearn_preproc = blocks.sklearn_preprocessing(features, [], config, train_mode, normalize, suffix)

        sklearn_clf = blocks.classifier_sklearn(sklearn_preproc,
                                                ClassifierClass,
                                                config,
                                                clf_name,
                                                train_mode,
                                                suffix)
    return sklearn_clf


PIPELINES = {'lightGBM': lightGBM,
             'catboost': catboost,
             'lightGBM_stacking': lightGBM_stacking,
             'XGBoost': xgboost,
             'random_forest': partial(sklearn_main,
                                      ClassifierClass=RandomForestClassifier,
                                      clf_name='random_forest'),
             'log_reg': partial(sklearn_main,
                                ClassifierClass=LogisticRegression,
                                clf_name='logistic_regression'),
             'svc': partial(sklearn_main,
                            ClassifierClass=SVC,
                            clf_name='SVC'),
             }
