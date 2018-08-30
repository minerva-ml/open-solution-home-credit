from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

from . import pipeline_blocks as blocks


def light_gbm(config, train_mode, suffix=''):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    features = blocks.feature_extraction(config,
                                         train_mode,
                                         suffix,
                                         persist_output=persist_output,
                                         cache_output=cache_output,
                                         load_persisted_output=load_persisted_output
                                         )
    light_gbm = blocks.classifier_light_gbm(features,
                                            config,
                                            train_mode, suffix)

    return light_gbm


def catboost(config, train_mode, suffix=''):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    features = blocks.feature_extraction(config,
                                         train_mode,
                                         suffix,
                                         persist_output=persist_output,
                                         cache_output=cache_output,
                                         load_persisted_output=load_persisted_output
                                         )

    catboost_features = blocks.catboost_preprocessing(features,
                                                      config,
                                                      train_mode,
                                                      suffix,
                                                      persist_output=persist_output,
                                                      cache_output=cache_output,
                                                      load_persisted_output=load_persisted_output)

    catboost = blocks.classifier_catboost(catboost_features,
                                          config,
                                          train_mode, suffix)

    return catboost


def xgboost(config, train_mode, suffix=''):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    features = blocks.feature_extraction(config,
                                         train_mode,
                                         suffix,
                                         persist_output=persist_output,
                                         cache_output=cache_output,
                                         load_persisted_output=load_persisted_output)
    xgb_features = blocks.xgb_preprocessing(features,
                                            config,
                                            train_mode,
                                            suffix,
                                            persist_output=persist_output,
                                            cache_output=cache_output,
                                            load_persisted_output=load_persisted_output)
    xgb = blocks.classifier_xgb(xgb_features,
                                config,
                                train_mode,
                                suffix)

    return xgb


def neural_network(config, train_mode, suffix='', normalize=False):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False
    features = blocks.feature_extraction(config,
                                         train_mode,
                                         suffix,
                                         persist_output=persist_output,
                                         cache_output=cache_output,
                                         load_persisted_output=load_persisted_output)
    neural_network_features = blocks.sklearn_preprocessing(features,
                                                           config,
                                                           train_mode,
                                                           suffix,
                                                           normalize,
                                                           persist_output=persist_output,
                                                           cache_output=cache_output,
                                                           load_persisted_output=load_persisted_output)
    neural_network = blocks.classifier_nn(neural_network_features,
                                          config,
                                          train_mode, suffix)

    return neural_network


def neural_network_stacking(config, train_mode, suffix='', normalize=False, use_features=False):
    features_oof_preds = blocks.oof_predictions(config, train_mode, suffix,
                                                persist_output=False,
                                                cache_output=False,
                                                load_persisted_output=False)
    if use_features:
        features_engineered = blocks.feature_extraction(config,
                                                        train_mode,
                                                        suffix,
                                                        persist_output=False,
                                                        cache_output=False,
                                                        load_persisted_output=False)
        features = blocks.concat_features([features_oof_preds, features_engineered],
                                          config,
                                          train_mode,
                                          suffix,
                                          persist_output=False,
                                          cache_output=False,
                                          load_persisted_output=False)
        features = blocks.sklearn_preprocessing(features,
                                                config,
                                                train_mode,
                                                suffix,
                                                normalize,
                                                persist_output=persist_output,
                                                cache_output=cache_output,
                                                load_persisted_output=load_persisted_output)
    else:
        features = features_oof_preds

    neural_network = blocks.classifier_nn(features,
                                          config,
                                          train_mode, suffix)

    return neural_network


def sklearn_pipeline(config, ClassifierClass, clf_name, train_mode, suffix='', normalize=False):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    features = blocks.feature_extraction(config,
                                         train_mode,
                                         suffix,
                                         persist_output=persist_output,
                                         cache_output=cache_output,
                                         load_persisted_output=load_persisted_output)

    sklearn_features = blocks.sklearn_preprocessing(features,
                                                    config,
                                                    train_mode,
                                                    suffix,
                                                    normalize,
                                                    persist_output=persist_output,
                                                    cache_output=cache_output,
                                                    load_persisted_output=load_persisted_output)

    sklearn_clf = blocks.classifier_sklearn(sklearn_features,
                                            ClassifierClass,
                                            config,
                                            clf_name,
                                            train_mode,
                                            suffix)
    return sklearn_clf


def light_gbm_stacking(config, train_mode, suffix='', use_features=False):
    features_oof_preds = blocks.oof_predictions(config, train_mode, suffix,
                                                persist_output=False,
                                                cache_output=False,
                                                load_persisted_output=False)
    if use_features:
        features_engineered = blocks.feature_extraction(config, train_mode, suffix,
                                                        persist_output=False,
                                                        cache_output=False,
                                                        load_persisted_output=False)
        features = blocks.concat_features([features_oof_preds, features_engineered], config, train_mode, suffix,
                                          persist_output=False,
                                          cache_output=False,
                                          load_persisted_output=False)
    else:
        features = features_oof_preds

    light_gbm = blocks.classifier_light_gbm(features, config, train_mode, suffix)

    return light_gbm


def xgboost_stacking(config, train_mode, suffix='', use_features=False):
    if use_features:
        raise NotImplementedError
    features = blocks.oof_predictions(config, train_mode, suffix,
                                      persist_output=False,
                                      cache_output=False,
                                      load_persisted_output=False)
    xgboost = blocks.classifier_xgb(features, config, train_mode, suffix,
                                    cache_output=False)
    return xgboost


def sklearn_pipeline_stacking(config, ClassifierClass, clf_name, train_mode, suffix=''):
    features = blocks.oof_predictions(config, train_mode, suffix,
                                      persist_output=False,
                                      cache_output=False,
                                      load_persisted_output=False)

    normalized_features = blocks.stacking_normalization(features, config, train_mode, suffix,
                                                        persist_output=False,
                                                        cache_output=False,
                                                        load_persisted_output=False
                                                        )
    log_reg = blocks.classifier_sklearn(normalized_features,
                                        ClassifierClass,
                                        config,
                                        clf_name,
                                        train_mode,
                                        suffix)
    return log_reg


PIPELINES = {'lightGBM': light_gbm,
             'catboost': catboost,
             'XGBoost': xgboost,
             'neural_network': partial(neural_network,
                                       normalize=True),
             'random_forest': partial(sklearn_pipeline,
                                      ClassifierClass=RandomForestClassifier,
                                      clf_name='random_forest'),
             'log_reg': partial(sklearn_pipeline,
                                ClassifierClass=LogisticRegression,
                                clf_name='log_reg',
                                normalize=True),
             'naive_bayes': partial(sklearn_pipeline,
                                    ClassifierClass=BernoulliNB,
                                    clf_name='naive_bayes',
                                    normalize=True),
             'svc': partial(sklearn_pipeline,
                            ClassifierClass=SVC,
                            clf_name='svc',
                            normalize=True),
             'neural_network_stacking': partial(neural_network_stacking,
                                                normalize=False,
                                                use_features=False),
             'neural_network_stacking_with_features': partial(neural_network_stacking,
                                                              normalize=False,
                                                              use_features=True),
             'XGBoost_stacking_with_features': partial(xgboost_stacking,
                                                       use_features=True),
             'XGBoost_stacking': partial(xgboost_stacking,
                                         use_features=False),
             'lightGBM_stacking_with_features': partial(light_gbm_stacking,
                                                        use_features=True),
             'lightGBM_stacking': partial(light_gbm_stacking,
                                          use_features=False),
             'log_reg_stacking': partial(sklearn_pipeline,
                                         ClassifierClass=LogisticRegression,
                                         clf_name='log_reg'),
             }
