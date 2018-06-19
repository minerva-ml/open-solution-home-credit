from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from steppy.adapter import Adapter, E
from steppy.base import Step

from pipeline_blocks import feature_extraction, classifier_light_gbm, preprocessing_fillna, classifier_sklearn, \
    classifier_xgb
from postprocessing import Clipper


def lightGBM(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      persist_output=True,
                                                      cache_output=True,
                                                      load_persisted_output=True)
        light_gbm = classifier_light_gbm((features, features_valid),
                                         config,
                                         train_mode)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      cache_output=True)
        light_gbm = classifier_light_gbm(features,
                                         config,
                                         train_mode)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[light_gbm],
                   adapter=Adapter({'prediction': E(light_gbm.name, 'prediction')}),
                   experiment_directory=config.pipeline.experiment_directory)

    return clipper


def xgboost(config, train_mode):
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      persist_output=True,
                                                      cache_output=True,
                                                      load_persisted_output=True)
        xgb = classifier_xgb((features, features_valid),
                             config,
                             train_mode)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      cache_output=True)
        xgb = classifier_xgb(features,
                             config,
                             train_mode)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[xgb],
                   adapter=Adapter({'prediction': E(xgb.name, 'prediction')}),
                   experiment_directory=config.pipeline.experiment_directory)

    return clipper


def sklearn_main(config, ClassifierClass, clf_name, train_mode, normalize=False):
    model_params = getattr(config, clf_name)
    random_search_config = getattr(config.random_search, clf_name)
    full_config = (config, model_params, random_search_config)
    if train_mode:
        features, features_valid = feature_extraction(config,
                                                      train_mode,
                                                      persist_output=True,
                                                      cache_output=True,
                                                      load_persisted_output=True)

        sklearn_preproc = preprocessing_fillna((features, features_valid), config, train_mode)
    else:
        features = feature_extraction(config,
                                      train_mode,
                                      cache_output=True)
        sklearn_preproc = preprocessing_fillna(features, config, train_mode)

    sklearn_clf = classifier_sklearn(sklearn_preproc,
                                     ClassifierClass,
                                     full_config,
                                     clf_name,
                                     train_mode,
                                     normalize)

    clipper = Step(name='clipper',
                   transformer=Clipper(**config.clipper),
                   input_steps=[sklearn_clf],
                   adapter=Adapter({'prediction': E(sklearn_clf.name, 'predicted')}),
                   experiment_directory=config.pipeline.experiment_directory)
    return clipper


PIPELINES = {'lightGBM': {'train': partial(lightGBM, train_mode=True),
                          'inference': partial(lightGBM, train_mode=False)
                          },
             'XGBoost': {'train': partial(xgboost, train_mode=True),
                         'inference': partial(xgboost, train_mode=False)
                         },
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
