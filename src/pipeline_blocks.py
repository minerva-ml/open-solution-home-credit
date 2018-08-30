import category_encoders as ce

import numpy as np
from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer, IdentityOperation

from . import feature_extraction as fe
from . import data_cleaning as dc
from .models import get_sklearn_classifier, XGBoost, LightGBM, CatBoost, NeuralNetwork
from .utils import Normalizer, MinMaxScaler


def classifier_light_gbm(features, config, train_mode, suffix, **kwargs):
    model_name = 'light_gbm{}'.format(suffix)

    if train_mode:
        features_train, features_valid = features

        light_gbm = Step(name=model_name,
                         transformer=LightGBM(name=model_name, **config.light_gbm),
                         input_data=['main_table'],
                         input_steps=[features_train, features_valid],
                         adapter=Adapter({'X': E(features_train.name, 'features'),
                                          'y': E('main_table', 'y'),
                                          'feature_names': E(features_train.name, 'feature_names'),
                                          'categorical_features': E(features_train.name, 'categorical_features'),
                                          'X_valid': E(features_valid.name, 'features'),
                                          'y_valid': E('main_table', 'y_valid'),
                                          }),
                         force_fitting=False,
                         experiment_directory=config.pipeline.experiment_directory, **kwargs)
    else:
        light_gbm = Step(name=model_name,
                         transformer=LightGBM(name=model_name, **config.light_gbm),
                         input_steps=[features],
                         adapter=Adapter({'X': E(features.name, 'features')}),
                         experiment_directory=config.pipeline.experiment_directory, **kwargs)
    return light_gbm


def classifier_catboost(features, config, train_mode, suffix, **kwargs):
    model_name = 'catboost{}'.format(suffix)

    if train_mode:
        features_train, features_valid = features

        catboost = Step(name=model_name,
                        transformer=CatBoost(**config.catboost),
                        input_data=['main_table'],
                        input_steps=[features_train, features_valid],
                        adapter=Adapter({'X': E(features_train.name, 'features'),
                                         'y': E('main_table', 'y'),
                                         'feature_names': E(features_train.name, 'feature_names'),
                                         'categorical_features': E(features_train.name, 'categorical_features'),
                                         'X_valid': E(features_valid.name, 'features'),
                                         'y_valid': E('main_table', 'y_valid'),
                                         }),
                        experiment_directory=config.pipeline.experiment_directory, **kwargs)
    else:
        catboost = Step(name=model_name,
                        transformer=CatBoost(**config.catboost),
                        input_steps=[features],
                        adapter=Adapter({'X': E(features.name, 'features')}),
                        experiment_directory=config.pipeline.experiment_directory, **kwargs)
    return catboost


def classifier_xgb(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features_train, features_valid = features

        xgboost = Step(name='xgboost{}'.format(suffix),
                       transformer=XGBoost(**config.xgboost),
                       input_data=['main_table'],
                       input_steps=[features_train, features_valid],
                       adapter=Adapter({'X': E(features_train.name, 'features'),
                                        'y': E('main_table', 'y'),
                                        'feature_names': E(features_train.name, 'feature_names'),
                                        'X_valid': E(features_valid.name, 'features'),
                                        'y_valid': E('main_table', 'y_valid'),
                                        }),
                       experiment_directory=config.pipeline.experiment_directory,
                       **kwargs)
    else:
        xgboost = Step(name='xgboost{}'.format(suffix),
                       transformer=XGBoost(**config.xgboost),
                       input_steps=[features],
                       adapter=Adapter({'X': E(features.name, 'features')}),
                       experiment_directory=config.pipeline.experiment_directory,
                       **kwargs)
    return xgboost


def classifier_nn(features, config, train_mode, suffix, **kwargs):
    model_name = 'nn{}'.format(suffix)

    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    if train_mode:
        features_train, features_valid = features

        nn = Step(name=model_name,
                  transformer=NeuralNetwork(**config.neural_network, suffix=suffix),
                  input_data=['main_table'],
                  input_steps=[features_train, features_valid],
                  adapter=Adapter({'X': E(features_train.name, 'features'),
                                   'y': E('main_table', 'y'),
                                   'validation_data': (E(features_valid.name, 'features'),
                                                       E('main_table', 'y_valid')),
                                   }),
                  force_fitting=True,
                  persist_output=persist_output,
                  cache_output=cache_output,
                  load_persisted_output=load_persisted_output,
                  experiment_directory=config.pipeline.experiment_directory, **kwargs)
    else:
        nn = Step(name=model_name,
                  transformer=NeuralNetwork(**config.neural_network, suffix=suffix),
                  input_steps=[features],
                  adapter=Adapter({'X': E(features.name, 'features')}),
                  persist_output=persist_output,
                  cache_output=cache_output,
                  load_persisted_output=load_persisted_output,
                  experiment_directory=config.pipeline.experiment_directory, **kwargs)
    return nn


def classifier_sklearn(features,
                       ClassifierClass,
                       config,
                       clf_name,
                       train_mode,
                       suffix,
                       **kwargs):
    model_name = '{}{}'.format(clf_name, suffix)
    model_params = getattr(config, clf_name)
    if train_mode:
        features_train, features_valid = features

        sklearn_clf = Step(name=model_name,
                           transformer=get_sklearn_classifier(ClassifierClass, **model_params),
                           input_data=['main_table'],
                           input_steps=[features_train, features_valid],
                           adapter=Adapter({'X': E(features_train.name, 'features'),
                                            'y': E('main_table', 'y'),
                                            'feature_names': E(features_train.name, 'feature_names'),
                                            'X_valid': E(features_valid.name, 'features'),
                                            'y_valid': E('main_table', 'y_valid'),
                                            }),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)
    else:
        sklearn_clf = Step(name=model_name,
                           transformer=get_sklearn_classifier(ClassifierClass, **model_params),
                           input_steps=[features],
                           adapter=Adapter({'X': E(features.name, 'features')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)
    return sklearn_clf


def feature_extraction(config, train_mode, suffix, **kwargs):
    application_cleaned = _application_cleaning(config, **kwargs)
    bureau_cleaned = _bureau_cleaning(config, **kwargs)
    bureau_balance_cleaned = _bureau_balance_cleaning(config, **kwargs)
    credit_card_balance_cleaned = _credit_card_balance_cleaning(config, **kwargs)
    installment_payments_cleaned = _installment_payments_cleaning(config, **kwargs)
    pos_cash_balance_cleaned = _pos_cash_balance_cleaning(config, **kwargs)
    previous_application_cleaned = _previous_application_cleaning(config, **kwargs)

    application = _application(application_cleaned, config, **kwargs)
    bureau = _bureau(bureau_cleaned, config, **kwargs)
    bureau_balance = _bureau_balance(bureau_balance_cleaned, config, **kwargs)
    credit_card_balance = _credit_card_balance(credit_card_balance_cleaned, config, **kwargs)
    installment_payments = _installment_payments(installment_payments_cleaned, config, **kwargs)
    pos_cash_balance = _pos_cash_balance(pos_cash_balance_cleaned, config, **kwargs)
    previous_application = _previous_application(previous_application_cleaned, config, **kwargs)
    application_previous_application = _application_previous_application(application_cleaned,
                                                                         previous_application_cleaned,
                                                                         config,
                                                                         **kwargs)

    application_agg = _application_groupby_agg(application_cleaned, config, **kwargs)
    bureau_agg = _bureau_groupby_agg(bureau_cleaned, config, **kwargs)
    credit_card_balance_agg = _credit_card_balance_groupby_agg(credit_card_balance_cleaned, config, **kwargs)
    installments_payments_agg = _installments_payments_groupby_agg(installment_payments_cleaned, config, **kwargs)
    pos_cash_balance_agg = _pos_cash_balance_groupby_agg(pos_cash_balance_cleaned, config, **kwargs)
    previous_applications_agg = _previous_applications_groupby_agg(previous_application_cleaned, config, **kwargs)

    application_categorical_encoder = _application_categorical_encoder(application, config, **kwargs)
    previous_application_categorical_encoder = _previous_application_categorical_encoder(previous_application, config,
                                                                                         **kwargs)
    application_previous_application_categorical_encoder = _application_previous_application_categorical_encoder(
        application_previous_application, config, **kwargs)

    numerical_features = []
    categorical_features = []

    if config.feature_selection.use_application:
        numerical_features.append(application)
    if config.feature_selection.use_bureau:
        numerical_features.append(bureau)
    if config.feature_selection.use_bureau_balance:
        numerical_features.append(bureau_balance)
    if config.feature_selection.use_credit_card_balance:
        numerical_features.append(credit_card_balance)
    if config.feature_selection.use_installments_payments:
        numerical_features.append(installment_payments)
    if config.feature_selection.use_pos_cash_balance:
        numerical_features.append(pos_cash_balance)
    if config.feature_selection.use_previous_applications:
        numerical_features.append(previous_application)
    if config.feature_selection.use_application_previous_application:
        numerical_features.append(application_previous_application)
    if config.feature_selection.use_application_aggregations:
        numerical_features.append(application_agg)
    if config.feature_selection.use_bureau_aggregations:
        numerical_features.append(bureau_agg)
    if config.feature_selection.use_credit_card_balance_aggregations:
        numerical_features.append(credit_card_balance_agg)
    if config.feature_selection.use_installments_payments_aggregations:
        numerical_features.append(installments_payments_agg)
    if config.feature_selection.use_pos_cash_balance_aggregations:
        numerical_features.append(pos_cash_balance_agg)
    if config.feature_selection.use_previous_applications_aggregations:
        numerical_features.append(previous_applications_agg)
    if config.feature_selection.use_application_categorical_features:
        categorical_features.append(application_categorical_encoder)
    if config.feature_selection.use_previous_application_categorical_features:
        categorical_features.append(previous_application_categorical_encoder)
    if config.feature_selection.use_application_previous_application_categorical_features:
        categorical_features.append(application_previous_application_categorical_encoder)

    feature_combiner = _join_features(numerical_features=numerical_features,
                                      categorical_features=categorical_features,
                                      config=config, train_mode=train_mode, **kwargs)

    if train_mode:
        idx_merge, idx_merge_valid = _split_samples(feature_combiner, config, train_mode, suffix, **kwargs)
        return idx_merge, idx_merge_valid
    else:
        idx_merge = _split_samples(feature_combiner, config, train_mode, suffix, **kwargs)
        return idx_merge


def xgb_preprocessing(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features, features_valid = features

    one_hot_encoder = Step(name='one_hot_encoder{}'.format(suffix),
                           transformer=fe.CategoricalEncodingWrapper(
                               encoder=ce.OneHotEncoder,
                               **config.xgb_preprocessing.one_hot_encoder),
                           input_steps=[features],
                           adapter=Adapter({'X': E(features.name, 'features'),
                                            'cols': E(features.name, 'categorical_features')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           )

    if train_mode:
        one_hot_encoder_valid = Step(name='one_hot_encoder_valid{}'.format(suffix),
                                     transformer=one_hot_encoder,
                                     input_steps=[features_valid],
                                     adapter=Adapter({'X': E(features_valid.name, 'features'),
                                                      'cols': E(features_valid.name, 'categorical_features')}),
                                     experiment_directory=config.pipeline.experiment_directory,
                                     )

        return one_hot_encoder, one_hot_encoder_valid
    else:
        return one_hot_encoder


def neural_network_preprocessing(features, config, train_mode, suffix, normalize, **kwargs):
    if train_mode:
        features, features_valid = features

    one_hot_encoder = Step(name='one_hot_encoder{}'.format(suffix),
                           transformer=fe.CategoricalEncodingWrapper(
                               encoder=ce.OneHotEncoder,
                               **config.neural_network_preprocessing.one_hot_encoder),
                           input_steps=[features],
                           adapter=Adapter({'X': E(features.name, 'features'),
                                            'cols': E(features.name, 'categorical_features')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           )

    fillnaer = Step(name='fillna{}'.format(suffix),
                    transformer=_fillna(**config.neural_network_preprocessing.fillna),
                    input_steps=[one_hot_encoder],
                    adapter=Adapter({'X': E(one_hot_encoder.name, 'features')}),
                    experiment_directory=config.pipeline.experiment_directory,
                    )

    if normalize:
        normalizer = Step(name='normalizer{}'.format(suffix),
                          transformer=MinMaxScaler(),
                          input_steps=[fillnaer],
                          adapter=Adapter({'X': E(fillnaer.name, 'X')}),
                          experiment_directory=config.pipeline.experiment_directory,
                          )
        last_step = normalizer
    else:
        last_step = fillnaer

    neural_network_preprocess = Step(name='neural_network_preprocess{}'.format(suffix),
                                     transformer=IdentityOperation(),
                                     input_steps=[last_step, one_hot_encoder],
                                     adapter=Adapter({'features': E(last_step.name, 'X'),
                                                      'feature_names': E(one_hot_encoder.name, 'feature_names'),
                                                     'categorical_features': E(one_hot_encoder.name, 'categorical_features')
                                                      }),
                                     experiment_directory=config.pipeline.experiment_directory,
                                     **kwargs
                                     )

    if train_mode:
        one_hot_encoder_valid = Step(name='one_hot_encoder_valid{}'.format(suffix),
                                     transformer=one_hot_encoder,
                                     input_steps=[features_valid],
                                     adapter=Adapter({'X': E(features_valid.name, 'features'),
                                                      'cols': E(features_valid.name, 'categorical_features')}),
                                     experiment_directory=config.pipeline.experiment_directory,
                                     )

        fillnaer_valid = Step(name='fillna_valid{}'.format(suffix),
                              transformer=fillnaer,
                              input_steps=[one_hot_encoder_valid],
                              adapter=Adapter({'X': E(one_hot_encoder_valid.name, 'features')}),
                              experiment_directory=config.pipeline.experiment_directory,
                              )

        if normalize:
            normalizer_valid = Step(name='normalizer_valid{}'.format(suffix),
                                    transformer=normalizer,
                                    input_steps=[fillnaer_valid],
                                    adapter=Adapter({'X': E(fillnaer_valid.name, 'X')}),
                                    experiment_directory=config.pipeline.experiment_directory,
                                    )
            last_step = normalizer_valid
        else:
            last_step = fillnaer_valid

        neural_network_preprocess_valid = Step(name='neural_network_preprocess_valid{}'.format(suffix),
                                               transformer=IdentityOperation(),
                                               input_steps=[last_step, one_hot_encoder_valid],
                                               adapter=Adapter({'features': E(last_step.name, 'X'),
                                                                'feature_names': E(one_hot_encoder_valid.name,
                                                                                   'feature_names'),
                                                                'categorical_features': E(one_hot_encoder_valid.name,
                                                                                          'categorical_features')
                                                                }),
                                               experiment_directory=config.pipeline.experiment_directory,
                                               **kwargs
                                               )
        return neural_network_preprocess, neural_network_preprocess_valid
    else:
        return neural_network_preprocess


def sklearn_preprocessing(features, config, train_mode, suffix, normalize, **kwargs):
    if train_mode:
        features, features_valid = features

    one_hot_encoder = Step(name='one_hot_encoder{}'.format(suffix),
                           transformer=fe.CategoricalEncodingWrapper(
                               encoder=ce.OneHotEncoder,
                               **config.sklearn_preprocessing.one_hot_encoder),
                           input_steps=[features],
                           adapter=Adapter({'X': E(features.name, 'features'),
                                            'cols': E(features.name, 'categorical_features')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           )

    fillnaer = Step(name='fillna{}'.format(suffix),
                    transformer=_fillna(**config.sklearn_preprocessing.fillna),
                    input_steps=[one_hot_encoder],
                    adapter=Adapter({'X': E(one_hot_encoder.name, 'features')}),
                    experiment_directory=config.pipeline.experiment_directory,
                    )

    if normalize:
        normalizer = Step(name='normalizer{}'.format(suffix),
                          transformer=Normalizer(),
                          input_steps=[fillnaer],
                          adapter=Adapter({'X': E(fillnaer.name, 'X')}),
                          experiment_directory=config.pipeline.experiment_directory,
                          )
        last_step = normalizer
    else:
        last_step = fillnaer

    sklearn_preprocess = Step(name='sklearn_preprocess{}'.format(suffix),
                              transformer=IdentityOperation(),
                              input_steps=[last_step, one_hot_encoder],
                              adapter=Adapter({'features': E(last_step.name, 'X'),
                                               'feature_names': E(one_hot_encoder.name, 'feature_names'),
                                               'categorical_features': E(one_hot_encoder.name, 'categorical_features')
                                               }),
                              experiment_directory=config.pipeline.experiment_directory,
                              **kwargs
                              )

    if train_mode:
        one_hot_encoder_valid = Step(name='one_hot_encoder_valid{}'.format(suffix),
                                     transformer=one_hot_encoder,
                                     input_steps=[features_valid],
                                     adapter=Adapter({'X': E(features_valid.name, 'features'),
                                                      'cols': E(features_valid.name, 'categorical_features')}),
                                     experiment_directory=config.pipeline.experiment_directory,
                                     )

        fillnaer_valid = Step(name='fillna_valid{}'.format(suffix),
                              transformer=fillnaer,
                              input_steps=[one_hot_encoder_valid],
                              adapter=Adapter({'X': E(one_hot_encoder_valid.name, 'features')}),
                              experiment_directory=config.pipeline.experiment_directory,
                              )

        if normalize:
            normalizer_valid = Step(name='normalizer_valid{}'.format(suffix),
                                    transformer=normalizer,
                                    input_steps=[fillnaer_valid],
                                    adapter=Adapter({'X': E(fillnaer_valid.name, 'X')}),
                                    experiment_directory=config.pipeline.experiment_directory,
                                    )
            last_step = normalizer_valid
        else:
            last_step = fillnaer_valid

        sklearn_preprocess_valid = Step(name='sklearn_preprocess_valid{}'.format(suffix),
                                        transformer=IdentityOperation(),
                                        input_steps=[last_step, one_hot_encoder_valid],
                                        adapter=Adapter({'features': E(last_step.name, 'X'),
                                                         'feature_names': E(one_hot_encoder_valid.name,
                                                                            'feature_names'),
                                                         'categorical_features': E(one_hot_encoder_valid.name,
                                                                                   'categorical_features')
                                                         }),
                                        experiment_directory=config.pipeline.experiment_directory,
                                        **kwargs
                                        )
        return sklearn_preprocess, sklearn_preprocess_valid
    else:
        return sklearn_preprocess


def catboost_preprocessing(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features, features_valid = features

    fillnaer = Step(name='fillna{}'.format(suffix),
                    transformer=_fillna(**config.sklearn_preprocessing.fillna),
                    input_steps=[features],
                    adapter=Adapter({'X': E(features.name, 'features')}),
                    experiment_directory=config.pipeline.experiment_directory,
                    )

    preprocessed = Step(name='preprocess{}'.format(suffix),
                        transformer=IdentityOperation(),
                        input_steps=[fillnaer, features],
                        adapter=Adapter({'features': E(fillnaer.name, 'transformed'),
                                         'feature_names': E(features.name, 'feature_names'),
                                         'categorical_features': E(features.name, 'categorical_features')
                                         }),
                        experiment_directory=config.pipeline.experiment_directory,
                        **kwargs
                        )

    if train_mode:
        fillnaer_valid = Step(name='fillna_valid{}'.format(suffix),
                              transformer=fillnaer,
                              input_steps=[features_valid],
                              adapter=Adapter({'X': E(features_valid.name, 'features')}),
                              experiment_directory=config.pipeline.experiment_directory,
                              )
        preprocessed_valid = Step(name='preprocess_valid{}'.format(suffix),
                                  transformer=IdentityOperation(),
                                  input_steps=[fillnaer_valid, features_valid],
                                  adapter=Adapter({'features': E(fillnaer_valid.name, 'transformed'),
                                                   'feature_names': E(features_valid.name,
                                                                      'feature_names'),
                                                   'categorical_features': E(features_valid.name,
                                                                             'categorical_features')
                                                   }),
                                  experiment_directory=config.pipeline.experiment_directory,
                                  **kwargs
                                  )
        return preprocessed, preprocessed_valid
    else:
        return preprocessed


def oof_predictions(config, train_mode, suffix, **kwargs):
    features = Step(name='oof_predictions{}'.format(suffix),
                    transformer=IdentityOperation(),
                    input_data=['oof_predictions'],
                    adapter=Adapter({'numerical_features': E('oof_predictions', 'X')
                                     }),
                    experiment_directory=config.pipeline.experiment_directory, **kwargs)

    feature_combiner = _join_features(numerical_features=[features],
                                      categorical_features=[],
                                      config=config, train_mode=train_mode, suffix=suffix, **kwargs)
    if train_mode:
        features_valid = Step(name='oof_predictions{}'.format(suffix),
                              transformer=IdentityOperation(),
                              input_data=['oof_predictions'],
                              adapter=Adapter({'numerical_features': E('oof_predictions', 'X_valid')
                                               }),
                              experiment_directory=config.pipeline.experiment_directory, **kwargs)

        feature_combiner_valid = _join_features(numerical_features=[features_valid],
                                                categorical_features=[],
                                                config=config, train_mode=train_mode, suffix='_valid{}'.format(suffix),
                                                **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        return feature_combiner


def stacking_normalization(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features, features_valid = features

    normalizer = Step(name='stacking_normalizer{}'.format(suffix),
                      transformer=Normalizer(),
                      input_steps=[features],
                      adapter=Adapter({'X': E(features.name, 'features')}),
                      experiment_directory=config.pipeline.experiment_directory,
                      )

    stacking_normalized = Step(name='stacking_normalization{}'.format(suffix),
                               transformer=IdentityOperation(),
                               input_steps=[normalizer, features],
                               adapter=Adapter({'features': E(normalizer.name, 'X'),
                                                'feature_names': E(features.name, 'feature_names'),
                                                'categorical_features': E(features.name, 'categorical_features')
                                                }),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs
                               )

    if train_mode:
        normalizer_valid = Step(name='stacking_normalizer_valid{}'.format(suffix),
                                transformer=normalizer,
                                input_steps=[features_valid],
                                adapter=Adapter({'X': E(features_valid.name, 'features')}),
                                experiment_directory=config.pipeline.experiment_directory,
                                )

        stacking_normalized_valid = Step(name='stacking_normalization_valid{}'.format(suffix),
                                         transformer=IdentityOperation(),
                                         input_steps=[normalizer_valid, features_valid],
                                         adapter=Adapter({'features': E(normalizer_valid.name, 'X'),
                                                          'feature_names': E(features_valid.name, 'feature_names'),
                                                          'categorical_features': E(features_valid.name,
                                                                                    'categorical_features')
                                                          }),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs
                                         )
        return stacking_normalized, stacking_normalized_valid
    else:
        return stacking_normalized


def concat_features(features_list, config, train_mode, suffix, **kwargs):
    if train_mode:
        features = [feature[0] for feature in features_list]
        features_valid = [feature[1] for feature in features_list]
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        features = features_list
        persist_output = False
        cache_output = True
        load_persisted_output = False

    feature_concat = Step(name='feature_concat{}'.format(suffix),
                          transformer=fe.FeatureConcat(),
                          input_steps=features,
                          adapter=Adapter({
                              'features_list': [
                                  E(feature.name, 'features') for feature in features],
                              'feature_names_list': [
                                  E(feature.name, 'feature_names') for feature in features],
                              'categorical_features_list': [
                                  E(feature.name, 'categorical_features') for feature in features],
                          }),
                          experiment_directory=config.pipeline.experiment_directory,
                          persist_output=persist_output,
                          cache_output=cache_output,
                          load_persisted_output=load_persisted_output)
    if train_mode:
        feature_concat_valid = Step(name='feature_concat_valid{}'.format(suffix),
                                    transformer=feature_concat,
                                    input_steps=features_valid,
                                    adapter=Adapter({
                                        'features_list': [
                                            E(feature.name, 'features') for feature in features_valid],
                                        'feature_names_list': [
                                            E(feature.name, 'feature_names') for feature in features_valid],
                                        'categorical_features_list': [
                                            E(feature.name, 'categorical_features') for feature in features_valid],
                                    }),
                                    experiment_directory=config.pipeline.experiment_directory,
                                    persist_output=persist_output,
                                    cache_output=cache_output,
                                    load_persisted_output=load_persisted_output)

        return feature_concat, feature_concat_valid
    else:
        return feature_concat


def _join_features(numerical_features,
                   categorical_features,
                   config, train_mode, suffix='', **kwargs):
    if train_mode:
        persist_output = True
        cache_output = True
        load_persisted_output = True
    else:
        persist_output = False
        cache_output = True
        load_persisted_output = False

    feature_joiner = Step(name='feature_joiner{}'.format(suffix),
                          transformer=fe.FeatureJoiner(**config.feature_joiner),
                          input_steps=numerical_features + categorical_features,
                          adapter=Adapter({
                              'numerical_feature_list': [
                                  E(feature.name, 'numerical_features') for feature in numerical_features],
                              'categorical_feature_list': [
                                  E(feature.name, 'categorical_features') for feature in categorical_features],
                          }),
                          experiment_directory=config.pipeline.experiment_directory,
                          persist_output=persist_output,
                          cache_output=cache_output,
                          load_persisted_output=load_persisted_output)

    return feature_joiner


def _split_samples(features, config, train_mode, suffix, **kwargs):
    idx_merge = Step(name='idx_merge{}'.format(suffix),
                     transformer=fe.IDXMerge(**config.idx_merge),
                     input_data=['main_table'],
                     input_steps=[features],
                     adapter=Adapter({'table': E('main_table', 'X'),
                                      'features': E(features.name, 'features'),
                                      'categorical_features': E(features.name, 'categorical_features')}),
                     experiment_directory=config.pipeline.experiment_directory,
                     **kwargs)

    if train_mode:
        idx_merge_valid = Step(name='idx_merge_valid{}'.format(suffix),
                               transformer=fe.IDXMerge(**config.idx_merge),
                               input_data=['main_table'],
                               input_steps=[features],
                               adapter=Adapter({'table': E('main_table', 'X_valid'),
                                                'features': E(features.name, 'features'),
                                                'categorical_features': E(features.name, 'categorical_features')}),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)
        return idx_merge, idx_merge_valid
    else:
        return idx_merge


def _application_categorical_encoder(application, config, **kwargs):
    categorical_encoder = Step(name='application_categorical_encoder',
                               transformer=fe.CategoricalEncoder(),
                               input_steps=[application],
                               adapter=Adapter({'X': E(application.name, 'categorical_features'),
                                                'categorical_columns': E(application.name, 'categorical_columns')}),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)

    return categorical_encoder


def _previous_application_categorical_encoder(previous_application, config, **kwargs):
    categorical_encoder = Step(name='previous_application_categorical_encoder',
                               transformer=fe.CategoricalEncoder(),
                               input_steps=[previous_application],
                               adapter=Adapter({'X': E(previous_application.name, 'categorical_features'),
                                                'categorical_columns': E(previous_application.name,
                                                                         'categorical_columns')}),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)

    return categorical_encoder


def _application_previous_application_categorical_encoder(application_previous_application, config, **kwargs):
    categorical_encoder = Step(name='application_previous_application_categorical_encoder',
                               transformer=fe.CategoricalEncoder(),
                               input_steps=[application_previous_application],
                               adapter=Adapter({'X': E(application_previous_application.name, 'categorical_features'),
                                                'categorical_columns': E(application_previous_application.name,
                                                                         'categorical_columns')}),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)

    return categorical_encoder


def _application_groupby_agg(application_cleaned, config, **kwargs):
    application_groupby_agg = Step(name='application_groupby_agg',
                                   transformer=fe.GroupbyAggregateDiffs(**config.applications),
                                   input_steps=[application_cleaned],
                                   adapter=Adapter({'main_table': E(application_cleaned.name, 'application')}),
                                   experiment_directory=config.pipeline.experiment_directory,
                                   **kwargs)

    return application_groupby_agg


def _bureau_groupby_agg(bureau_cleaned, config, **kwargs):
    bureau_groupby_agg = Step(name='bureau_groupby_agg',
                              transformer=fe.GroupbyAggregate(**config.bureau),
                              input_steps=[bureau_cleaned],
                              adapter=Adapter({'table': E(bureau_cleaned.name, 'bureau')}),
                              experiment_directory=config.pipeline.experiment_directory,
                              **kwargs)

    return bureau_groupby_agg


def _credit_card_balance_groupby_agg(credit_card_balance_cleaned, config, **kwargs):
    credit_card_balance_groupby_agg = Step(name='credit_card_balance_groupby_agg',
                                           transformer=fe.GroupbyAggregate(**config.credit_card_balance),
                                           input_steps=[credit_card_balance_cleaned],
                                           adapter=Adapter({'table': E(credit_card_balance_cleaned.name,
                                                                       'credit_card')}),
                                           experiment_directory=config.pipeline.experiment_directory,
                                           **kwargs)

    return credit_card_balance_groupby_agg


def _installments_payments_groupby_agg(installments_payments_cleaned, config, **kwargs):
    installments_payments_groupby_agg = Step(name='installments_payments_groupby_agg',
                                             transformer=fe.GroupbyAggregate(**config.installments_payments),
                                             input_steps=[installments_payments_cleaned],
                                             adapter=Adapter({'table': E(installments_payments_cleaned.name,
                                                                         'installments')}),
                                             experiment_directory=config.pipeline.experiment_directory,
                                             **kwargs)

    return installments_payments_groupby_agg


def _pos_cash_balance_groupby_agg(pos_cash_balance_cleaned, config, **kwargs):
    pos_cash_balance_groupby_agg = Step(name='pos_cash_balance_groupby_agg',
                                        transformer=fe.GroupbyAggregate(**config.pos_cash_balance),
                                        input_steps=[pos_cash_balance_cleaned],
                                        adapter=Adapter({'table': E(pos_cash_balance_cleaned.name, 'pos_cash')}),
                                        experiment_directory=config.pipeline.experiment_directory,
                                        **kwargs)

    return pos_cash_balance_groupby_agg


def _previous_applications_groupby_agg(previous_application_cleaned, config, **kwargs):
    previous_applications_groupby_agg = Step(name='previous_applications_groupby_agg',
                                             transformer=fe.GroupbyAggregate(**config.previous_applications),
                                             input_steps=[previous_application_cleaned],
                                             adapter=Adapter({'table': E(previous_application_cleaned.name,
                                                                         'previous_application')}),
                                             experiment_directory=config.pipeline.experiment_directory, **kwargs)

    return previous_applications_groupby_agg


def _application_cleaning(config, **kwargs):
    application_cleaning = Step(name='application_cleaning',
                                transformer=dc.ApplicationCleaning(**config.preprocessing.impute_missing),
                                input_data=['application'],
                                adapter=Adapter({'application': E('application', 'X')}),
                                experiment_directory=config.pipeline.experiment_directory,
                                **kwargs)

    return application_cleaning


def _application(application_cleaned, config, **kwargs):
    application_hand_crafted = Step(name='application_hand_crafted',
                                    transformer=fe.ApplicationFeatures(**config.applications),
                                    input_steps=[application_cleaned],
                                    adapter=Adapter({'application': E(application_cleaned.name, 'application')}),
                                    experiment_directory=config.pipeline.experiment_directory,
                                    **kwargs)

    return application_hand_crafted


def _bureau_cleaning(config, **kwargs):
    bureau_cleaning = Step(name='bureau_cleaning',
                           transformer=dc.BureauCleaning(**config.preprocessing.impute_missing),
                           input_data=['bureau'],
                           adapter=Adapter({'bureau': E('bureau', 'X')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)

    return bureau_cleaning


def _bureau(bureau_cleaned, config, **kwargs):
    bureau_hand_crafted = Step(name='bureau_hand_crafted',
                               transformer=fe.BureauFeatures(**config.bureau),
                               input_steps=[bureau_cleaned],
                               adapter=Adapter({'bureau': E(bureau_cleaned.name, 'bureau')}),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)

    return bureau_hand_crafted


def _bureau_balance_cleaning(config, **kwargs):
    bureau_cleaning = Step(name='bureau_balance_cleaning',
                           transformer=dc.BureauBalanceCleaning(**config.preprocessing.impute_missing),
                           input_data=['bureau_balance'],
                           adapter=Adapter({'bureau_balance': E('bureau_balance', 'X')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)

    return bureau_cleaning


def _bureau_balance(bureau_balance_cleaned, config, **kwargs):
    bureau_balance_hand_crafted = Step(name='bureau_balance_hand_crafted',
                                       transformer=fe.BureauBalanceFeatures(**config.bureau_balance),
                                       input_steps=[bureau_balance_cleaned],
                                       adapter=Adapter({'bureau_balance': E(bureau_balance_cleaned.name,
                                                                            'bureau_balance')}),
                                       experiment_directory=config.pipeline.experiment_directory, **kwargs)

    return bureau_balance_hand_crafted


def _credit_card_balance_cleaning(config, **kwargs):
    credit_card_balance_cleaning = Step(name='credit_card_balance_cleaning',
                                        transformer=dc.CreditCardCleaning(
                                            **config.preprocessing.impute_missing),
                                        input_data=['credit_card_balance'],
                                        adapter=Adapter({'credit_card': E('credit_card_balance', 'X')}),
                                        experiment_directory=config.pipeline.experiment_directory,
                                        **kwargs)

    return credit_card_balance_cleaning


def _credit_card_balance(credit_card_balance_cleaned, config, **kwargs):
    credit_card_balance_hand_crafted = Step(name='credit_card_balance_hand_crafted',
                                            transformer=fe.CreditCardBalanceFeatures(**config.credit_card_balance),
                                            input_steps=[credit_card_balance_cleaned],
                                            adapter=Adapter({'credit_card': E(credit_card_balance_cleaned.name,
                                                                              'credit_card')}),
                                            experiment_directory=config.pipeline.experiment_directory,
                                            **kwargs)

    return credit_card_balance_hand_crafted


def _pos_cash_balance_cleaning(config, **kwargs):
    credit_card_balance_cleaning = Step(name='pos_cash_balance_cleaning',
                                        transformer=dc.PosCashCleaning(
                                            **config.preprocessing.impute_missing),
                                        input_data=['pos_cash_balance'],
                                        adapter=Adapter({'pos_cash': E('pos_cash_balance', 'X')}),
                                        experiment_directory=config.pipeline.experiment_directory,
                                        **kwargs)

    return credit_card_balance_cleaning


def _pos_cash_balance(pos_cash_balance_cleaned, config, **kwargs):
    pos_cash_balance_hand_crafted = Step(name='pos_cash_balance_hand_crafted',
                                         transformer=fe.POSCASHBalanceFeatures(**config.pos_cash_balance),
                                         input_steps=[pos_cash_balance_cleaned],
                                         adapter=Adapter({'pos_cash': E(pos_cash_balance_cleaned.name, 'pos_cash')}),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)

    return pos_cash_balance_hand_crafted


def _previous_application_cleaning(config, **kwargs):
    previous_application_cleaning = Step(name='previous_application_cleaning',
                                         transformer=dc.PreviousApplicationCleaning(
                                             **config.preprocessing.impute_missing),
                                         input_data=['previous_application'],
                                         adapter=Adapter({'previous_application': E('previous_application', 'X')}),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)

    return previous_application_cleaning


def _previous_application(previous_application_cleaned, config, **kwargs):
    previous_applications_hand_crafted = Step(name='previous_applications_hand_crafted',
                                              transformer=fe.PreviousApplicationFeatures(
                                                  **config.previous_applications),
                                              input_steps=[previous_application_cleaned],
                                              adapter=Adapter(
                                                  {'previous_application': E(previous_application_cleaned.name,
                                                                             'previous_application')}),
                                              experiment_directory=config.pipeline.experiment_directory,
                                              **kwargs)

    return previous_applications_hand_crafted


def _installment_payments_cleaning(config, **kwargs):
    installment_payments_cleaning = Step(name='installment_payments_cleaning',
                                         transformer=dc.InstallmentPaymentsCleaning(
                                             **config.preprocessing.impute_missing),
                                         input_data=['installments_payments'],
                                         adapter=Adapter({'installments': E('installments_payments', 'X')}),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)

    return installment_payments_cleaning


def _installment_payments(installment_payments_cleaned, config, **kwargs):
    installment_payments_hand_crafted = Step(name='installment_payments_hand_crafted',
                                             transformer=fe.InstallmentPaymentsFeatures(**config.installments_payments),
                                             input_steps=[installment_payments_cleaned],
                                             adapter=Adapter({'installments': E(installment_payments_cleaned.name,
                                                                                'installments')}),
                                             experiment_directory=config.pipeline.experiment_directory,
                                             **kwargs)

    return installment_payments_hand_crafted


def _application_previous_application(application_cleaned, previous_application_cleaned, config, **kwargs):
    app_prev_app_hand_crafted = Step(name='application_previous_application',
                                     transformer=fe.ApplicationPreviousApplicationFeatures(
                                         **config.application_previous_application),
                                     input_steps=[application_cleaned, previous_application_cleaned],
                                     adapter=Adapter({'application': E(application_cleaned.name, 'application'),
                                                      'previous_application': E(previous_application_cleaned.name,
                                                                                'previous_application')}),
                                     experiment_directory=config.pipeline.experiment_directory,
                                     **kwargs)

    return app_prev_app_hand_crafted


def _fillna(fill_value, **kwargs):
    def _inner_fillna(X):
        X_filled = X.replace([np.inf, -np.inf], fill_value)
        X_filled = X_filled.fillna(fill_value)
        return {'X': X_filled}

    return make_transformer(_inner_fillna)
