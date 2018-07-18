from functools import partial

from sklearn.metrics import roc_auc_score
from steppy.adapter import Adapter, E
from steppy.base import Step, make_transformer, IdentityOperation

from . import feature_extraction as fe
from . import data_cleaning as dc
from .hyperparameter_tuning import RandomSearchOptimizer, NeptuneMonitor, PersistResults
from .models import get_sklearn_classifier, XGBoost, LightGBM


def classifier_light_gbm(features, config, train_mode, suffix, **kwargs):
    model_name = 'light_gbm{}'.format(suffix)

    if train_mode:
        features_train, features_valid = features
        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(TransformerClass=LightGBM,
                                                params=config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[
                                                    NeptuneMonitor(
                                                        **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    PersistResults(
                                                        **config.random_search.light_gbm.callbacks.persist_results)]
                                                )
        else:
            transformer = LightGBM(name=model_name, **config.light_gbm)

        light_gbm = Step(name=model_name,
                         transformer=transformer,
                         input_data=['application'],
                         input_steps=[features_train, features_valid],
                         adapter=Adapter({'X': E(features_train.name, 'features'),
                                          'y': E('application', 'y'),
                                          'feature_names': E(features_train.name, 'feature_names'),
                                          'categorical_features': E(features_train.name, 'categorical_features'),
                                          'X_valid': E(features_valid.name, 'features'),
                                          'y_valid': E('application', 'y_valid'),
                                          }),
                         force_fitting=True,
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)
    else:
        light_gbm = Step(name=model_name,
                         transformer=LightGBM(name=model_name, **config.light_gbm),
                         input_steps=[features],
                         adapter=Adapter({'X': E(features.name, 'features')}),
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)
    return light_gbm


def classifier_light_gbm_stacking(features, config, train_mode, suffix, **kwargs):
    model_name = 'light_gbm{}'.format(suffix)

    if train_mode:
        features_train, features_valid = features
        if config.random_search.light_gbm.n_runs:
            transformer = RandomSearchOptimizer(TransformerClass=LightGBM,
                                                params=config.light_gbm,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=config.random_search.light_gbm.n_runs,
                                                callbacks=[
                                                    NeptuneMonitor(
                                                        **config.random_search.light_gbm.callbacks.neptune_monitor),
                                                    PersistResults(
                                                        **config.random_search.light_gbm.callbacks.persist_results)]
                                                )
        else:
            transformer = LightGBM(name=model_name, **config.light_gbm)

        light_gbm = Step(name=model_name,
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
                         force_fitting=True,
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)
    else:
        light_gbm = Step(name=model_name,
                         transformer=LightGBM(name=model_name, **config.light_gbm),
                         input_steps=[features],
                         adapter=Adapter({'X': E(features.name, 'features')}),
                         experiment_directory=config.pipeline.experiment_directory,
                         **kwargs)
    return light_gbm


def classifier_xgb(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features_train, features_valid = features
        if config.random_search.xgboost.n_runs:
            transformer = RandomSearchOptimizer(TransformerClass=XGBoost,
                                                params=config.xgboost,
                                                train_input_keys=[],
                                                valid_input_keys=['X_valid', 'y_valid'],
                                                score_func=roc_auc_score,
                                                maximize=True,
                                                n_runs=config.random_search.xgboost.n_runs,
                                                callbacks=[
                                                    NeptuneMonitor(
                                                        **config.random_search.xgboost.callbacks.neptune_monitor),
                                                    PersistResults(
                                                        **config.random_search.xgboost.callbacks.persist_results)]
                                                )
        else:
            transformer = XGBoost(**config.xgboost)

        xgboost = Step(name='xgboost{}'.format(suffix),
                       transformer=transformer,
                       input_data=['application'],
                       input_steps=[features_train, features_valid],
                       adapter=Adapter({'X': E(features_train.name, 'features'),
                                        'y': E('application', 'y'),
                                        'feature_names': E(features_train.name, 'feature_names'),
                                        'X_valid': E(features_valid.name, 'features'),
                                        'y_valid': E('application', 'y_valid'),
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


def classifier_sklearn(sklearn_features,
                       ClassifierClass,
                       full_config,
                       clf_name,
                       train_mode,
                       suffix,
                       normalize,
                       **kwargs):
    config, model_params, rs_config = full_config
    if train_mode:
        if config.random_search.random_forest.n_runs:
            transformer = RandomSearchOptimizer(
                partial(get_sklearn_classifier,
                        ClassifierClass=ClassifierClass,
                        normalize=normalize),
                model_params,
                train_input_keys=[],
                valid_input_keys=['X_valid', 'y_valid'],
                score_func=roc_auc_score,
                maximize=True,
                n_runs=rs_config.n_runs,
                callbacks=[NeptuneMonitor(**rs_config.callbacks.neptune_monitor),
                           PersistResults(**rs_config.callbacks.persist_results)]
            )
        else:
            transformer = get_sklearn_classifier(ClassifierClass, normalize, **model_params)

        sklearn_clf = Step(name='{}{}'.format(clf_name, suffix),
                           transformer=transformer,
                           input_data=['application'],
                           input_steps=[sklearn_features],
                           adapter=Adapter({'X': E(sklearn_features.name, 'X'),
                                            'y': E('application', 'y'),
                                            'X_valid': E(sklearn_features.name, 'X_valid'),
                                            'y_valid': E('application', 'y_valid'),
                                            }),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)
    else:
        sklearn_clf = Step(name='{}{}'.format(clf_name, suffix),
                           transformer=get_sklearn_classifier(ClassifierClass, normalize, **model_params),
                           input_steps=[sklearn_features],
                           adapter=Adapter({'X': E(sklearn_features.name, 'X')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)
    return sklearn_clf


def feature_extraction(config, train_mode, suffix, **kwargs):
    if train_mode:
        application, application_valid = _application(config, train_mode, suffix, **kwargs)
        bureau_cleaned = _bureau_cleaning(config, suffix, **kwargs)
        bureau, bureau_valid = _bureau(
            bureau_cleaned,
            config,
            train_mode,
            suffix,
            **kwargs)
        credit_card_balance_cleaned = _credit_card_balance_cleaning(config, suffix, **kwargs)
        credit_card_balance, credit_card_balance_valid = _credit_card_balance(
            credit_card_balance_cleaned,
            config,
            train_mode,
            suffix,
            **kwargs)
        pos_cash_balance, pos_cash_balance_valid = _pos_cash_balance(config, train_mode, suffix, **kwargs)
        previous_application_cleaned = _previous_application_cleaning(config, suffix, **kwargs)
        previous_application, previous_application_valid = _previous_application(
            previous_application_cleaned,
            config,
            train_mode,
            suffix,
            **kwargs)
        installment_payments, installment_payments_valid = _installment_payments(config, train_mode, suffix, **kwargs)

        application_agg, application_agg_valid = _application_groupby_agg(config, train_mode, suffix, **kwargs)
        bureau_agg, bureau_agg_valid = _bureau_groupby_agg(
            bureau_cleaned,
            config,
            train_mode,
            suffix,
            **kwargs)
        credit_card_balance_agg, credit_card_balance_agg_valid = _credit_card_balance_groupby_agg(
            credit_card_balance_cleaned,
            config,
            train_mode, suffix,
            **kwargs)
        installments_payments_agg, installments_payments_agg_valid = _installments_payments_groupby_agg(
            config,
            train_mode, suffix,
            **kwargs)
        pos_cash_balance_agg, pos_cash_balance_agg_valid = _pos_cash_balance_groupby_agg(
            config,
            train_mode, suffix,
            **kwargs)
        previous_applications_agg, previous_applications_agg_valid = _previous_applications_groupby_agg(
            previous_application_cleaned,
            config,
            train_mode, suffix,
            **kwargs)

        categorical_encoder, categorical_encoder_valid = _categorical_encoders(config, train_mode, suffix, **kwargs)

        feature_combiner, feature_combiner_valid = _join_features(
            numerical_features=[application,
                                application_agg,
                                bureau,
                                bureau_agg,
                                credit_card_balance,
                                credit_card_balance_agg,
                                installment_payments,
                                installments_payments_agg,
                                pos_cash_balance,
                                pos_cash_balance_agg,
                                previous_application,
                                previous_applications_agg,
                                ],
            numerical_features_valid=[application_valid,
                                      application_agg_valid,
                                      bureau_valid,
                                      bureau_agg_valid,
                                      credit_card_balance_valid,
                                      credit_card_balance_agg_valid,
                                      installment_payments_valid,
                                      installments_payments_agg_valid,
                                      pos_cash_balance_valid,
                                      pos_cash_balance_agg_valid,
                                      previous_application_valid,
                                      previous_applications_agg_valid,
                                      ],
            categorical_features=[categorical_encoder
                                  ],
            categorical_features_valid=[categorical_encoder_valid
                                        ],
            config=config,
            train_mode=train_mode,
            suffix=suffix,
            **kwargs)

        return feature_combiner, feature_combiner_valid
    else:
        application = _application(config, train_mode, suffix, **kwargs)
        bureau_cleaned = _bureau_cleaning(config, suffix, **kwargs)
        bureau = _bureau(bureau_cleaned, config, train_mode, suffix, **kwargs)
        credit_card_balance_cleaned = _credit_card_balance_cleaning(config, suffix, **kwargs)
        credit_card_balance = _credit_card_balance(credit_card_balance_cleaned, config, train_mode, suffix, **kwargs)
        pos_cash_balance = _pos_cash_balance(config, train_mode, suffix, **kwargs)
        previous_application_cleaned = _previous_application_cleaning(config, suffix, **kwargs)
        previous_application = _previous_application(previous_application_cleaned, config, train_mode, suffix, **kwargs)
        installment_payments = _installment_payments(config, train_mode, suffix, **kwargs)

        application_agg = _application_groupby_agg(config, train_mode, suffix, **kwargs)
        bureau_agg = _bureau_groupby_agg(bureau_cleaned, config, train_mode, suffix, **kwargs)
        credit_card_balance_agg = _credit_card_balance_groupby_agg(credit_card_balance_cleaned,
                                                                   config, train_mode, suffix, **kwargs)
        installments_payments_agg = _installments_payments_groupby_agg(config, train_mode, suffix, **kwargs)
        pos_cash_balance_agg = _pos_cash_balance_groupby_agg(config, train_mode, suffix, **kwargs)
        previous_applications_agg = _previous_applications_groupby_agg(previous_application_cleaned,
                                                                       config, train_mode, suffix, **kwargs)
        categorical_encoder = _categorical_encoders(config, train_mode, suffix, **kwargs)
        feature_combiner = _join_features(numerical_features=[application,
                                                              application_agg,
                                                              bureau,
                                                              bureau_agg,
                                                              credit_card_balance,
                                                              credit_card_balance_agg,
                                                              installment_payments,
                                                              installments_payments_agg,
                                                              pos_cash_balance,
                                                              pos_cash_balance_agg,
                                                              previous_application,
                                                              previous_applications_agg,
                                                              ],
                                          numerical_features_valid=[],
                                          categorical_features=[categorical_encoder
                                                                ],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix,
                                          **kwargs)

        return feature_combiner


def preprocessing_fillna(features, config, train_mode, suffix, **kwargs):
    if train_mode:
        features_train, features_valid = features
        fillna = Step(name='fillna{}'.format(suffix),
                      transformer=_fillna(**config.preprocessing),
                      input_steps=[features_train, features_valid],
                      adapter=Adapter({'X': E(features_train.name, 'features'),
                                       'X_valid': E(features_valid.name, 'features'),
                                       }),
                      experiment_directory=config.pipeline.experiment_directory,
                      **kwargs
                      )
    else:
        fillna = Step(name='fillna{}'.format(suffix),
                      transformer=_fillna(**config.preprocessing),
                      input_steps=[features],
                      adapter=Adapter({'X': E(features.name, 'features')}),
                      experiment_directory=config.pipeline.experiment_directory,
                      **kwargs
                      )
    return fillna


def stacking_features(config, train_mode, suffix, **kwargs):
    features = Step(name='stacking_features{}'.format(suffix),
                    transformer=IdentityOperation(),
                    input_data=['input'],
                    adapter=Adapter({'numerical_features': E('input', 'X')}),
                    experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        features_valid = Step(name='stacking_features_valid{}'.format(suffix),
                              transformer=IdentityOperation(),
                              input_data=['input'],
                              adapter=Adapter({'numerical_features': E('input', 'X_valid')}),
                              experiment_directory=config.pipeline.experiment_directory, **kwargs)
        feature_combiner, feature_combiner_valid = _join_features(numerical_features=[features],
                                                                  numerical_features_valid=[features_valid],
                                                                  categorical_features=[],
                                                                  categorical_features_valid=[],
                                                                  config=config,
                                                                  train_mode=train_mode,
                                                                  suffix=suffix, **kwargs)
        return feature_combiner, feature_combiner_valid
    else:
        feature_combiner = _join_features(numerical_features=[features],
                                          numerical_features_valid=[],
                                          categorical_features=[],
                                          categorical_features_valid=[],
                                          config=config,
                                          train_mode=train_mode,
                                          suffix=suffix, **kwargs)
        return feature_combiner


def _join_features(numerical_features,
                   numerical_features_valid,
                   categorical_features,
                   categorical_features_valid,
                   config, train_mode, suffix,
                   **kwargs):
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
    if train_mode:
        feature_joiner_valid = Step(name='feature_joiner_valid{}'.format(suffix),
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
                                    experiment_directory=config.pipeline.experiment_directory,
                                    persist_output=persist_output,
                                    cache_output=cache_output,
                                    load_persisted_output=load_persisted_output)

        return feature_joiner, feature_joiner_valid

    else:
        return feature_joiner


def _categorical_encoders(config, train_mode, suffix, **kwargs):
    categorical_encoder = Step(name='categorical_encoder{}'.format(suffix),
                               transformer=fe.CategoricalEncoder(**config.preprocessing.categorical_encoder),
                               input_data=['application'],
                               adapter=Adapter({'X': E('application', 'X'),
                                                'y': E('application', 'y')}
                                               ),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)
    if train_mode:
        categorical_encoder_valid = Step(name='categorical_encoder_valid{}'.format(suffix),
                                         transformer=categorical_encoder,
                                         input_data=['application'],
                                         adapter=Adapter(
                                             {'X': E('application', 'X_valid'),
                                              'y': E('application', 'y_valid')}
                                         ),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)
        return categorical_encoder, categorical_encoder_valid
    else:
        return categorical_encoder


def _application_groupby_agg(config, train_mode, suffix, **kwargs):
    if train_mode:
        application_cleaning, application_cleaning_valid = _application_cleaning(config, train_mode, suffix, **kwargs)
    else:
        application_cleaning = _application_cleaning(config, train_mode, suffix, **kwargs)

    application_groupby_agg = Step(name='application_groupby_agg{}'.format(suffix),
                                   transformer=fe.GroupbyAggregateDiffs(**config.applications.aggregations),
                                   input_steps=[application_cleaning],
                                   adapter=Adapter({'main_table': E(application_cleaning.name, 'X')}),
                                   experiment_directory=config.pipeline.experiment_directory,
                                   **kwargs)

    if train_mode:

        application_groupby_agg_valid = Step(name='application_groupby_agg_valid{}'.format(suffix),
                                             transformer=application_groupby_agg,
                                             input_steps=[application_cleaning_valid],
                                             adapter=Adapter({'main_table': E(application_cleaning_valid.name, 'X')}),
                                             experiment_directory=config.pipeline.experiment_directory,
                                             **kwargs)

        return application_groupby_agg, application_groupby_agg_valid

    else:
        return application_groupby_agg


def _bureau_groupby_agg(bureau_cleaned, config, train_mode, suffix, **kwargs):
    bureau_groupby_agg = Step(name='bureau_groupby_agg',
                              transformer=fe.GroupbyAggregate(**config.bureau),
                              input_steps=[bureau_cleaned],
                              adapter=Adapter({'table': E(bureau_cleaned.name, 'bureau')}),
                              experiment_directory=config.pipeline.experiment_directory,
                              **kwargs)

    bureau_agg_merge = Step(name='bureau_agg_merge{}'.format(suffix),
                            transformer=fe.GroupbyMerge(**config.bureau),
                            input_data=['application'],
                            input_steps=[bureau_groupby_agg],
                            adapter=Adapter({'table': E('application', 'X'),
                                             'features': E(bureau_groupby_agg.name, 'features_table')}),
                            experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        bureau_agg_merge_valid = Step(name='bureau_agg_merge_valid{}'.format(suffix),
                                      transformer=bureau_agg_merge,
                                      input_data=['application'],
                                      input_steps=[bureau_groupby_agg],
                                      adapter=Adapter({'table': E('application', 'X_valid'),
                                                       'features': E(bureau_groupby_agg.name, 'features_table')}),
                                      experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return bureau_agg_merge, bureau_agg_merge_valid
    else:
        return bureau_agg_merge


def _credit_card_balance_groupby_agg(credit_card_balance_cleaned, config, train_mode, suffix, **kwargs):
    credit_card_balance_groupby_agg = Step(name='credit_card_balance_groupby_agg',
                                           transformer=fe.GroupbyAggregate(**config.credit_card_balance),
                                           input_steps=[credit_card_balance_cleaned],
                                           adapter=Adapter({'table': E(credit_card_balance_cleaned.name,
                                                                       'credit_card')}),
                                           experiment_directory=config.pipeline.experiment_directory,
                                           **kwargs)

    credit_card_balance_agg_merge = Step(name='credit_card_balance_agg_merge{}'.format(suffix),
                                         transformer=fe.GroupbyMerge(**config.credit_card_balance),
                                         input_data=['application'],
                                         input_steps=[credit_card_balance_groupby_agg],
                                         adapter=Adapter({'table': E('application', 'X'),
                                                          'features': E(credit_card_balance_groupby_agg.name,
                                                                        'features_table')}),
                                         experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        credit_card_balance_agg_merge_valid = Step(name='credit_card_balance_agg_merge_valid{}'.format(suffix),
                                                   transformer=credit_card_balance_agg_merge,
                                                   input_data=['application'],
                                                   input_steps=[credit_card_balance_groupby_agg],
                                                   adapter=Adapter({'table': E('application', 'X_valid'),
                                                                    'features': E(credit_card_balance_groupby_agg.name,
                                                                                  'features_table')}),
                                                   experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return credit_card_balance_agg_merge, credit_card_balance_agg_merge_valid
    else:
        return credit_card_balance_agg_merge


def _installments_payments_groupby_agg(config, train_mode, suffix, **kwargs):
    installments_payments_groupby_agg = Step(name='installments_payments_groupby_agg',
                                             transformer=fe.GroupbyAggregate(**config.installments_payments),
                                             input_data=['installments_payments'],
                                             adapter=Adapter({'table': E('installments_payments', 'X')}),
                                             experiment_directory=config.pipeline.experiment_directory,
                                             **kwargs)

    installments_payments_agg_merge = Step(name='installments_payments_agg_merge{}'.format(suffix),
                                           transformer=fe.GroupbyMerge(**config.installments_payments),
                                           input_data=['application'],
                                           input_steps=[installments_payments_groupby_agg],
                                           adapter=Adapter({'table': E('application', 'X'),
                                                            'features': E(installments_payments_groupby_agg.name,
                                                                          'features_table')}),
                                           experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        installments_payments_agg_merge_valid = Step(name='installments_payments_agg_merge_valid{}'.format(suffix),
                                                     transformer=installments_payments_agg_merge,
                                                     input_data=['application'],
                                                     input_steps=[installments_payments_groupby_agg],
                                                     adapter=Adapter({'table': E('application', 'X_valid'),
                                                                      'features': E(
                                                                          installments_payments_groupby_agg.name,
                                                                          'features_table')}),
                                                     experiment_directory=config.pipeline.experiment_directory,
                                                     **kwargs)
        return installments_payments_agg_merge, installments_payments_agg_merge_valid
    else:
        return installments_payments_agg_merge


def _pos_cash_balance_groupby_agg(config, train_mode, suffix, **kwargs):
    pos_cash_balance_groupby_agg = Step(name='pos_cash_balance_groupby_agg',
                                        transformer=fe.GroupbyAggregate(**config.pos_cash_balance),
                                        input_data=['pos_cash_balance'],
                                        adapter=Adapter({'table': E('pos_cash_balance', 'X')}),
                                        experiment_directory=config.pipeline.experiment_directory,
                                        **kwargs)

    pos_cash_balance_agg_merge = Step(name='pos_cash_balance_agg_merge{}'.format(suffix),
                                      transformer=fe.GroupbyMerge(**config.pos_cash_balance),
                                      input_data=['application'],
                                      input_steps=[pos_cash_balance_groupby_agg],
                                      adapter=Adapter({'table': E('application', 'X'),
                                                       'features': E(pos_cash_balance_groupby_agg.name,
                                                                     'features_table')}),
                                      experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        pos_cash_balance_agg_merge_valid = Step(name='pos_cash_balance_agg_merge_valid{}'.format(suffix),
                                                transformer=pos_cash_balance_agg_merge,
                                                input_data=['application'],
                                                input_steps=[pos_cash_balance_groupby_agg],
                                                adapter=Adapter({'table': E('application', 'X_valid'),
                                                                 'features': E(
                                                                     pos_cash_balance_groupby_agg.name,
                                                                     'features_table')}),
                                                experiment_directory=config.pipeline.experiment_directory,
                                                **kwargs)
        return pos_cash_balance_agg_merge, pos_cash_balance_agg_merge_valid
    else:
        return pos_cash_balance_agg_merge


def _previous_applications_groupby_agg(previous_application_cleaned, config, train_mode, suffix, **kwargs):
    previous_applications_groupby_agg = Step(name='previous_applications_groupby_agg',
                                             transformer=fe.GroupbyAggregate(**config.previous_applications),
                                             input_steps=[previous_application_cleaned],
                                             adapter=Adapter({'table': E(previous_application_cleaned.name,
                                                                         'previous_application')}),
                                             experiment_directory=config.pipeline.experiment_directory, **kwargs)

    previous_applications_agg_merge = Step(name='previous_applications_agg_merge{}'.format(suffix),
                                           transformer=fe.GroupbyMerge(**config.previous_applications),
                                           input_data=['application'],
                                           input_steps=[previous_applications_groupby_agg],
                                           adapter=Adapter({'table': E('application', 'X'),
                                                            'features': E(previous_applications_groupby_agg.name,
                                                                          'features_table')}),
                                           experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        previous_applications_agg_merge_valid = Step(name='previous_applications_agg_merge_valid{}'.format(suffix),
                                                     transformer=previous_applications_agg_merge,
                                                     input_data=['application'],
                                                     input_steps=[previous_applications_groupby_agg],
                                                     adapter=Adapter({'table': E('application', 'X_valid'),
                                                                      'features': E(
                                                                          previous_applications_groupby_agg.name,
                                                                          'features_table')}),
                                                     experiment_directory=config.pipeline.experiment_directory,
                                                     **kwargs)
        return previous_applications_agg_merge, previous_applications_agg_merge_valid
    else:
        return previous_applications_agg_merge


def _application_cleaning(config, train_mode, suffix, **kwargs):
    application_cleaning = Step(name='application_cleaning{}'.format(suffix),
                                transformer=dc.ApplicationCleaning(**config.preprocessing.impute_missing),
                                input_data=['application'],
                                adapter=Adapter({'X': E('application', 'X')}),
                                experiment_directory=config.pipeline.experiment_directory,
                                **kwargs)
    if train_mode:
        application_cleaning_valid = Step(name='application_cleaning_valid{}'.format(suffix),
                                          transformer=dc.ApplicationCleaning(),
                                          input_data=['application'],
                                          adapter=Adapter({'X': E('application', 'X_valid')}),
                                          experiment_directory=config.pipeline.experiment_directory,
                                          **kwargs)
        return application_cleaning, application_cleaning_valid
    else:
        return application_cleaning


def _application(config, train_mode, suffix, **kwargs):
    if train_mode:
        application_cleaning, application_cleaning_valid = _application_cleaning(config, train_mode, suffix, **kwargs)
    else:
        application_cleaning = _application_cleaning(config, train_mode, suffix, **kwargs)

    application = Step(name='application_hand_crafted{}'.format(suffix),
                       transformer=fe.ApplicationFeatures(**config.applications.columns),
                       input_steps=[application_cleaning],
                       adapter=Adapter({'X': E(application_cleaning.name, 'X')}),
                       experiment_directory=config.pipeline.experiment_directory,
                       **kwargs)
    if train_mode:
        application_valid = Step(name='application_hand_crafted_valid{}'.format(suffix),
                                 transformer=application,
                                 input_steps=[application_cleaning_valid],
                                 adapter=Adapter({'X': E(application_cleaning_valid.name, 'X')}),
                                 experiment_directory=config.pipeline.experiment_directory,
                                 **kwargs)
        return application, application_valid
    else:
        return application


def _bureau_cleaning(config, suffix, **kwargs):
    bureau_cleaning = Step(name='bureau_cleaning',
                           transformer=dc.BureauCleaning(**config.preprocessing.impute_missing),
                           input_data=['bureau'],
                           adapter=Adapter({'bureau': E('bureau', 'X')}),
                           experiment_directory=config.pipeline.experiment_directory,
                           **kwargs)

    return bureau_cleaning


def _bureau(bureau_cleaned, config, train_mode, suffix, **kwargs):
    bureau_hand_crafted = Step(name='bureau_hand_crafted',
                               transformer=fe.BureauFeatures(**config.bureau),
                               input_steps=[bureau_cleaned],
                               adapter=Adapter({'bureau': E(bureau_cleaned.name, 'bureau')}),
                               experiment_directory=config.pipeline.experiment_directory,
                               **kwargs)

    bureau_hand_crafted_merge = Step(name='bureau_hand_crafted_merge{}'.format(suffix),
                                     transformer=fe.GroupbyMerge(**config.bureau),
                                     input_data=['application'],
                                     input_steps=[bureau_hand_crafted],
                                     adapter=Adapter({'table': E('application', 'X'),
                                                      'features': E(bureau_hand_crafted.name, 'features_table')}),
                                     experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        bureau_hand_crafted_merge_valid = Step(name='bureau_hand_crafted_merge_valid{}'.format(suffix),
                                               transformer=bureau_hand_crafted_merge,
                                               input_data=['application'],
                                               input_steps=[bureau_hand_crafted],
                                               adapter=Adapter({'table': E('application', 'X_valid'),
                                                                'features': E(bureau_hand_crafted.name,
                                                                              'features_table')}),
                                               experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return bureau_hand_crafted_merge, bureau_hand_crafted_merge_valid
    else:
        return bureau_hand_crafted_merge


def _credit_card_balance_cleaning(config, suffix, **kwargs):
    credit_card_balance_cleaning = Step(name='credit_card_balance_cleaning{}'.format(suffix),
                                        transformer=dc.CreditCardCleaning(
                                            **config.preprocessing.impute_missing),
                                        input_data=['credit_card_balance'],
                                        adapter=Adapter({'credit_card': E('credit_card_balance', 'X')}),
                                        experiment_directory=config.pipeline.experiment_directory,
                                        **kwargs)

    return credit_card_balance_cleaning


def _credit_card_balance(credit_card_balance_cleaned, config, train_mode, suffix, **kwargs):
    credit_card_balance_hand_crafted = Step(name='credit_card_balance_hand_crafted',
                                            transformer=fe.CreditCardBalanceFeatures(**config.credit_card_balance),
                                            input_steps=[credit_card_balance_cleaned],
                                            adapter=Adapter({'credit_card': E(credit_card_balance_cleaned.name,
                                                                              'credit_card')}),
                                            experiment_directory=config.pipeline.experiment_directory,
                                            **kwargs)

    credit_card_balance_hand_crafted_merge = Step(name='credit_card_balance_hand_crafted_merge{}'.format(suffix),
                                                  transformer=fe.GroupbyMerge(**config.credit_card_balance),
                                                  input_data=['application'],
                                                  input_steps=[credit_card_balance_hand_crafted],
                                                  adapter=Adapter({'table': E('application', 'X'),
                                                                   'features': E(credit_card_balance_hand_crafted.name,
                                                                                 'features_table')}),
                                                  experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        credit_card_balance_hand_crafted_merge_valid = Step(
            name='credit_card_balance_hand_crafted_merge_valid{}'.format(suffix),
            transformer=credit_card_balance_hand_crafted_merge,
            input_data=['application'],
            input_steps=[credit_card_balance_hand_crafted],
            adapter=Adapter({'table': E('application', 'X_valid'),
                             'features': E(credit_card_balance_hand_crafted.name,
                                           'features_table')}),
            experiment_directory=config.pipeline.experiment_directory, **kwargs)
        return credit_card_balance_hand_crafted_merge, credit_card_balance_hand_crafted_merge_valid
    else:
        return credit_card_balance_hand_crafted_merge


def _pos_cash_balance(config, train_mode, suffix, **kwargs):
    pos_cash_balance_hand_crafted = Step(name='pos_cash_balance_hand_crafted',
                                         transformer=fe.POSCASHBalanceFeatures(**config.pos_cash_balance),
                                         input_data=['pos_cash_balance'],
                                         adapter=Adapter({'pos_cash': E('pos_cash_balance', 'X')}),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)

    pos_cash_balance_hand_crafted_merge = Step(name='pos_cash_balance_hand_crafted_merge{}'.format(suffix),
                                               transformer=fe.GroupbyMerge(**config.pos_cash_balance),
                                               input_data=['application'],
                                               input_steps=[pos_cash_balance_hand_crafted],
                                               adapter=Adapter({'table': E('application', 'X'),
                                                                'features': E(pos_cash_balance_hand_crafted.name,
                                                                              'features_table')}),
                                               experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        pos_cash_balance_hand_crafted_merge_valid = Step(
            name='pos_cash_balance_hand_crafted_merge_valid{}'.format(suffix),
            transformer=pos_cash_balance_hand_crafted_merge,
            input_data=['application'],
            input_steps=[pos_cash_balance_hand_crafted],
            adapter=Adapter({'table': E('application', 'X_valid'),
                             'features': E(
                                 pos_cash_balance_hand_crafted.name,
                                 'features_table')}),
            experiment_directory=config.pipeline.experiment_directory,
            **kwargs)
        return pos_cash_balance_hand_crafted_merge, pos_cash_balance_hand_crafted_merge_valid
    else:
        return pos_cash_balance_hand_crafted_merge


def _previous_application_cleaning(config, suffix, **kwargs):
    previous_application_cleaning = Step(name='previous_application_cleaning{}'.format(suffix),
                                         transformer=dc.PreviousApplicationCleaning(
                                             **config.preprocessing.impute_missing),
                                         input_data=['previous_application'],
                                         adapter=Adapter({'previous_application': E('previous_application', 'X')}),
                                         experiment_directory=config.pipeline.experiment_directory,
                                         **kwargs)

    return previous_application_cleaning


def _previous_application(previous_application_cleaned, config, train_mode, suffix, **kwargs):
    previous_applications_hand_crafted = Step(name='previous_applications_hand_crafted',
                                              transformer=fe.PreviousApplicationFeatures(
                                                  **config.previous_applications),
                                              input_steps=[previous_application_cleaned],
                                              adapter=Adapter(
                                                  {'prev_applications': E(previous_application_cleaned.name,
                                                                          'previous_application')}),
                                              experiment_directory=config.pipeline.experiment_directory,
                                              **kwargs)

    previous_applications_hand_crafted_merge = Step(name='previous_applications_hand_crafted_merge{}'.format(suffix),
                                                    transformer=fe.GroupbyMerge(**config.previous_applications),
                                                    input_data=['application'],
                                                    input_steps=[previous_applications_hand_crafted],
                                                    adapter=Adapter({'table': E('application', 'X'),
                                                                     'features': E(
                                                                         previous_applications_hand_crafted.name,
                                                                         'features_table')}),
                                                    experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        previous_applications_hand_crafted_merge_valid = Step(
            name='previous_applications_hand_crafted_merge_valid{}'.format(suffix),
            transformer=previous_applications_hand_crafted_merge,
            input_data=['application'],
            input_steps=[previous_applications_hand_crafted],
            adapter=Adapter({'table': E('application', 'X_valid'),
                             'features': E(
                                 previous_applications_hand_crafted.name,
                                 'features_table')}),
            experiment_directory=config.pipeline.experiment_directory,
            **kwargs)
        return previous_applications_hand_crafted_merge, previous_applications_hand_crafted_merge_valid
    else:
        return previous_applications_hand_crafted_merge


def _installment_payments(config, train_mode, suffix, **kwargs):
    installment_payments_hand_crafted = Step(name='installment_payments_hand_crafted',
                                             transformer=fe.InstallmentPaymentsFeatures(**config.installments_payments),
                                             input_data=['installments_payments'],
                                             adapter=Adapter({'installments': E('installments_payments', 'X')}),
                                             experiment_directory=config.pipeline.experiment_directory,
                                             **kwargs)

    installment_payments_hand_crafted_merge = Step(name='installment_payments_hand_crafted_merge{}'.format(suffix),
                                                   transformer=fe.GroupbyMerge(**config.installments_payments),
                                                   input_data=['application'],
                                                   input_steps=[installment_payments_hand_crafted],
                                                   adapter=Adapter({'table': E('application', 'X'),
                                                                    'features': E(
                                                                        installment_payments_hand_crafted.name,
                                                                        'features_table')}),
                                                   experiment_directory=config.pipeline.experiment_directory, **kwargs)

    if train_mode:
        installment_payments_hand_crafted_merge_valid = Step(
            name='installment_payments_hand_crafted_merge_valid{}'.format(suffix),
            transformer=installment_payments_hand_crafted_merge,
            input_data=['application'],
            input_steps=[installment_payments_hand_crafted],
            adapter=Adapter({'table': E('application', 'X_valid'),
                             'features': E(
                                 installment_payments_hand_crafted.name,
                                 'features_table')}),
            experiment_directory=config.pipeline.experiment_directory,
            **kwargs)
        return installment_payments_hand_crafted_merge, installment_payments_hand_crafted_merge_valid
    else:
        return installment_payments_hand_crafted_merge


def _fillna(fillna_value):
    def _inner_fillna(X, X_valid=None):
        if X_valid is None:
            return {'X': X.fillna(fillna_value)}
        else:
            return {'X': X.fillna(fillna_value),
                    'X_valid': X_valid.fillna(fillna_value)}

    return make_transformer(_inner_fillna)
