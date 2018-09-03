from attrdict import AttrDict
from deepsense import neptune

from .utils import read_params, parameter_eval

ctx = neptune.Context()
params = read_params(ctx, fallback_file='../configs/neptune.yaml')

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 100

ID_COLUMNS = ['SK_ID_CURR']
TARGET_COLUMNS = ['TARGET']

CATEGORICAL_COLUMNS = ['CODE_GENDER',
                       'EMERGENCYSTATE_MODE',
                       'FLAG_CONT_MOBILE',
                       'FLAG_DOCUMENT_3',
                       'FLAG_DOCUMENT_4',
                       'FLAG_DOCUMENT_5',
                       'FLAG_DOCUMENT_6',
                       'FLAG_DOCUMENT_7',
                       'FLAG_DOCUMENT_8',
                       'FLAG_DOCUMENT_9',
                       'FLAG_DOCUMENT_11',
                       'FLAG_DOCUMENT_18',
                       'FLAG_EMAIL',
                       'FLAG_EMP_PHONE',
                       'FLAG_MOBIL',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'FLAG_PHONE',
                       'FLAG_WORK_PHONE',
                       'FONDKAPREMONT_MODE',
                       'HOUR_APPR_PROCESS_START',
                       'HOUSETYPE_MODE',
                       'LIVE_CITY_NOT_WORK_CITY',
                       'LIVE_REGION_NOT_WORK_REGION',
                       'NAME_CONTRACT_TYPE',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'ORGANIZATION_TYPE',
                       'REG_CITY_NOT_LIVE_CITY',
                       'REG_CITY_NOT_WORK_CITY',
                       'REG_REGION_NOT_LIVE_REGION',
                       'REG_REGION_NOT_WORK_REGION',
                       'WALLSMATERIAL_MODE',
                       'WEEKDAY_APPR_PROCESS_START']

NUMERICAL_COLUMNS = ['AMT_ANNUITY',
                     'AMT_CREDIT',
                     'AMT_INCOME_TOTAL',
                     'AMT_REQ_CREDIT_BUREAU_HOUR',
                     'AMT_REQ_CREDIT_BUREAU_DAY',
                     'AMT_REQ_CREDIT_BUREAU_WEEK',
                     'AMT_REQ_CREDIT_BUREAU_MON',
                     'AMT_REQ_CREDIT_BUREAU_QRT',
                     'AMT_REQ_CREDIT_BUREAU_YEAR',
                     'APARTMENTS_AVG',
                     'BASEMENTAREA_AVG',
                     'COMMONAREA_AVG',
                     'CNT_CHILDREN',
                     'CNT_FAM_MEMBERS',
                     'DAYS_BIRTH',
                     'DAYS_EMPLOYED',
                     'DAYS_ID_PUBLISH',
                     'DAYS_LAST_PHONE_CHANGE',
                     'DAYS_REGISTRATION',
                     'DEF_30_CNT_SOCIAL_CIRCLE',
                     'DEF_60_CNT_SOCIAL_CIRCLE',
                     'ELEVATORS_AVG',
                     'ENTRANCES_AVG',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'EXT_SOURCE_3',
                     'FLOORSMAX_AVG',
                     'FLOORSMIN_AVG',
                     'LANDAREA_AVG',
                     'LIVINGAPARTMENTS_AVG',
                     'LIVINGAREA_AVG',
                     'NONLIVINGAPARTMENTS_AVG',
                     'NONLIVINGAREA_AVG',
                     'OBS_30_CNT_SOCIAL_CIRCLE',
                     'OWN_CAR_AGE',
                     'REGION_POPULATION_RELATIVE',
                     'REGION_RATING_CLIENT',
                     'TOTALAREA_MODE',
                     'YEARS_BEGINEXPLUATATION_AVG',
                     'YEARS_BUILD_AVG']

USELESS_COLUMNS = ['FLAG_DOCUMENT_10',
                   'FLAG_DOCUMENT_12',
                   'FLAG_DOCUMENT_13',
                   'FLAG_DOCUMENT_14',
                   'FLAG_DOCUMENT_15',
                   'FLAG_DOCUMENT_16',
                   'FLAG_DOCUMENT_17',
                   'FLAG_DOCUMENT_19',
                   'FLAG_DOCUMENT_2',
                   'FLAG_DOCUMENT_20',
                   'FLAG_DOCUMENT_21']

HIGHLY_CORRELATED_NUMERICAL_COLUMNS = ['AMT_GOODS_PRICE',
                                       'APARTMENTS_MEDI',
                                       'APARTMENTS_MODE',
                                       'BASEMENTAREA_MEDI',
                                       'BASEMENTAREA_MODE',
                                       'COMMONAREA_MEDI',
                                       'COMMONAREA_MODE',
                                       'ELEVATORS_MEDI',
                                       'ELEVATORS_MODE',
                                       'ENTRANCES_MEDI',
                                       'ENTRANCES_MODE',
                                       'FLAG_EMP_PHONE',
                                       'FLOORSMAX_MEDI',
                                       'FLOORSMAX_MODE',
                                       'FLOORSMIN_MEDI',
                                       'FLOORSMIN_MODE',
                                       'LANDAREA_MEDI',
                                       'LANDAREA_MODE',
                                       'LIVINGAPARTMENTS_MEDI',
                                       'LIVINGAPARTMENTS_MODE',
                                       'LIVINGAREA_MEDI',
                                       'LIVINGAREA_MODE',
                                       'NONLIVINGAPARTMENTS_MEDI',
                                       'NONLIVINGAPARTMENTS_MODE',
                                       'NONLIVINGAREA_MEDI',
                                       'NONLIVINGAREA_MODE',
                                       'OBS_60_CNT_SOCIAL_CIRCLE',
                                       'REGION_RATING_CLIENT_W_CITY',
                                       'YEARS_BEGINEXPLUATATION_MEDI',
                                       'YEARS_BEGINEXPLUATATION_MODE',
                                       'YEARS_BUILD_MEDI',
                                       'YEARS_BUILD_MODE']

cols_to_agg = ['AMT_CREDIT',
               'AMT_ANNUITY',
               'AMT_INCOME_TOTAL',
               'AMT_GOODS_PRICE',
               'EXT_SOURCE_1',
               'EXT_SOURCE_2',
               'EXT_SOURCE_3',
               'OWN_CAR_AGE',
               'REGION_POPULATION_RELATIVE',
               'DAYS_REGISTRATION',
               'CNT_CHILDREN',
               'CNT_FAM_MEMBERS',
               'DAYS_ID_PUBLISH',
               'DAYS_BIRTH',
               'DAYS_EMPLOYED'
               ]

aggs = ['min', 'mean', 'max', 'sum', 'var', 'median']
aggregation_pairs = [(col, agg) for col in cols_to_agg for agg in aggs]

APPLICATION_AGGREGATION_RECIPIES = [
    (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], aggregation_pairs),
    (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], aggregation_pairs),
    (['NAME_FAMILY_STATUS', 'CODE_GENDER'], aggregation_pairs),
    (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                            ('AMT_INCOME_TOTAL', 'mean'),
                                            ('DAYS_REGISTRATION', 'mean'),
                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                 ('CNT_CHILDREN', 'mean'),
                                                 ('DAYS_ID_PUBLISH', 'mean')]),
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                           ('EXT_SOURCE_2', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                  ('APARTMENTS_AVG', 'mean'),
                                                  ('BASEMENTAREA_AVG', 'mean'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('EXT_SOURCE_3', 'mean'),
                                                  ('NONLIVINGAREA_AVG', 'mean'),
                                                  ('OWN_CAR_AGE', 'mean'),
                                                  ('YEARS_BUILD_AVG', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                            ('EXT_SOURCE_1', 'mean')]),
    (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                           ('CNT_CHILDREN', 'mean'),
                           ('CNT_FAM_MEMBERS', 'mean'),
                           ('DAYS_BIRTH', 'mean'),
                           ('DAYS_EMPLOYED', 'mean'),
                           ('DAYS_ID_PUBLISH', 'mean'),
                           ('DAYS_REGISTRATION', 'mean'),
                           ('EXT_SOURCE_1', 'mean'),
                           ('EXT_SOURCE_2', 'mean'),
                           ('EXT_SOURCE_3', 'mean')]),
]

BUREAU_AGGREGATION_RECIPIES = [('CREDIT_TYPE', 'count'),
                               ('CREDIT_ACTIVE', 'size')
                               ]
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_CREDIT_SUM',
                   'AMT_CREDIT_SUM_DEBT',
                   'AMT_CREDIT_SUM_LIMIT',
                   'AMT_CREDIT_SUM_OVERDUE',
                   'AMT_CREDIT_MAX_OVERDUE',
                   'CNT_CREDIT_PROLONG',
                   'CREDIT_DAY_OVERDUE',
                   'DAYS_CREDIT',
                   'DAYS_CREDIT_ENDDATE',
                   'DAYS_CREDIT_UPDATE'
                   ]:
        BUREAU_AGGREGATION_RECIPIES.append((select, agg))
BUREAU_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_AGGREGATION_RECIPIES)]

CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_BALANCE',
                   'AMT_CREDIT_LIMIT_ACTUAL',
                   'AMT_DRAWINGS_ATM_CURRENT',
                   'AMT_DRAWINGS_CURRENT',
                   'AMT_DRAWINGS_OTHER_CURRENT',
                   'AMT_DRAWINGS_POS_CURRENT',
                   'AMT_PAYMENT_CURRENT',
                   'CNT_DRAWINGS_ATM_CURRENT',
                   'CNT_DRAWINGS_CURRENT',
                   'CNT_DRAWINGS_OTHER_CURRENT',
                   'CNT_INSTALMENT_MATURE_CUM',
                   'MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]

INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_INSTALMENT',
                   'AMT_PAYMENT',
                   'DAYS_ENTRY_PAYMENT',
                   'DAYS_INSTALMENT',
                   'NUM_INSTALMENT_NUMBER',
                   'NUM_INSTALMENT_VERSION'
                   ]:
        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]

POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]

PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

SOLUTION_CONFIG = AttrDict({
    'pipeline': {'experiment_directory': params.experiment_directory
                 },
    'hyperparameter_search': {'method': parameter_eval(params.hyperparameter_search__method),
                              'runs': parameter_eval(params.hyperparameter_search__runs)
                              },

    'feature_joiner': {'id_column': 'SK_ID_CURR',
                       'use_nan_count': params.use_nan_count
                       },

    'feature_selection': {'use_application': params.use_application,
                          'use_bureau': params.use_bureau,
                          'use_bureau_balance': params.use_bureau_balance,
                          'use_credit_card_balance': params.use_credit_card_balance,
                          'use_installments_payments': params.use_installments_payments,
                          'use_pos_cash_balance': params.use_pos_cash_balance,
                          'use_previous_applications': params.use_previous_applications,
                          'use_application_previous_application': params.use_application_previous_applications,

                          'use_application_aggregations': params.use_application_aggregations,
                          'use_bureau_aggregations': params.use_bureau_aggregations,
                          'use_credit_card_balance_aggregations': params.use_credit_card_balance_aggregations,
                          'use_installments_payments_aggregations': params.use_installments_payments_aggregations,
                          'use_pos_cash_balance_aggregations': params.use_pos_cash_balance_aggregations,
                          'use_previous_applications_aggregations': params.use_previous_applications_aggregations,

                          'use_application_categorical_features': params.use_application_categorical_features,
                          'use_previous_application_categorical_features':
                              params.use_previous_application_categorical_features,
                          'use_application_previous_application_categorical_features':
                              params.use_application_previous_application_categorical_features,
                          },

    'preprocessing': {'impute_missing': {'fill_missing': params.fill_missing,
                                         'fill_value': params.fill_value},
                      },

    'xgb_preprocessing': {'one_hot_encoder': {'drop_invariant': True}
                          },

    'neural_network_preprocessing': {'one_hot_encoder': {'drop_invariant': False},
                                     'fillna': {'fill_value': params.fill_value},
                                     },

    'sklearn_preprocessing': {'one_hot_encoder': {'drop_invariant': True},
                              'fillna': {'fill_value': params.fill_value},
                              },

    'idx_merge': {'id_column': 'SK_ID_CURR'},

    'applications': {'table_name': 'applications',
                     'id_column': 'SK_ID_CURR',
                     'categorical_columns': CATEGORICAL_COLUMNS,
                     'numerical_columns': NUMERICAL_COLUMNS,
                     'groupby_aggregations': APPLICATION_AGGREGATION_RECIPIES,
                     'use_diffs_only': params.application_aggregation__use_diffs_only
                     },

    'bureau': {'table_name': 'bureau',
               'id_column': 'SK_ID_CURR',
               'groupby_aggregations': BUREAU_AGGREGATION_RECIPIES,
               'last_k_agg_periods': parameter_eval(params.bureau__last_k_agg_periods),
               'last_k_agg_period_fractions': parameter_eval(params.bureau__last_k_agg_period_fractions),
               'num_workers': params.num_workers
               },

    'bureau_balance': {'table_name': 'bureau_balance',
                       'id_column': 'SK_ID_CURR',
                       'last_k_agg_periods': parameter_eval(params.bureau_balance__last_k_agg_periods),
                       'last_k_agg_period_fractions': parameter_eval(
                           params.bureau_balance__last_k_agg_period_fractions),
                       'last_k_trend_periods': parameter_eval(params.bureau_balance__last_k_trend_periods),
                       'num_workers': params.num_workers
                       },

    'credit_card_balance': {'table_name': 'credit_card_balance',
                            'id_column': 'SK_ID_CURR',
                            'groupby_aggregations': CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES,
                            'num_workers': params.num_workers
                            },

    'installments_payments': {'table_name': 'installments_payments',
                              'id_column': 'SK_ID_CURR',
                              'groupby_aggregations': INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES,
                              'last_k_agg_periods': parameter_eval(params.installments__last_k_agg_periods),
                              'last_k_agg_period_fractions': parameter_eval(
                                  params.installments__last_k_agg_period_fractions),
                              'last_k_trend_periods': parameter_eval(params.installments__last_k_trend_periods),
                              'num_workers': params.num_workers
                              },

    'pos_cash_balance': {'table_name': 'POS_CASH_balance',
                         'id_column': 'SK_ID_CURR',
                         'groupby_aggregations': POS_CASH_BALANCE_AGGREGATION_RECIPIES,
                         'last_k_agg_periods': parameter_eval(params.pos_cash__last_k_agg_periods),
                         'last_k_trend_periods': parameter_eval(params.pos_cash__last_k_trend_periods),
                         'last_k_agg_period_fractions': parameter_eval(
                             params.pos_cash__last_k_agg_period_fractions),
                         'num_workers': params.num_workers
                         },

    'previous_applications': {'table_name': 'previous_application',
                              'id_column': 'SK_ID_CURR',
                              'groupby_aggregations': PREVIOUS_APPLICATION_AGGREGATION_RECIPIES,
                              'numbers_of_applications': parameter_eval(params.previous_applications__last_k_credits),
                              'num_workers': params.num_workers
                              },

    'application_previous_application': {'table_name': 'application_previous_application',
                                         'id_column': 'SK_ID_CURR',
                                         'numbers_of_applications': parameter_eval(
                                             params.application_previous_application__last_k_credits),
                                         },

    'light_gbm': {'device': parameter_eval(params.lgbm__device),
                  'boosting_type': parameter_eval(params.lgbm__boosting_type),
                  'objective': parameter_eval(params.lgbm__objective),
                  'metric': parameter_eval(params.lgbm__metric),
                  'is_unbalance': parameter_eval(params.lgbm__is_unbalance),
                  'scale_pos_weight': parameter_eval(params.lgbm__scale_pos_weight),
                  'learning_rate': parameter_eval(params.lgbm__learning_rate),
                  'max_bin': parameter_eval(params.lgbm__max_bin),
                  'max_depth': parameter_eval(params.lgbm__max_depth),
                  'num_leaves': parameter_eval(params.lgbm__num_leaves),
                  'min_child_samples': parameter_eval(params.lgbm__min_child_samples),
                  'subsample': parameter_eval(params.lgbm__subsample),
                  'colsample_bytree': parameter_eval(params.lgbm__colsample_bytree),
                  'subsample_freq': parameter_eval(params.lgbm__subsample_freq),
                  'min_gain_to_split': parameter_eval(params.lgbm__min_gain_to_split),
                  'reg_lambda': parameter_eval(params.lgbm__reg_lambda),
                  'reg_alpha': parameter_eval(params.lgbm__reg_alpha),
                  'nthread': parameter_eval(params.num_workers),
                  'number_boosting_rounds': parameter_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': parameter_eval(params.lgbm__early_stopping_rounds),
                  'verbose': parameter_eval(params.verbose),
                  'callbacks_on': parameter_eval(params.callbacks_on)
                  },

    'catboost': {'loss_function': parameter_eval(params.catboost__loss_function),
                 'eval_metric': parameter_eval(params.catboost__eval_metric),
                 'iterations': parameter_eval(params.catboost__iterations),
                 'learning_rate': parameter_eval(params.catboost__learning_rate),
                 'depth': parameter_eval(params.catboost__depth),
                 'l2_leaf_reg': parameter_eval(params.catboost__l2_leaf_reg),
                 'model_size_reg': parameter_eval(params.catboost__model_size_reg),
                 'colsample_bylevel': parameter_eval(params.catboost__colsample_bylevel),
                 'border_count': parameter_eval(params.catboost__border_count),
                 'max_ctr_complexity': parameter_eval(params.catboost__max_ctr_complexity),
                 'od_type': parameter_eval(params.catboost__od_type),
                 'od_wait': parameter_eval(params.catboost__od_wait),
                 'random_seed': RANDOM_SEED,
                 'thread_count': params.num_workers,
                 'verbose': params.verbose,
                 },

    'xgboost': {'booster': parameter_eval(params.xgb__booster),
                'objective': parameter_eval(params.xgb__objective),
                'tree_method': parameter_eval(params.xgb__tree_method),
                'eval_metric': parameter_eval(params.xgb__eval_metric),
                'eta': parameter_eval(params.xgb__eta),
                'max_depth': parameter_eval(params.xgb__max_depth),
                'subsample': parameter_eval(params.xgb__subsample),
                'colsample_bytree': parameter_eval(params.xgb__colsample_bytree),
                'colsample_bylevel': parameter_eval(params.xgb__colsample_bylevel),
                'min_child_weight': parameter_eval(params.xgb__min_child_weight),
                'lambda': parameter_eval(params.xgb__lambda),
                'alpha': parameter_eval(params.xgb__alpha),
                'max_bin': parameter_eval(params.xgb__max_bin),
                'num_leaves': parameter_eval(params.xgb__max_leaves),
                'nthread': parameter_eval(params.num_workers),
                'nrounds': parameter_eval(params.xgb__nrounds),
                'early_stopping_rounds': parameter_eval(params.xgb__early_stopping_rounds),
                'scale_pos_weight': parameter_eval(params.xgb__scale_pos_weight),
                'verbose': parameter_eval(params.verbose),
                },

    'random_forest': {'n_estimators': parameter_eval(params.rf__n_estimators),
                      'criterion': parameter_eval(params.rf__criterion),
                      'max_features': parameter_eval(params.rf__max_features),
                      'max_depth': parameter_eval(params.rf__max_depth),
                      'min_samples_split': parameter_eval(params.rf__min_samples_split),
                      'min_samples_leaf': parameter_eval(params.rf__min_samples_leaf),
                      'max_leaf_nodes': parameter_eval(params.rf__max_leaf_nodes),
                      'n_jobs': parameter_eval(params.num_workers),
                      'random_state': RANDOM_SEED,
                      'verbose': parameter_eval(params.verbose),
                      'class_weight': parameter_eval(params.rf__class_weight),
                      },

    'log_reg': {'penalty': parameter_eval(params.lr__penalty),
                'tol': parameter_eval(params.lr__tol),
                'C': parameter_eval(params.lr__C),
                'fit_intercept': parameter_eval(params.lr__fit_intercept),
                'class_weight': parameter_eval(params.lr__class_weight),
                'random_state': RANDOM_SEED,
                'solver': parameter_eval(params.lr__solver),
                'max_iter': parameter_eval(params.lr__max_iter),
                'verbose': parameter_eval(params.verbose),
                'n_jobs': parameter_eval(params.num_workers),
                },

    'naive_bayes': {'alpha': parameter_eval(params.nb__alpha),
                    'binarize': parameter_eval(params.nb__binarize),
                    },

    'neural_network': {'architecture_config': {'model_params': {'layers': parameter_eval(params.nn__layers),
                                                                'neurons': parameter_eval(params.nn__neurons),
                                                                'activation': parameter_eval(params.nn__activation),
                                                                'dropout': parameter_eval(params.nn__dropout),
                                                                'batch_norm': parameter_eval(params.nn__batch_norm),
                                                                'l1': parameter_eval(params.nn__l1),
                                                                'l2': parameter_eval(params.nn__l2)},
                                               'optimizer_params': {'lr': parameter_eval(params.nn__learning_rate),
                                                                    'beta_1': parameter_eval(params.nn__beta_1),
                                                                    'beta_2': parameter_eval(params.nn__beta_2)}
                                               },
                       'training_config': {'epochs': parameter_eval(params.nn__epochs),
                                           'batch_size': parameter_eval(params.nn__batch_size)},
                       'callbacks_config': {},
                       },

    'svc': {'kernel': parameter_eval(params.svc__kernel),
            'C': parameter_eval(params.svc__C),
            'degree': parameter_eval(params.svc__degree),
            'gamma': parameter_eval(params.svc__gamma),
            'coef0': parameter_eval(params.svc__coef0),
            'probability': parameter_eval(params.svc__probability),
            'tol': parameter_eval(params.svc__tol),
            'max_iter': parameter_eval(params.svc__max_iter),
            'verbose': parameter_eval(params.verbose),
            'random_state': RANDOM_SEED,
            },

    'tuner': {'light_gbm': {'max_depth': ([3, 8], "choice"),
                            'num_leaves': ([20, 100], "choice"),
                            'min_child_samples': ([40, 200], "choice"),
                            'reg_lambda': ([1e-8, 100.], "log-uniform"),
                            'reg_alpha': ([1e-8, 10.], "log-uniform"),
                            'min_gain_to_split': ([0., 1], "uniform")
                            },
              }
})
