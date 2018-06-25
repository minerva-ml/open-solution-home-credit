import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, parameter_eval

ctx = neptune.Context()
params = read_params(ctx)

RANDOM_SEED = 90210
DEV_SAMPLE_SIZE = 1000

ID_COLUMN = 'SK_ID_CURR'
TARGET_COLUMN = 'TARGET'

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
                     'AMT_GOODS_PRICE',
                     'AMT_INCOME_TOTAL',
                     'AMT_REQ_CREDIT_BUREAU_HOUR',
                     'AMT_REQ_CREDIT_BUREAU_DAY',
                     'AMT_REQ_CREDIT_BUREAU_WEEK',
                     'AMT_REQ_CREDIT_BUREAU_MON',
                     'AMT_REQ_CREDIT_BUREAU_QRT',
                     'AMT_REQ_CREDIT_BUREAU_YEAR',
                     'APARTMENTS_AVG',
                     'APARTMENTS_MEDI',
                     'APARTMENTS_MODE',
                     'BASEMENTAREA_AVG',
                     'BASEMENTAREA_MEDI',
                     'BASEMENTAREA_MODE',
                     'COMMONAREA_AVG',
                     'COMMONAREA_MEDI',
                     'COMMONAREA_MODE',
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
                     'ELEVATORS_MEDI',
                     'ELEVATORS_MODE',
                     'ENTRANCES_AVG',
                     'ENTRANCES_MEDI',
                     'ENTRANCES_MODE',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'EXT_SOURCE_3',
                     'FLOORSMAX_AVG',
                     'FLOORSMAX_MEDI',
                     'FLOORSMAX_MODE',
                     'FLOORSMIN_AVG',
                     'FLOORSMIN_MEDI',
                     'FLOORSMIN_MODE',
                     'LANDAREA_AVG',
                     'LANDAREA_MEDI',
                     'LANDAREA_MODE',
                     'LIVINGAPARTMENTS_AVG',
                     'LIVINGAPARTMENTS_MEDI',
                     'LIVINGAPARTMENTS_MODE',
                     'LIVINGAREA_AVG',
                     'LIVINGAREA_MEDI',
                     'LIVINGAREA_MODE',
                     'NONLIVINGAPARTMENTS_AVG',
                     'NONLIVINGAPARTMENTS_MEDI',
                     'NONLIVINGAPARTMENTS_MODE',
                     'NONLIVINGAREA_AVG',
                     'NONLIVINGAREA_MEDI',
                     'NONLIVINGAREA_MODE',
                     'OBS_30_CNT_SOCIAL_CIRCLE',
                     'OBS_60_CNT_SOCIAL_CIRCLE',
                     'OWN_CAR_AGE',
                     'REGION_POPULATION_RELATIVE',
                     'REGION_RATING_CLIENT',
                     'REGION_RATING_CLIENT_W_CITY',
                     'TOTALAREA_MODE',
                     'YEARS_BEGINEXPLUATATION_AVG',
                     'YEARS_BEGINEXPLUATATION_MEDI',
                     'YEARS_BEGINEXPLUATATION_MODE',
                     'YEARS_BUILD_AVG',
                     'YEARS_BUILD_MEDI',
                     'YEARS_BUILD_MODE']
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

AGGREGATION_RECIPIES = []
for agg in ['mean', 'size', 'var', 'min', 'max']:
    for select in NUMERICAL_COLUMNS:
        for group in [['CODE_GENDER'],
                      ['CODE_GENDER', 'OCCUPATION_TYPE'],
                      ['CODE_GENDER', 'FLAG_OWN_REALTY'],
                      ['CODE_GENDER', 'ORGANIZATION_TYPE'],
                      ['CODE_GENDER', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
                      ['FLAG_OWN_REALTY', 'NAME_HOUSING_TYPE'],
                      ['FLAG_OWN_REALTY', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
                      ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'],
                      ]:
            AGGREGATION_RECIPIES.append({'groupby': group, 'select': select, 'agg': agg})

BUREAU_AGGREGATION_RECIPIES = [{'groupby': ['SK_ID_CURR'], 'select': 'CREDIT_TYPE', 'agg': 'count'},
                               {'groupby': ['SK_ID_CURR'], 'select': 'CREDIT_ACTIVE', 'agg': 'size'}]
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
        BUREAU_AGGREGATION_RECIPIES.append({'groupby': ['SK_ID_CURR'], 'select': select, 'agg': agg})

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
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append({'groupby': ['SK_ID_CURR'], 'select': select, 'agg': agg})

INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_INSTALMENT',
                   'AMT_PAYMENT',
                   'DAYS_ENTRY_PAYMENT',
                   'DAYS_INSTALMENT',
                   'NUM_INSTALMENT_NUMBER',
                   'NUM_INSTALMENT_VERSION'
                   ]:
        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append({'groupby': ['SK_ID_CURR'], 'select': select, 'agg': agg})

POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        POS_CASH_BALANCE_AGGREGATION_RECIPIES.append({'groupby': ['SK_ID_CURR'], 'select': select, 'agg': agg})

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
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append({'groupby': ['SK_ID_CURR'], 'select': select, 'agg': agg})


SOLUTION_CONFIG = AttrDict({
    'pipeline': {'experiment_directory': params.experiment_directory
                 },

    'preprocessing': {'fillna_value': params.fillna_value},

    'dataframe_by_type_splitter': {'numerical_columns': NUMERICAL_COLUMNS,
                                   'categorical_columns': CATEGORICAL_COLUMNS,
                                   'timestamp_columns': [],
                                   },

    'light_gbm': {'device': parameter_eval(params.lgbm__device),
                  'boosting_type': parameter_eval(params.lgbm__boosting_type),
                  'objective': parameter_eval(params.lgbm__objective),
                  'metric': parameter_eval(params.lgbm__metric),
                  'learning_rate': parameter_eval(params.lgbm__learning_rate),
                  'max_depth': parameter_eval(params.lgbm__max_depth),
                  'subsample': parameter_eval(params.lgbm__subsample),
                  'colsample_bytree': parameter_eval(params.lgbm__colsample_bytree),
                  'min_child_weight': parameter_eval(params.lgbm__min_child_weight),
                  'reg_lambda': parameter_eval(params.lgbm__reg_lambda),
                  'reg_alpha': parameter_eval(params.lgbm__reg_alpha),
                  'subsample_freq': parameter_eval(params.lgbm__subsample_freq),
                  'max_bin': parameter_eval(params.lgbm__max_bin),
                  'min_child_samples': parameter_eval(params.lgbm__min_child_samples),
                  'num_leaves': parameter_eval(params.lgbm__num_leaves),
                  'nthread': parameter_eval(params.num_workers),
                  'number_boosting_rounds': parameter_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': parameter_eval(params.lgbm__early_stopping_rounds),
                  'verbose': parameter_eval(params.verbose),
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
                'verbose': parameter_eval(params.verbose)
                },

    'random_forest': {'n_estimators': parameter_eval(params.rf__n_estimators),
                      'criterion': parameter_eval(params.rf__criterion),
                      'max_features': parameter_eval(params.rf__max_features),
                      'min_samples_split': parameter_eval(params.rf__min_samples_split),
                      'min_samples_leaf': parameter_eval(params.rf__min_samples_leaf),
                      'n_jobs': parameter_eval(params.num_workers),
                      'random_state': RANDOM_SEED,
                      'verbose': parameter_eval(params.verbose),
                      'class_weight': parameter_eval(params.rf__class_weight),
                      },

    'logistic_regression': {'penalty': parameter_eval(params.lr__penalty),
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

    'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                    'callbacks':
                                        {'neptune_monitor': {'name': 'light_gbm'},
                                         'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                      'random_search_light_gbm.pkl')}
                                         },
                                    },
                      'xgboost': {'n_runs': params.xgb_random_search_runs,
                                  'callbacks':
                                      {'neptune_monitor': {'name': 'xgboost'},
                                       'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                    'random_search_xgboost.pkl')}
                                       },
                                  },
                      'random_forest': {'n_runs': params.rf_random_search_runs,
                                        'callbacks':
                                            {'neptune_monitor': {'name': 'random_forest'},
                                             'persist_results':
                                                 {'filepath': os.path.join(params.experiment_directory,
                                                                           'random_search_random_forest.pkl')}
                                             },
                                        },
                      'logistic_regression': {'n_runs': params.lr_random_search_runs,
                                              'callbacks':
                                                  {'neptune_monitor': {'name': 'logistic_regression'},
                                                   'persist_results':
                                                       {'filepath': os.path.join(params.experiment_directory,
                                                                                 'random_search_logistic_regression.pkl')}
                                                   },
                                              },
                      'svc': {'n_runs': params.svc_random_search_runs,
                              'callbacks': {'neptune_monitor': {'name': 'svc'},
                                            'persist_results': {'filepath': os.path.join(params.experiment_directory,
                                                                                         'random_search_svc.pkl')}
                                            },
                              },
                      },

    'bureau': {'filename': 'bureau',
               'id_columns': ('SK_ID_CURR', 'SK_ID_CURR'),
               'groupby_aggregations': BUREAU_AGGREGATION_RECIPIES
               },

    'credit_card_balance': {'filename': 'credit_card_balance',
                            'id_columns': ('SK_ID_CURR', 'SK_ID_CURR'),
                            'groupby_aggregations': CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES
                            },

    'installments_payments': {'filename': 'installments_payments',
                              'id_columns': ('SK_ID_CURR', 'SK_ID_CURR'),
                              'groupby_aggregations': INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES
                              },

    'pos_cash_balance': {'filename': 'POS_CASH_balance',
                         'id_columns': ('SK_ID_CURR', 'SK_ID_CURR'),
                         'groupby_aggregations': POS_CASH_BALANCE_AGGREGATION_RECIPIES
                         },

    'previous_applications': {'filename': 'previous_application',
                              'id_columns': ('SK_ID_CURR', 'SK_ID_CURR'),
                              'groupby_aggregations': PREVIOUS_APPLICATION_AGGREGATION_RECIPIES
                              },

    'clipper': {'min_val': 0,
                'max_val': 1
                },

    'groupby_aggregation': {'groupby_aggregations': AGGREGATION_RECIPIES
                            },
})
