import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params, safe_eval

ctx = neptune.Context()
params = read_params(ctx)

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
TIMESTAMP_COLUMNS = []
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

ID_COLUMNS = ['SK_ID_CURR']
TARGET_COLUMNS = ['TARGET']

DEV_SAMPLE_SIZE = int(10e4)

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir
            },

    'dataframe_by_type_splitter': {'numerical_columns': NUMERICAL_COLUMNS,
                                   'categorical_columns': CATEGORICAL_COLUMNS,
                                   'timestamp_columns': TIMESTAMP_COLUMNS,
                                   },

    'light_gbm': {'boosting_type': safe_eval(params.lgbm__boosting_type),
                  'objective': safe_eval(params.lgbm__objective),
                  'metric': safe_eval(params.lgbm__metric),
                  'learning_rate': safe_eval(params.lgbm__learning_rate),
                  'max_depth': safe_eval(params.lgbm__max_depth),
                  'subsample': safe_eval(params.lgbm__subsample),
                  'colsample_bytree': safe_eval(params.lgbm__colsample_bytree),
                  'min_child_weight': safe_eval(params.lgbm__min_child_weight),
                  'reg_lambda': safe_eval(params.lgbm__reg_lambda),
                  'reg_alpha': safe_eval(params.lgbm__reg_alpha),
                  'subsample_freq': safe_eval(params.lgbm__subsample_freq),
                  'max_bin': safe_eval(params.lgbm__max_bin),
                  'min_child_samples': safe_eval(params.lgbm__min_child_samples),
                  'num_leaves': safe_eval(params.lgbm__num_leaves),
                  'nthread': safe_eval(params.num_workers),
                  'number_boosting_rounds': safe_eval(params.lgbm__number_boosting_rounds),
                  'early_stopping_rounds': safe_eval(params.lgbm__early_stopping_rounds),
                  'verbose': safe_eval(params.verbose)
                  },

    'random_forest': {'n_estimators': safe_eval(params.rf__n_estimators),
                      'max_features': safe_eval(params.rf__max_features),
                      'max_depth': safe_eval(params.rf__max_depth),
                      'min_samples_split': safe_eval(params.rf__min_samples_split),
                      'min_samples_leaf': safe_eval(params.rf__min_samples_leaf),
                      'min_weight_fraction_leaf': safe_eval(params.rf__min_weight_fraction_leaf),
                      'n_jobs': safe_eval(params.num_workers),
                      'verbose': safe_eval(params.verbose)
                      },

    'logistic_regression': {'penalty': safe_eval(params.lr__penalty),
                            'tol': safe_eval(params.lr__tol),
                            'C': safe_eval(params.lr__C),
                            'solver': safe_eval(params.lr__solver),
                            'n_jobs': safe_eval(params.num_workers),
                            'verbose': safe_eval(params.verbose),
                            },

    'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                    'callbacks': {'neptune_monitor': {'name': 'light_gbm'
                                                                      },
                                                  'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                            'random_search_light_gbm.pkl')
                                                                   }
                                                  }
                                    },
                      'random_forest': {'n_runs': params.rf_random_search_runs,
                                        'callbacks': {'neptune_monitor': {'name': 'random_forest'
                                                                          },
                                                      'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                                'random_search_random_forest.pkl')
                                                                       }
                                                      }
                                        },
                      'logistic_regression': {'n_runs': params.lr_random_search_runs,
                                              'callbacks': {'neptune_monitor': {'name': 'logistic_regression'
                                                                                },
                                                            'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                                'random_search_logistic_regression.pkl')
                                                                            }
                                                            }
                                              }

                      },

    'clipper': {'min_val': 0,
                'max_val': 1
                }
})
