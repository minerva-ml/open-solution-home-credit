import os

from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

CATEGORICAL_COLUMNS = ['CODE_GENDER',
                       'EMERGENCYSTATE_MODE',
                       'FLAG_MOBIL',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'FONDKAPREMONT_MODE',
                       'HOUSETYPE_MODE',
                       'NAME_CONTRACT_TYPE',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'ORGANIZATION_TYPE',
                       'WALLSMATERIAL_MODE',
                       'WEEKDAY_APPR_PROCESS_START']
NUMERICAL_COLUMNS = ['AMT_ANNUITY',
                     'AMT_CREDIT',
                     'AMT_GOODS_PRICE',
                     'AMT_INCOME_TOTAL',
                     'CNT_CHILDREN',
                     'DAYS_BIRTH',
                     'DAYS_EMPLOYED',
                     'DAYS_ID_PUBLISH',
                     'DAYS_REGISTRATION',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'EXT_SOURCE_3',
                     'OWN_CAR_AGE',
                     'REGION_POPULATION_RELATIVE',
                     'REGION_RATING_CLIENT',
                     'REGION_RATING_CLIENT_W_CITY']
TIMESTAMP_COLUMNS = []
ID_COLUMNS = ['SK_ID_CURR']
TARGET_COLUMNS = ['TARGET']

DEV_SAMPLE_SIZE = int(20e4)

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir
            },

    'dataframe_by_type_splitter': {'numerical_columns': NUMERICAL_COLUMNS,
                                   'categorical_columns': CATEGORICAL_COLUMNS,
                                   'timestamp_columns': TIMESTAMP_COLUMNS,
                                   },

    'light_gbm': {'boosting_type': params.lgbm__boosting_type,
                  'objective': params.lgbm__objective,
                  'metric': params.lgbm__metric,
                  'learning_rate': params.lgbm__learning_rate,
                  'max_depth': params.lgbm__max_depth,
                  'subsample': params.lgbm__subsample,
                  'colsample_bytree': params.lgbm__colsample_bytree,
                  'min_child_weight': params.lgbm__min_child_weight,
                  'reg_lambda': params.lgbm__reg_lambda,
                  'reg_alpha': params.lgbm__reg_alpha,
                  'subsample_freq': params.lgbm__subsample_freq,
                  'max_bin': params.lgbm__max_bin,
                  'min_child_samples': params.lgbm__min_child_samples,
                  'num_leaves': params.lgbm__num_leaves,
                  'nthread': params.num_workers,
                  'number_boosting_rounds': params.lgbm__number_boosting_rounds,
                  'early_stopping_rounds': params.lgbm__early_stopping_rounds,
                  'verbose': params.verbose
                  },

    'random_search': {'light_gbm': {'n_runs': params.lgbm_random_search_runs,
                                    'callbacks': {'neptune_monitor': {'name': 'light_gbm'
                                                                      },
                                                  'save_results': {'filepath': os.path.join(params.experiment_dir,
                                                                                            'random_search_light_gbm.pkl')
                                                                   }
                                                  }
                                    }
},

    'clipper': {'min_val': 0,
                'max_val': 1
                }
})

