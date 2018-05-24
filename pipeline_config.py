from attrdict import AttrDict
from deepsense import neptune

from utils import read_params

ctx = neptune.Context()
params = read_params(ctx)

# FEATURE_COLUMNS = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
CATEGORICAL_COLUMNS = ['CODE_GENDER',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'NAME_CONTRACT_TYPE',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE',
                       'FONDKAPREMONT_MODE',
                       'HOUSETYPE_MODE',
                       'WALLSMATERIAL_MODE',
                       'EMERGENCYSTATE_MODE']
NUMERICAL_COLUMNS = ['AMT_CREDIT',
                     'AMT_GOODS_PRICE',
                     'DAYS_BIRTH',
                     'DAYS_EMPLOYED',
                     'DAYS_ID_PUBLISH',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'EXT_SOURCE_3',
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

    'clipper': {'min_val': 0,
                'max_val': 1
                }
})

