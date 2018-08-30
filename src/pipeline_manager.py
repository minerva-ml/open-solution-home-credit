import os
import shutil

from attrdict import AttrDict
import numpy as np
import pandas as pd
from scipy.stats import gmean
from deepsense import neptune
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from . import pipeline_config as cfg
from .pipelines import PIPELINES
from .hyperparameter_tuning import RandomSearchTuner, HyperoptTuner, SkoptTuner, set_params
from .utils import init_logger, read_params, set_seed, create_submission, verify_submission, calculate_rank, \
    read_oof_predictions, parameter_eval

set_seed(cfg.RANDOM_SEED)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx, fallback_file='./configs/neptune.yaml')


class PipelineManager:
    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode):
        evaluate(pipeline_name, dev_mode)

    def predict(self, pipeline_name, dev_mode, submit_predictions):
        predict(pipeline_name, dev_mode, submit_predictions)

    def train_evaluate_cv(self, pipeline_name, model_level, dev_mode):
        train_evaluate_cv(pipeline_name, model_level, dev_mode)

    def train_evaluate_predict_cv(self, pipeline_name, model_level, dev_mode, submit_predictions):
        train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions)


def train(pipeline_name, dev_mode):
    logger.info('TRAINING')
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode)

    logger.info('Shuffling and splitting into train and test...')
    train_data_split, valid_data_split = train_test_split(tables.train_set,
                                                          test_size=params.validation_size,
                                                          random_state=cfg.RANDOM_SEED,
                                                          shuffle=params.shuffle)

    logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Train shape: {}'.format(train_data_split.shape))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    train_data = {'main_table': {'X': train_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                 'y': train_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                 'X_valid': valid_data_split.drop[cfg.TARGET_COLUMNS].values.reshape(-1),
                                 'y_valid': valid_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                 },
                  'application': {'X': tables.application},
                  'bureau_balance': {'X': tables.bureau_balance},
                  'bureau': {'X': tables.bureau},
                  'credit_card_balance': {'X': tables.credit_card_balance},
                  'installments_payments': {'X': tables.installments_payments},
                  'pos_cash_balance': {'X': tables.pos_cash_balance},
                  'previous_application': {'X': tables.previous_application},
                  }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True)
    pipeline.clean_cache()
    logger.info('Start pipeline fit and transform')
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    logger.info('Reading data...')

    tables = _read_data(dev_mode)

    logger.info('Shuffling and splitting to get validation split...')
    _, valid_data_split = train_test_split(tables.train_set,
                                           test_size=params.validation_size,
                                           random_state=cfg.RANDOM_SEED,
                                           shuffle=params.shuffle)

    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    y_true = valid_data_split[cfg.TARGET_COLUMNS].values

    eval_data = {'main_table': {'X': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                'y': None,
                                },
                 'application': {'X': tables.application},
                 'bureau_balance': {'X': tables.bureau_balance},
                 'bureau': {'X': tables.bureau},
                 'credit_card_balance': {'X': tables.credit_card_balance},
                 'installments_payments': {'X': tables.installments_payments},
                 'pos_cash_balance': {'X': tables.pos_cash_balance},
                 'previous_application': {'X': tables.previous_application},
                 }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False)
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(eval_data)
    pipeline.clean_cache()

    y_pred = output['prediction']

    logger.info('Calculating ROC_AUC on validation set')
    score = roc_auc_score(y_true, y_pred)
    logger.info('ROC_AUC score on validation is {}'.format(score))
    ctx.channel_send('ROC_AUC', 0, score)


def predict(pipeline_name, dev_mode, submit_predictions):
    logger.info('PREDICTION')

    tables = _read_data(dev_mode)

    test_data = {'main_table': {'X': tables.test_set,
                                'y': None,
                                },
                 'application': {'X': tables.application},
                 'bureau_balance': {'X': tables.bureau_balance},
                 'bureau': {'X': tables.bureau},
                 'credit_card_balance': {'X': tables.credit_card_balance},
                 'installments_payments': {'X': tables.installments_payments},
                 'pos_cash_balance': {'X': tables.pos_cash_balance},
                 'previous_application': {'X': tables.previous_application},
                 }

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False)

    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(test_data)
    pipeline.clean_cache()
    y_pred = output['prediction']

    if not dev_mode:
        logger.info('creating submission file...')
        submission = create_submission(tables.test_set, y_pred)

        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(submission, sample_submission)

        submission_filepath = os.path.join(params.experiment_directory, 'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('submission persisted to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission.head()))

        if submit_predictions and params.kaggle_api:
            make_submission(submission_filepath)


def train_evaluate_cv(pipeline_name, model_level, dev_mode):
    if parameter_eval(params.hyperparameter_search__method) is not None:
        score_mean, score_std = train_evaluate_cv_tuning(pipeline_name, model_level, dev_mode)
    else:
        score_mean, score_std = train_evaluate_cv_one_run(pipeline_name, model_level, cfg.SOLUTION_CONFIG, dev_mode)

    logger.info('ROC_AUC mean {}, ROC_AUC std {}'.format(score_mean, score_std))
    ctx.channel_send('ROC_AUC', 0, score_mean)
    ctx.channel_send('ROC_AUC STD', 0, score_std)


def train_evaluate_cv_tuning(pipeline_name, model_level, dev_mode):
    config = cfg.SOLUTION_CONFIG
    searchable_config = cfg.SOLUTION_CONFIG.tuner

    if params.hyperparameter_search__method == 'random':
        tuner = RandomSearchTuner(config=searchable_config,
                                  runs=params.hyperparameter_search__runs)
    elif params.hyperparameter_search__method == 'skopt':
        tuner = SkoptTuner(config=searchable_config,
                           runs=params.hyperparameter_search__runs,
                           maximize=True)
    elif params.hyperparameter_search__method == 'hyperopt':
        tuner = HyperoptTuner(config=searchable_config,
                              runs=params.hyperparameter_search__runs,
                              maximize=True)
    else:
        raise NotImplementedError

    results = []
    while tuner.in_progress:
        if tuner.run_id == 0:
            proposed_config = tuner.next(None)
        else:
            proposed_config = tuner.next(score_mean)

        config = set_params(config, proposed_config)

        score_mean, score_std = train_evaluate_cv_one_run(pipeline_name, model_level, config, dev_mode,
                                                          tunable_mode=True)

        logger.info('Run {} ROC_AUC mean {}, ROC_AUC std {}'.format(tuner.run_id, score_mean, score_std))
        ctx.channel_send('Tuning CONFIG', tuner.run_id, proposed_config)
        ctx.channel_send('Tuning ROC_AUC', tuner.run_id, score_mean)
        ctx.channel_send('Tuning ROC_AUC STD', tuner.run_id, score_std)
        results.append((score_mean, score_std, proposed_config))
    best_score_mean, best_score_std, best_config = sorted(results, key=lambda x: x[0])[-1]

    logger.info('ROC_AUC mean {}, ROC_AUC std {}'.format(best_score_mean, best_score_std))
    logger.info('Best Params'.format(best_config))
    ctx.channel_send('BEST_CONFIG', str(best_config))

    return best_score_mean, best_score_std


def train_evaluate_cv_one_run(pipeline_name, model_level, config, dev_mode, tunable_mode=False):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    if model_level == 'first':
        tables = _read_data(dev_mode)
        main_table_train = tables.train_set
    elif model_level == 'second':
        tables = _read_data(dev_mode=False)
        main_table_train, main_table_test = read_oof_predictions(params.first_level_oof_predictions_dir,
                                                                 params.train_filepath,
                                                                 id_column=cfg.ID_COLUMNS[0],
                                                                 target_column=cfg.TARGET_COLUMNS[0])
    else:
        raise NotImplementedError

    target_values = main_table_train[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores = []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        train_data_split, valid_data_split = main_table_train.iloc[train_idx], main_table_train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, _, _ = _fold_fit_evaluate_loop(train_data_split,
                                              valid_data_split,
                                              tables,
                                              fold_id, pipeline_name, config, model_level)

        logger.info('Fold {} ROC_AUC {}'.format(fold_id, score))
        if not tunable_mode:
            ctx.channel_send('Fold {} ROC_AUC'.format(fold_id), 0, score)

        fold_scores.append(score)

    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)
    return score_mean, score_std


def train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    if model_level == 'first':
        tables = _read_data(dev_mode)
        main_table_train = tables.train_set
        main_table_test = tables.test_set
    elif model_level == 'second':
        tables = _read_data(dev_mode=False)
        main_table_train, main_table_test = read_oof_predictions(params.first_level_oof_predictions_dir,
                                                                 params.train_filepath,
                                                                 id_column=cfg.ID_COLUMNS[0],
                                                                 target_column=cfg.TARGET_COLUMNS[0])
        main_table_test = main_table_test.groupby(cfg.ID_COLUMNS).mean().reset_index()
    else:
        raise NotImplementedError

    target_values = main_table_train[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        train_data_split, valid_data_split = main_table_train.iloc[train_idx], main_table_train.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,
                                                                                         valid_data_split,
                                                                                         main_table_test,
                                                                                         tables,
                                                                                         fold_id,
                                                                                         pipeline_name,
                                                                                         model_level)

        logger.info('Fold {} ROC_AUC {}'.format(fold_id, score))
        ctx.channel_send('Fold {} ROC_AUC'.format(fold_id), 0, score)

        out_of_fold_train_predictions.append(out_of_fold_prediction)
        out_of_fold_test_predictions.append(test_prediction)
        fold_scores.append(score)

    out_of_fold_train_predictions = pd.concat(out_of_fold_train_predictions, axis=0)
    out_of_fold_test_predictions = pd.concat(out_of_fold_test_predictions, axis=0)

    test_prediction_aggregated = _aggregate_test_prediction(out_of_fold_test_predictions)
    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('ROC_AUC mean {}, ROC_AUC std {}'.format(score_mean, score_std))
    ctx.channel_send('ROC_AUC', 0, score_mean)
    ctx.channel_send('ROC_AUC STD', 0, score_std)

    logger.info('Saving predictions')
    out_of_fold_train_predictions.to_csv(os.path.join(params.experiment_directory,
                                                      '{}_out_of_fold_train_predictions.csv'.format(pipeline_name)),
                                         index=None)
    out_of_fold_test_predictions.to_csv(os.path.join(params.experiment_directory,
                                                     '{}_out_of_fold_test_predictions.csv'.format(pipeline_name)),
                                        index=None)
    test_aggregated_file_path = os.path.join(params.experiment_directory,
                                             '{}_test_predictions_{}.csv'.format(pipeline_name,
                                                                                 params.aggregation_method))
    test_prediction_aggregated.to_csv(test_aggregated_file_path, index=None)

    if not dev_mode:
        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(test_prediction_aggregated, sample_submission)

        if submit_predictions and params.kaggle_api:
            make_submission(test_aggregated_file_path)


def make_submission(submission_filepath):
    logger.info('making Kaggle submit...')
    os.system('kaggle competitions submit -c home-credit-default-risk -f {} -m {}'
              .format(submission_filepath, params.kaggle_message))


def _read_data(dev_mode):
    logger.info('Reading data...')
    if dev_mode:
        nrows = cfg.DEV_SAMPLE_SIZE
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
    else:
        nrows = None

    if any([parameter_eval(params.use_bureau),
            parameter_eval(params.use_bureau_aggregations)]):
        nrows_bureau = nrows
    else:
        nrows_bureau = 1

    if parameter_eval(params.use_bureau_balance):
        nrows_bureau_balance = nrows
    else:
        nrows_bureau_balance = 1

    if any([parameter_eval(params.use_credit_card_balance),
            parameter_eval(params.use_credit_card_balance_aggregations)]):
        nrows_credit_card_balance = nrows
    else:
        nrows_credit_card_balance = 1

    if any([parameter_eval(params.use_installments_payments),
            parameter_eval(params.use_installments_payments_aggregations)]):
        nrows_installments_payments = nrows
    else:
        nrows_installments_payments = 1

    if any([parameter_eval(params.use_pos_cash_balance),
            parameter_eval(params.use_pos_cash_balance_aggregations)]):
        nrows_pos_cash_balance = nrows
    else:
        nrows_pos_cash_balance = 1

    if any([parameter_eval(params.use_previous_applications),
            parameter_eval(params.use_previous_applications_aggregations),
            parameter_eval(params.use_previous_application_categorical_features),
            parameter_eval(params.use_application_previous_application_categorical_features)]):
        nrows_previous_applications = nrows
    else:
        nrows_previous_applications = 1

    raw_data = {}

    logger.info('Reading application_train ...')
    application_train = pd.read_csv(params.train_filepath, nrows=nrows)
    logger.info("Reading application_test ...")
    application_test = pd.read_csv(params.test_filepath, nrows=nrows)
    raw_data['application'] = pd.concat([application_train, application_test],
                                        sort=False).drop(cfg.TARGET_COLUMNS, axis='columns')
    raw_data['train_set'] = pd.DataFrame(application_train[cfg.ID_COLUMNS + cfg.TARGET_COLUMNS])
    raw_data['test_set'] = pd.DataFrame(application_test[cfg.ID_COLUMNS])

    logger.info("Reading bureau ...")
    raw_data['bureau'] = pd.read_csv(params.bureau_filepath, nrows=nrows_bureau)
    logger.info("Reading credit_card_balance ...")
    raw_data['credit_card_balance'] = pd.read_csv(params.credit_card_balance_filepath, nrows=nrows_credit_card_balance)
    logger.info("Reading pos_cash_balance ...")
    raw_data['pos_cash_balance'] = pd.read_csv(params.POS_CASH_balance_filepath, nrows=nrows_pos_cash_balance)
    logger.info("Reading previous_application ...")
    raw_data['previous_application'] = pd.read_csv(params.previous_application_filepath,
                                                   nrows=nrows_previous_applications)
    logger.info("Reading bureau_balance ...")
    raw_data['bureau_balance'] = pd.read_csv(params.bureau_balance_filepath, nrows=nrows_bureau_balance)
    raw_data['bureau_balance'] = raw_data['bureau_balance'].merge(raw_data['bureau'][['SK_ID_CURR', 'SK_ID_BUREAU']],
                                                                  on='SK_ID_BUREAU', how='right')
    logger.info("Reading installments_payments ...")
    raw_data['installments_payments'] = pd.read_csv(params.installments_payments_filepath,
                                                    nrows=nrows_installments_payments)
    logger.info("Reading Done!!")
    return AttrDict(raw_data)


def _get_fold_generator(target_values):
    if params.stratified_cv:
        cv = StratifiedKFold(n_splits=params.n_cv_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        cv.get_n_splits(target_values)
        fold_generator = cv.split(target_values, target_values)
    else:
        cv = KFold(n_splits=params.n_cv_splits, shuffle=True, random_state=cfg.RANDOM_SEED)
        fold_generator = cv.split(target_values)
    return fold_generator


def _fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, test, tables, fold_id, pipeline_name,
                                    model_level):
    score, y_valid_pred, pipeline = _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables,
                                                            fold_id, pipeline_name, cfg.SOLUTION_CONFIG, model_level)
    if model_level == 'first':
        test_data = {'main_table': {'X': test[cfg.ID_COLUMNS],
                                    'y': None,
                                    },
                     'application': {'X': tables.application},
                     'bureau_balance': {'X': tables.bureau_balance},
                     'bureau': {'X': tables.bureau},
                     'credit_card_balance': {'X': tables.credit_card_balance},
                     'installments_payments': {'X': tables.installments_payments},
                     'pos_cash_balance': {'X': tables.pos_cash_balance},
                     'previous_application': {'X': tables.previous_application},
                     }
    elif model_level == 'second':

        test_data = {'main_table': {'X': test[cfg.ID_COLUMNS],
                                    'y': None,
                                    },
                     'oof_predictions': {'X': test.drop('fold_id', axis=1),
                                         },
                     'application': {'X': tables.application},
                     'bureau_balance': {'X': tables.bureau_balance},
                     'bureau': {'X': tables.bureau},
                     'credit_card_balance': {'X': tables.credit_card_balance},
                     'installments_payments': {'X': tables.installments_payments},
                     'pos_cash_balance': {'X': tables.pos_cash_balance},
                     'previous_application': {'X': tables.previous_application},
                     }
    else:
        raise NotImplementedError

    logger.info('Start pipeline transform on test')
    pipeline.clean_cache()
    output_test = pipeline.transform(test_data)
    pipeline.clean_cache()
    y_test_pred = output_test.get('prediction', output_test.get('predicted', None))

    train_out_of_fold_prediction_chunk = valid_data_split[cfg.ID_COLUMNS]
    train_out_of_fold_prediction_chunk['fold_id'] = fold_id
    train_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_valid_pred

    test_out_of_fold_prediction_chunk = test[cfg.ID_COLUMNS]
    test_out_of_fold_prediction_chunk['fold_id'] = fold_id
    test_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_test_pred

    return score, train_out_of_fold_prediction_chunk, test_out_of_fold_prediction_chunk


def _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name, config, model_level):
    if model_level == 'first':
        train_data = {'main_table': {'X': train_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                     'y': train_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                     'X_valid': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                     'y_valid': valid_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                     },
                      'application': {'X': tables.application},
                      'bureau_balance': {'X': tables.bureau_balance},
                      'bureau': {'X': tables.bureau},
                      'credit_card_balance': {'X': tables.credit_card_balance},
                      'installments_payments': {'X': tables.installments_payments},
                      'pos_cash_balance': {'X': tables.pos_cash_balance},
                      'previous_application': {'X': tables.previous_application},
                      }

        valid_data = {'main_table': {'X': valid_data_split.drop(cfg.TARGET_COLUMNS, axis=1),
                                     'y': None,
                                     },
                      'application': {'X': tables.application},
                      'bureau_balance': {'X': tables.bureau_balance},
                      'bureau': {'X': tables.bureau},
                      'credit_card_balance': {'X': tables.credit_card_balance},
                      'installments_payments': {'X': tables.installments_payments},
                      'pos_cash_balance': {'X': tables.pos_cash_balance},
                      'previous_application': {'X': tables.previous_application},
                      }
    elif model_level == 'second':
        drop_columns = cfg.TARGET_COLUMNS + ['fold_id']
        train_data = {'main_table': {'X': train_data_split[cfg.ID_COLUMNS],
                                     'y': train_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                     'X_valid': valid_data_split[cfg.ID_COLUMNS],
                                     'y_valid': valid_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                     },
                      'oof_predictions': {'X': train_data_split.drop(drop_columns, axis=1),
                                          'X_valid': valid_data_split.drop(drop_columns, axis=1),
                                          },
                      'application': {'X': tables.application},
                      'bureau_balance': {'X': tables.bureau_balance},
                      'bureau': {'X': tables.bureau},
                      'credit_card_balance': {'X': tables.credit_card_balance},
                      'installments_payments': {'X': tables.installments_payments},
                      'pos_cash_balance': {'X': tables.pos_cash_balance},
                      'previous_application': {'X': tables.previous_application},
                      }
        valid_data = {'main_table': {'X': valid_data_split[cfg.ID_COLUMNS],
                                     'y': None,
                                     },
                      'oof_predictions': {'X': valid_data_split.drop(drop_columns, axis=1),
                                          },
                      'application': {'X': tables.application},
                      'bureau_balance': {'X': tables.bureau_balance},
                      'bureau': {'X': tables.bureau},
                      'credit_card_balance': {'X': tables.credit_card_balance},
                      'installments_payments': {'X': tables.installments_payments},
                      'pos_cash_balance': {'X': tables.pos_cash_balance},
                      'previous_application': {'X': tables.previous_application},
                      }
    else:
        raise NotImplementedError

    pipeline = PIPELINES[pipeline_name](config=config, train_mode=True,
                                        suffix='_fold_{}'.format(fold_id))

    logger.info('Start pipeline fit and transform on train')
    pipeline.clean_cache()
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()

    pipeline = PIPELINES[pipeline_name](config=config, train_mode=False,
                                        suffix='_fold_{}'.format(fold_id))
    logger.info('Start pipeline transform on valid')
    pipeline.clean_cache()
    output_valid = pipeline.transform(valid_data)
    pipeline.clean_cache()

    y_valid_pred = output_valid.get('prediction', output_valid.get('predicted', None))
    y_valid_true = valid_data_split[cfg.TARGET_COLUMNS].values
    score = roc_auc_score(y_valid_true, y_valid_pred)

    return score, y_valid_pred, pipeline


def _aggregate_test_prediction(out_of_fold_test_predictions):
    agg_methods = {'mean': np.mean,
                   'gmean': gmean}
    prediction_column = [col for col in out_of_fold_test_predictions.columns if '_prediction' in col][0]
    if params.aggregation_method == 'rank_mean':
        rank_column = prediction_column.replace('_prediction', '_rank')
        test_predictions_with_ranks = []
        for fold_id, fold_df in out_of_fold_test_predictions.groupby('fold_id'):
            fold_df[rank_column] = calculate_rank(fold_df[prediction_column])
            test_predictions_with_ranks.append(fold_df)
        test_predictions_with_ranks = pd.concat(test_predictions_with_ranks, axis=0)

        test_prediction_aggregated = test_predictions_with_ranks.groupby(cfg.ID_COLUMNS)[rank_column].apply(
            np.mean).reset_index()
    else:
        test_prediction_aggregated = out_of_fold_test_predictions.groupby(cfg.ID_COLUMNS)[prediction_column].apply(
            agg_methods[params.aggregation_method]).reset_index()

    test_prediction_aggregated.columns = [cfg.ID_COLUMNS + cfg.TARGET_COLUMNS]
    return test_prediction_aggregated
