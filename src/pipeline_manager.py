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
from .utils import init_logger, read_params, set_seed, create_submission, verify_submission, calculate_rank, \
    read_oof_predictions

set_seed(cfg.RANDOM_SEED)
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx, fallback_file='../configs/neptune.yaml')


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

    tables = _read_data(dev_mode, read_train=True, read_test=False)

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

    tables = _read_data(dev_mode, read_train=True, read_test=False)

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

    tables = _read_data(dev_mode, read_train=False, read_test=True)

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
    if model_level == 'first':
        train_evaluate_cv_first_level(pipeline_name, dev_mode)
    elif model_level == 'second':
        train_evaluate_cv_second_level(pipeline_name)
    else:
        raise NotImplementedError


def train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions):
    if model_level == 'first':
        train_evaluate_predict_cv_first_level(pipeline_name, dev_mode, submit_predictions)
    elif model_level == 'second':
        train_evaluate_predict_cv_second_level(pipeline_name, submit_predictions)
    else:
        raise NotImplementedError


def make_submission(submission_filepath):
    logger.info('making Kaggle submit...')
    os.system('kaggle competitions submit -c home-credit-default-risk -f {} -m {}'
              .format(submission_filepath, params.kaggle_message))


def train_evaluate_cv_first_level(pipeline_name, dev_mode):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode, read_train=True, read_test=False)

    target_values = tables.train_set[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores = []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        (train_data_split,
         valid_data_split) = tables.train_set.iloc[train_idx], tables.train_set.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, _, _ = _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name,
                                              model_level='first')

        logger.info('Fold {} ROC_AUC {}'.format(fold_id, score))
        ctx.channel_send('Fold {} ROC_AUC'.format(fold_id), 0, score)

        fold_scores.append(score)

    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('ROC_AUC mean {}, ROC_AUC std {}'.format(score_mean, score_std))
    ctx.channel_send('ROC_AUC', 0, score_mean)
    ctx.channel_send('ROC_AUC STD', 0, score_std)


def train_evaluate_cv_second_level(pipeline_name):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    logger.info('Reading data...')

    train, test = read_oof_predictions(params.first_level_oof_predictions_dir,
                                       params.train_filepath,
                                       id_column=cfg.ID_COLUMNS[0],
                                       target_column=cfg.TARGET_COLUMNS[0])

    fold_scores = []
    for fold_id in range(params.n_cv_splits):
        train_data_split = train[train['fold_id'] != fold_id]
        valid_data_split = train[train['fold_id'] == fold_id]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, _, _ = _fold_fit_evaluate_loop(train_data_split, valid_data_split, None, fold_id, pipeline_name,
                                              model_level='second')

        logger.info('Fold {} ROC_AUC {}'.format(fold_id, score))
        ctx.channel_send('Fold {} ROC_AUC'.format(fold_id), 0, score)

        fold_scores.append(score)

    score_mean, score_std = np.mean(fold_scores), np.std(fold_scores)

    logger.info('ROC_AUC mean {}, ROC_AUC std {}'.format(score_mean, score_std))
    ctx.channel_send('ROC_AUC', 0, score_mean)
    ctx.channel_send('ROC_AUC STD', 0, score_std)


def train_evaluate_predict_cv_first_level(pipeline_name, dev_mode, submit_predictions):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    tables = _read_data(dev_mode, read_train=True, read_test=True)

    target_values = tables.train_set[cfg.TARGET_COLUMNS].values.reshape(-1)
    fold_generator = _get_fold_generator(target_values)

    fold_scores, out_of_fold_train_predictions, out_of_fold_test_predictions = [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(fold_generator):
        (train_data_split,
         valid_data_split) = tables.train_set.iloc[train_idx], tables.train_set.iloc[valid_idx]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,
                                                                                         valid_data_split,
                                                                                         tables,
                                                                                         fold_id, pipeline_name,
                                                                                         model_level='first')

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


def train_evaluate_predict_cv_second_level(pipeline_name, submit_predictions):
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    train, test = read_oof_predictions(params.first_level_oof_predictions_dir,
                                       params.train_filepath,
                                       id_column=cfg.ID_COLUMNS[0],
                                       target_column=cfg.TARGET_COLUMNS[0])

    out_of_fold_train_predictions, out_of_fold_test_predictions, fold_scores = [], [], []
    for fold_id in range(params.n_cv_splits):
        train_data_split = train[train['fold_id'] != fold_id]
        valid_data_split = train[train['fold_id'] == fold_id]

        logger.info('Started fold {}'.format(fold_id))
        logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMNS].mean()))
        logger.info('Train shape: {}'.format(train_data_split.shape))
        logger.info('Valid shape: {}'.format(valid_data_split.shape))

        score, out_of_fold_prediction, test_prediction = _fold_fit_evaluate_predict_loop(train_data_split,
                                                                                         valid_data_split,
                                                                                         test,
                                                                                         fold_id, pipeline_name,
                                                                                         model_level='second')

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

    logger.info('verifying submission...')
    sample_submission = pd.read_csv(params.sample_submission_filepath)
    verify_submission(test_prediction_aggregated, sample_submission)

    if submit_predictions and params.kaggle_api:
        make_submission(test_aggregated_file_path)


def _read_data(dev_mode, read_train=True, read_test=False):
    logger.info('Reading data...')
    if dev_mode:
        nrows = cfg.DEV_SAMPLE_SIZE
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
    else:
        nrows = None

    raw_data = {}

    application_train = pd.read_csv(params.train_filepath, nrows=nrows)
    application_test = pd.read_csv(params.test_filepath, nrows=nrows)
    raw_data['application'] = pd.concat([application_train, application_test],
                                        sort=False).drop(cfg.TARGET_COLUMNS, axis='columns')
    if read_train:
        raw_data['train_set'] = pd.DataFrame(application_train[cfg.ID_COLUMNS + cfg.TARGET_COLUMNS])
    if read_test:
        raw_data['test_set'] = pd.DataFrame(application_test[cfg.ID_COLUMNS])
    raw_data['bureau'] = pd.read_csv(params.bureau_filepath, nrows=nrows)
    raw_data['credit_card_balance'] = pd.read_csv(params.credit_card_balance_filepath, nrows=nrows)
    raw_data['pos_cash_balance'] = pd.read_csv(params.POS_CASH_balance_filepath, nrows=nrows)
    raw_data['previous_application'] = pd.read_csv(params.previous_application_filepath, nrows=nrows)
    raw_data['bureau_balance'] = pd.read_csv(params.bureau_balance_filepath, nrows=nrows)
    raw_data['bureau_balance'] = raw_data['bureau_balance'].merge(raw_data['bureau'][['SK_ID_CURR', 'SK_ID_BUREAU']],
                                                                  on='SK_ID_BUREAU', how='right')
    raw_data['installments_payments'] = pd.read_csv(params.installments_payments_filepath, nrows=nrows)

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


def _fold_fit_evaluate_predict_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name, model_level):
    score, y_valid_pred, pipeline = _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables,
                                                            fold_id, pipeline_name, model_level)
    if model_level == 'first':
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
    elif model_level == 'second':
        test_data = {'input': {'X': tables.drop(cfg.ID_COLUMNS, axis=1),
                               'y': None,
                               },
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

    if model_level == 'first':
        test_out_of_fold_prediction_chunk = tables.test_set[cfg.ID_COLUMNS]
    elif model_level == 'second':
        test_out_of_fold_prediction_chunk = tables[cfg.ID_COLUMNS]
    else:
        raise NotImplementedError

    test_out_of_fold_prediction_chunk['fold_id'] = fold_id
    test_out_of_fold_prediction_chunk['{}_prediction'.format(pipeline_name)] = y_test_pred

    return score, train_out_of_fold_prediction_chunk, test_out_of_fold_prediction_chunk


def _fold_fit_evaluate_loop(train_data_split, valid_data_split, tables, fold_id, pipeline_name, model_level):
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
        train_data = {'input': {'X': train_data_split.drop(drop_columns, axis=1),
                                'y': train_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                'X_valid': valid_data_split.drop(drop_columns, axis=1),
                                'y_valid': valid_data_split[cfg.TARGET_COLUMNS].values.reshape(-1),
                                },
                      }

        valid_data = {'input': {'X': valid_data_split.drop(drop_columns, axis=1),
                                'y': None,
                                },
                      }
    else:
        raise NotImplementedError

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=True,
                                        suffix='_fold_{}'.format(fold_id))

    logger.info('Start pipeline fit and transform on train')
    pipeline.clean_cache()
    pipeline.fit_transform(train_data)
    pipeline.clean_cache()

    pipeline = PIPELINES[pipeline_name](config=cfg.SOLUTION_CONFIG, train_mode=False,
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
