import os
import shutil

import click
import pandas as pd
from deepsense import neptune
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import pipeline_config as cfg
from pipelines import PIPELINES
from utils import create_submission, init_logger, read_params, persist_evaluation_predictions, \
    set_seed, verify_submission

set_seed()
logger = init_logger()
ctx = neptune.Context()
params = read_params(ctx)


@click.group()
def action():
    pass


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, dev_mode):
    _train(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, dev_mode):
    _evaluate(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def predict(pipeline_name, dev_mode):
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict(pipeline_name, dev_mode):
    _train(pipeline_name, dev_mode)
    _evaluate(pipeline_name, dev_mode)
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate_predict(pipeline_name, dev_mode):
    _evaluate(pipeline_name, dev_mode)
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    _train(pipeline_name, dev_mode)
    _evaluate(pipeline_name, dev_mode)


def _train(pipeline_name, dev_mode):
    logger.info('TRAINING')
    if bool(params.clean_experiment_directory_before_training) and os.path.isdir(params.experiment_directory):
        logger.info('Cleaning experiment_directory...')
        shutil.rmtree(params.experiment_directory)

    logger.info('Reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        application_train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        bureau_balance = pd.read_csv(params.bureau_balance_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        bureau = pd.read_csv(params.bureau_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        credit_card_balance = pd.read_csv(params.credit_card_balance_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        installments_payments = pd.read_csv(params.installments_payments_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        pos_cash_balance = pd.read_csv(params.POS_CASH_balance_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
        previous_application = pd.read_csv(params.previous_application_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        application_train = pd.read_csv(params.train_filepath)
        bureau_balance = pd.read_csv(params.bureau_balance_filepath)
        bureau = pd.read_csv(params.bureau_filepath)
        credit_card_balance = pd.read_csv(params.credit_card_balance_filepath)
        installments_payments = pd.read_csv(params.installments_payments_filepath)
        pos_cash_balance = pd.read_csv(params.POS_CASH_balance_filepath)
        previous_application = pd.read_csv(params.previous_application_filepath)

    logger.info('Shuffling and splitting into train and test...')
    train_data_split, valid_data_split = train_test_split(application_train,
                                                          test_size=params.validation_size,
                                                          random_state=cfg.RANDOM_SEED,
                                                          shuffle=params.shuffle)

    logger.info('Target mean in train: {}'.format(train_data_split[cfg.TARGET_COLUMN].mean()))
    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMN].mean()))
    logger.info('Train shape: {}'.format(train_data_split.shape))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    data = {'input': {'X': train_data_split.drop(cfg.TARGET_COLUMN, axis=1),
                      'y': train_data_split[cfg.TARGET_COLUMN],
                      'X_valid': valid_data_split.drop(cfg.TARGET_COLUMN, axis=1),
                      'y_valid': valid_data_split[cfg.TARGET_COLUMN],
                      'bureau_balance': bureau_balance,
                      'bureau': bureau,
                      'credit_card_balance': credit_card_balance,
                      'installments_payments': installments_payments,
                      'pos_cash_balance': pos_cash_balance,
                      'previous_application': previous_application
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    logger.info('Start pipeline fit and transform')
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def _evaluate(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    logger.info('reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        application_train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        application_train = pd.read_csv(params.train_filepath)

    logger.info('Shuffling and splitting to get validation split...')
    _, valid_data_split = train_test_split(application_train,
                                           test_size=params.validation_size,
                                           random_state=cfg.RANDOM_SEED,
                                           shuffle=params.shuffle)

    logger.info('Target mean in valid: {}'.format(valid_data_split[cfg.TARGET_COLUMN].mean()))
    logger.info('Valid shape: {}'.format(valid_data_split.shape))

    y_true = valid_data_split[cfg.TARGET_COLUMN].values
    data = {'input': {'X': valid_data_split.drop(cfg.TARGET_COLUMN, axis=1),
                      'y': valid_data_split[cfg.TARGET_COLUMN],
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(data)
    pipeline.clean_cache()

    y_pred = output['clipped_prediction']

    logger.info('Saving evaluation predictions to the {}'.format(params.experiment_directory))
    persist_evaluation_predictions(params.experiment_directory,
                                   y_pred,
                                   valid_data_split,
                                   cfg.ID_COLUMN,
                                   cfg.TARGET_COLUMN)

    logger.info('Calculating ROC_AUC on validation set')
    score = roc_auc_score(y_true, y_pred)
    logger.info('ROC_AUC score on validation is {}'.format(score))
    ctx.channel_send('ROC_AUC', 0, score)


def _predict(pipeline_name, dev_mode):
    logger.info('PREDICTION')
    logger.info('reading data...')
    if dev_mode:
        logger.info('running in "dev-mode". Sample size is: {}'.format(cfg.DEV_SAMPLE_SIZE))
        application_test = pd.read_csv(params.test_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        application_test = pd.read_csv(params.test_filepath)

    data = {'input': {'X': application_test,
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    logger.info('Start pipeline transform')
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['clipped_prediction']

    if not dev_mode:
        logger.info('creating submission file...')
        submission = create_submission(application_test, y_pred)

        logger.info('verifying submission...')
        sample_submission = pd.read_csv(params.sample_submission_filepath)
        verify_submission(submission, sample_submission)

        submission_filepath = os.path.join(params.experiment_directory, 'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('submission persisted to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission.head()))

        if params.kaggle_api:
            logger.info('making Kaggle submit...')
            os.system('kaggle competitions submit -c home-credit-default-risk -f {} -m {}'
                      .format(submission_filepath, params.kaggle_message))


if __name__ == "__main__":
    action()
