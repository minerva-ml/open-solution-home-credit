import os
import shutil

import click
import pandas as pd
from deepsense import neptune
from sklearn.metrics import roc_auc_score

import pipeline_config as cfg
from pipelines import PIPELINES
from utils import create_submission, init_logger, read_params, save_evaluation_predictions, \
    set_seed, stratified_train_valid_split, verify_submission

set_seed(1234)
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


def _train(pipeline_name, dev_mode):
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    logger.info('reading data in')
    if dev_mode:
        meta_train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_train = pd.read_csv(params.train_filepath)

    meta_train_split, meta_valid_split = stratified_train_valid_split(meta_train,
                                                                      target_column=cfg.TARGET_COLUMNS,
                                                                      target_bins=params.target_bins,
                                                                      valid_size=params.validation_size,
                                                                      random_state=1234)

    logger.info('Target distribution in train: {}'.format(meta_train_split[cfg.TARGET_COLUMNS].mean()))
    logger.info('Target distribution in valid: {}'.format(meta_valid_split[cfg.TARGET_COLUMNS].mean()))

    logger.info('shuffling data')
    meta_train_split = meta_train_split.sample(frac=1)
    meta_valid_split = meta_valid_split.sample(frac=1)

    data = {'input': {'X': meta_train_split.drop(cfg.TARGET_COLUMNS, axis=1),
                      'y': meta_train_split[cfg.TARGET_COLUMNS],
                      'X_valid': meta_valid_split.drop(cfg.TARGET_COLUMNS, axis=1),
                      'y_valid': meta_valid_split[cfg.TARGET_COLUMNS],
                      },
            }

    pipeline = PIPELINES[pipeline_name]['train'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, dev_mode):
    _evaluate(pipeline_name, dev_mode)


def _evaluate(pipeline_name, dev_mode):
    logger.info('reading data in')
    if dev_mode:
        meta_train = pd.read_csv(params.train_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_train = pd.read_csv(params.train_filepath)

    _, meta_valid_split = stratified_train_valid_split(meta_train,
                                                       target_column=cfg.TARGET_COLUMNS,
                                                       target_bins=params.target_bins,
                                                       valid_size=params.validation_size,
                                                       random_state=1234)

    logger.info('Target distribution in valid: {}'.format(meta_valid_split[cfg.TARGET_COLUMNS].mean()))

    data = {'input': {'X': meta_valid_split,
                      'y': None,
                      },
            }
    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['clipped_prediction']
    y_true = meta_valid_split[cfg.TARGET_COLUMNS].values.reshape(-1)

    logger.info('Saving evaluation predictions')
    save_evaluation_predictions(params.experiment_dir, y_true, y_pred, meta_valid_split)

    logger.info('Calculating ROC_AUC Full Scores')
    score = roc_auc_score(y_true, y_pred)
    logger.info('ROC_AUC score on validation is {}'.format(score))
    ctx.channel_send('ROC_AUC', 0, score)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def predict(pipeline_name, dev_mode):
    _predict(pipeline_name, dev_mode)


def _predict(pipeline_name, dev_mode):
    logger.info('reading data in')
    if dev_mode:
        meta_test = pd.read_csv(params.test_filepath, nrows=cfg.DEV_SAMPLE_SIZE)
    else:
        meta_test = pd.read_csv(params.test_filepath)

    data = {'input': {'X': meta_test,
                      'y': None,
                      },
            }

    pipeline = PIPELINES[pipeline_name]['inference'](cfg.SOLUTION_CONFIG)
    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['clipped_prediction']

    logger.info('creating submission...')
    submission = create_submission(meta_test, y_pred)

    logger.info('verifying submittion')
    sample_submission = pd.read_csv(params.sample_submission_filepath)
    verify_submission(submission, sample_submission)

    if dev_mode:
        logger.info('submittion can\'t be saved in dev mode')
    else:
        submission_filepath = os.path.join(params.experiment_dir, 'submission.csv')
        submission.to_csv(submission_filepath, index=None, encoding='utf-8')
        logger.info('submission saved to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission.head()))


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict(pipeline_name, dev_mode):
    logger.info('TRAINING')
    _train(pipeline_name, dev_mode)
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)
    logger.info('PREDICTION')
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate_predict(pipeline_name, dev_mode):
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)
    logger.info('PREDICTION')
    _predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, dev_mode):
    logger.info('TRAINING')
    _train(pipeline_name, dev_mode)
    logger.info('EVALUATION')
    _evaluate(pipeline_name, dev_mode)


if __name__ == "__main__":
    action()
