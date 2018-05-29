import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
from sklearn.model_selection import train_test_split


def create_submission(meta, predictions):
    submission = pd.DataFrame({'SK_ID_CURR': meta['SK_ID_CURR'].tolist(),
                               'TARGET': predictions
                               })
    return submission


def verify_submission(submission, sample_submission):

    assert submission.shape == sample_submission.shape, \
        'Expected submission to have shape {} but got {}'.format(sample_submission.shape, submission.shape)

    for submission_id, correct_id in zip(submission['SK_ID_CURR'].values, sample_submission['SK_ID_CURR'].values):
        assert correct_id == submission_id, \
            'Wrong id: expected {} but got {}'.format(correct_id, submission_id)


def get_logger():
    return logging.getLogger('home-credit')


def init_logger():
    logger = logging.getLogger('home-credit')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def log_loss_row(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    scores = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    return scores


def read_params(ctx):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        try:
            neptune_config = read_yaml('neptune.yaml')
        except FileNotFoundError:
            neptune_config = read_yaml('../neptune.yaml')
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def safe_eval(obj):
    try:
        return eval(obj)
    except Exception:
        return obj


def save_evaluation_predictions(experiment_dir, y_true, y_pred, raw_data):
    raw_data['y_pred'] = y_pred
    raw_data['score'] = log_loss_row(y_true, y_pred)

    raw_data.sort_values('score', ascending=False, inplace=True)

    filepath = os.path.join(experiment_dir, 'evaluation_predictions.csv')
    raw_data.to_csv(filepath, index=None)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def stratified_train_valid_split(meta_train, target_column, target_bins, valid_size, random_state=1234):
    y = meta_train[target_column].values
    bins = np.linspace(0, y.shape[0], target_bins)
    y_binned = np.digitize(y, bins)

    return train_test_split(meta_train, test_size=valid_size, stratify=y_binned, random_state=random_state)
