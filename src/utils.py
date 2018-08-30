import logging
import os
import random
import sys
import multiprocessing as mp
from functools import reduce

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from attrdict import AttrDict
from steppy.base import BaseTransformer
from sklearn.externals import joblib
import sklearn.preprocessing as prep


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


def read_params(ctx, fallback_file):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml(fallback_file)
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def parameter_eval(param):
    try:
        return eval(param)
    except Exception:
        return param


def persist_evaluation_predictions(experiment_directory, y_pred, raw_data, id_column, target_column):
    raw_data.loc[:, 'y_pred'] = y_pred.reshape(-1)
    predictions_df = raw_data.loc[:, [id_column, target_column, 'y_pred']]
    filepath = os.path.join(experiment_directory, 'evaluation_predictions.csv')
    logging.info('evaluation predictions csv shape: {}'.format(predictions_df.shape))
    predictions_df.to_csv(filepath, index=None)


def set_seed(seed=90210):
    random.seed(seed)
    np.random.seed(seed)


def calculate_rank(predictions):
    rank = (1 + predictions.rank().values) / (predictions.shape[0] + 1)
    return rank


def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def read_oof_predictions(prediction_dir, train_filepath, id_column, target_column):
    labels = pd.read_csv(train_filepath, usecols=[id_column, target_column])

    filepaths_train, filepaths_test = [], []
    for filepath in sorted(glob.glob('{}/*'.format(prediction_dir))):
        if filepath.endswith('_oof_train.csv'):
            filepaths_train.append(filepath)
        elif filepath.endswith('_oof_test.csv'):
            filepaths_test.append(filepath)

    train_dfs = []
    for filepath in filepaths_train:
        train_dfs.append(pd.read_csv(filepath))
    train_dfs = reduce(lambda df1, df2: pd.merge(df1, df2, on=[id_column, 'fold_id']), train_dfs)
    train_dfs.columns = _clean_columns(train_dfs, keep_colnames=[id_column, 'fold_id'], filepaths=filepaths_train)
    train_dfs = pd.merge(train_dfs, labels, on=[id_column])

    test_dfs = []
    for filepath in filepaths_test:
        test_dfs.append(pd.read_csv(filepath))
    test_dfs = reduce(lambda df1, df2: pd.merge(df1, df2, on=[id_column, 'fold_id']), test_dfs)
    test_dfs.columns = _clean_columns(test_dfs, keep_colnames=[id_column, 'fold_id'], filepaths=filepaths_test)

    return train_dfs, test_dfs


def _clean_columns(df, keep_colnames, filepaths):
    new_colnames = keep_colnames
    feature_colnames = df.drop(keep_colnames, axis=1).columns
    for i, colname in enumerate(feature_colnames):
        model_name = filepaths[i].split('/')[-1].split('.')[0].replace('_oof_train', '').replace('_oof_test', '')
        new_colnames.append(model_name)
    return new_colnames


def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0


def flatten_list(l):
    return [item for sublist in l for item in sublist]


class Normalizer(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = prep.Normalizer()

    def fit(self, X, **kwargs):
        self.estimator.fit(X)
        return self

    def transform(self, X, **kwargs):
        X_ = self.estimator.transform(X)
        return {'X': X_}

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self


class MinMaxScaler(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.estimator = prep.MinMaxScaler()

    def fit(self, X, **kwargs):
        self.estimator.fit(X)
        return self

    def transform(self, X, **kwargs):
        X_ = self.estimator.transform(X)
        return {'X': X_}

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self
