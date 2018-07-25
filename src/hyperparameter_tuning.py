import gc

import numpy as np
from deepsense import neptune
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger

from .utils import set_seed

logger = get_logger()


class RandomSearchOptimizer(BaseTransformer):
    def __init__(self,
                 TransformerClass,
                 params,
                 score_func,
                 maximize,
                 train_input_keys,
                 valid_input_keys,
                 n_runs,
                 callbacks=None):
        super().__init__()
        self.TransformerClass = TransformerClass
        self.param_space = create_param_space(params, n_runs)
        self.train_input_keys = train_input_keys
        self.valid_input_keys = valid_input_keys
        self.score_func = score_func
        self.maximize = maximize
        self.callbacks = callbacks or []
        self.best_transformer = TransformerClass(**self.param_space[0])

    def fit(self, **kwargs):
        if self.train_input_keys:
            train_inputs = {input_key: kwargs[input_key] for input_key in self.train_input_keys}
        else:
            train_inputs = kwargs
        X_valid, y_valid = kwargs[self.valid_input_keys[0]], kwargs[self.valid_input_keys[1]]

        results = []
        for i, param_set in enumerate(self.param_space):
            logger.info('training run {}'.format(i))
            logger.info('parameters: {}'.format(str(param_set)))
            transformer = self.TransformerClass(**param_set)
            transformer.fit(**train_inputs)

            y_pred_valid = transformer.transform(X_valid)
            y_pred_valid_value = list(y_pred_valid.values())[0]
            run_score = self.score_func(y_valid, y_pred_valid_value)
            results.append((run_score, param_set))

            del y_pred_valid, transformer
            gc.collect()

            for callback in self.callbacks:
                callback.on_run_end(score=run_score, params=param_set)

        assert len(results) > 0, 'All random search runs failed, check your parameter space'
        results_sorted = sorted(results, key=lambda x: x[0])

        if self.maximize:
            best_score, best_param_set = results_sorted[-1]
        else:
            best_score, best_param_set = results_sorted[0]

        for callback in self.callbacks:
            callback.on_search_end(results=results)

        self.best_transformer = self.TransformerClass(**best_param_set)
        self.best_transformer.fit(**train_inputs)
        return self

    def transform(self, **kwargs):
        return self.best_transformer.transform(**kwargs)

    def persist(self, filepath):
        self.best_transformer.persist(filepath)

    def load(self, filepath):
        self.best_transformer.load(filepath)
        return self


def create_param_space(params, n_runs):
    seed = np.random.randint(1000)
    param_space = []
    for i in range(n_runs):
        set_seed(seed + i)
        param_choice = {}
        for param, value in params.items():
            if isinstance(value, list):
                if len(value) == 2:
                    mode = 'choice'
                    param_choice[param] = sample_param_space(value, mode)
                else:
                    mode = value[-1]
                    param_choice[param] = sample_param_space(value[:-1], mode)
            else:
                param_choice[param] = value
        param_space.append(param_choice)
    set_seed()
    return param_space


def sample_param_space(value_range, mode):
    if mode == 'list':
        value = np.random.choice(value_range)
        if isinstance(value, np.str_):
            value = str(value)
    else:
        range_min, range_max = value_range
        if mode == 'choice':
            value = np.random.choice(range(range_min, range_max, 1))
        elif mode == 'uniform':
            value = np.random.uniform(low=range_min, high=range_max)
        elif mode == 'log-uniform':
            value = np.exp(np.random.uniform(low=np.log(range_min), high=np.log(range_max)))
        else:
            raise NotImplementedError
    return value


class GridSearchCallback:
    def on_run_end(self, score, params):
        return NotImplementedError

    def on_search_end(self, results):
        return NotImplementedError


class NeptuneMonitor(GridSearchCallback):
    def __init__(self, name):
        self.name = name
        self.ctx = neptune.Context()
        self.highest_params_channel = self._create_text_channel(name='highest params')
        self.lowest_params_channel = self._create_text_channel(name='lowest params')
        self.run_params_channel = self._create_text_channel(name='run params')
        self.run_id = 0

    def on_run_end(self, score, params):
        self.ctx.channel_send('score on run', x=self.run_id, y=score)
        self.run_params_channel.send(y=params)
        self.run_id += 1

    def on_search_end(self, results):
        results_sorted = sorted(results, key=lambda x: x[0])
        highest_score, highest_param_set = results_sorted[-1]
        lowest_score, lowest_param_set = results_sorted[0]

        self.ctx.channel_send('highest score', x=0, y=highest_score)
        self.ctx.channel_send('lowest score', x=0, y=lowest_score)

        self.highest_params_channel.send(y=highest_param_set)
        self.lowest_params_channel.send(y=lowest_param_set)

    def _create_text_channel(self, name=''):
        return self.ctx.create_channel(name=name, channel_type=neptune.ChannelType.TEXT)


class PersistResults(GridSearchCallback):
    def __init__(self, filepath):
        self.filepath = filepath

    def on_search_end(self, results):
        joblib.dump(results, self.filepath)
