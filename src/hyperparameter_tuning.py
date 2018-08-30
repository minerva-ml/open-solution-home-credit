from attrdict import AttrDict
import numpy as np
import hyperopt as hp


class Tuner:
    def __init__(self, config, runs, maximize=None):
        self.config = config
        self.runs = runs
        self.maximize = maximize
        self.run_id = 0

    @property
    def in_progress(self):
        return self.run_id != self.runs

    def next(self, score=None):
        self.run_id += 1
        return self._next(score)

    def _next(self, score):
        return NotImplementedError


class RandomSearchTuner(Tuner):
    def _next(self, score):
        return RandomSearchTuner.get_random_config(self.config)

    @staticmethod
    def get_random_config(tuning_config):
        config_run = {}
        for tunable_name in tuning_config.keys():
            param_choice = {}
            for param, value in tuning_config[tunable_name].items():
                param_range, sampling_mode = value
                param_choice[param] = RandomSearchTuner.random_sample_from_param_space(param_range, sampling_mode)
            config_run[tunable_name] = param_choice
        return AttrDict(config_run)

    @staticmethod
    def random_sample_from_param_space(param_range, mode):
        if mode == 'list':
            value = np.random.choice(param_range)
            if isinstance(value, np.str_):
                value = str(value)
        else:
            range_min, range_max = param_range
            if mode == 'choice':
                value = np.random.choice(range(range_min, range_max, 1))
            elif mode == 'uniform':
                value = np.random.uniform(low=range_min, high=range_max)
            elif mode == 'log-uniform':
                value = np.exp(np.random.uniform(low=np.log(range_min), high=np.log(range_max)))
            else:
                raise NotImplementedError
        return value


class HyperoptTuner(Tuner):
    def __init__(self, config, runs, maximize=None):
        super().__init__(config, runs, maximize=None)
        self.space = config

    def _next(self, score):
        return NotImplementedError


class SkoptTuner(Tuner):
    def _next(self, score):
        return NotImplementedError


def set_params(config, config_run):
    for name, params in config_run.items():
        for param, value in params.items():
            config[name][param] = value
    return config
