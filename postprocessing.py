import numpy as np

from steps.base import BaseTransformer


class Clipper(BaseTransformer):
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, prediction):
        prediction_ = np.where(prediction < self.min_val, self.min_val, prediction)
        prediction_ = np.where(prediction_ > self.max_val, self.max_val, prediction_)
        return {'clipped_prediction': prediction_}
