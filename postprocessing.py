import numpy as np

from steppy.base import BaseTransformer


class Clipper(BaseTransformer):
    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def transform(self, prediction):
        prediction_ = np.clip(prediction, self.min_val, self.max_val)
        return {'clipped_prediction': prediction_}
