import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class ApplicationCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, X):
        X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        return {'X': X}


class BureauCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, bureau):
        if self.fill_missing:
            bureau['AMT_CREDIT_SUM'].fillna(self.fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_DEBT'].fillna(self.fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(self.fill_value, inplace=True)
            bureau['CNT_CREDIT_PROLONG'].fillna(self.fill_value, inplace=True)

        return {'bureau': bureau}
