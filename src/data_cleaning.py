import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class ApplicationCleaning(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        return {'X': X}


class BureauCleaning(BaseTransformer):
    def __init__(self):
        super().__init__()

    def transform(self, bureau):
        bureau['AMT_CREDIT_SUM'].fillna(0, inplace=True)
        bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
        bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
        bureau['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)

        return {'bureau': bureau}
