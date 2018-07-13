import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class ApplicationCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, X):
        X['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        X['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        X['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)

        return {'X': X}


class BureauCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, bureau):
        bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
        bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
        bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan

        if self.fill_missing:
            bureau['AMT_CREDIT_SUM'].fillna(self.fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_DEBT'].fillna(self.fill_value, inplace=True)
            bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(self.fill_value, inplace=True)
            bureau['CNT_CREDIT_PROLONG'].fillna(self.fill_value, inplace=True)

        return {'bureau': bureau}


class CreditCardCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, credit_card):
        credit_card['AMT_DRAWINGS_ATM_CURRENT'][credit_card['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
        credit_card['AMT_DRAWINGS_CURRENT'][credit_card['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

        return {'credit_card': credit_card}


class PreviousApplicationCleaning(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, previous_application):
        previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

        return {'previous_application': previous_application}
