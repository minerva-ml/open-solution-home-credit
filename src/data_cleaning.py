import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger
from . import pipeline_config as cfg

logger = get_logger()


class ApplicationCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        super().__init__()
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, application):
        application['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        application['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        application['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        application['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        application['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
        application[cfg.CATEGORICAL_COLUMNS].fillna(-1, inplace=True)

        if self.fill_missing:
            application.fillna(self.fill_value, inplace=True)

        return {'application': application}


class BureauCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, bureau):
        bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
        bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
        bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan

        if self.fill_missing:
            bureau.fillna(self.fill_value, inplace=True)

        bureau['AMT_CREDIT_SUM'].fillna(self.fill_value, inplace=True)
        bureau['AMT_CREDIT_SUM_DEBT'].fillna(self.fill_value, inplace=True)
        bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(self.fill_value, inplace=True)
        bureau['CNT_CREDIT_PROLONG'].fillna(self.fill_value, inplace=True)

        return {'bureau': bureau}


class BureauBalanceCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, bureau_balance):
        if self.fill_missing:
            bureau_balance.fillna(self.fill_value, inplace=True)

        return {'bureau_balance': bureau_balance}


class CreditCardCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        super().__init__()
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, credit_card):
        credit_card['AMT_DRAWINGS_ATM_CURRENT'][credit_card['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
        credit_card['AMT_DRAWINGS_CURRENT'][credit_card['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

        if self.fill_missing:
            credit_card.fillna(self.fill_value, inplace=True)

        return {'credit_card': credit_card}


class InstallmentPaymentsCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, installments):
        if self.fill_missing:
            installments.fillna(self.fill_value, inplace=True)

        return {'installments': installments}


class PosCashCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, pos_cash):
        if self.fill_missing:
            pos_cash.fillna(self.fill_value, inplace=True)

        return {'pos_cash': pos_cash}


class PreviousApplicationCleaning(BaseTransformer):
    def __init__(self, fill_missing=False, fill_value=0, **kwargs):
        super().__init__()
        self.fill_missing = fill_missing
        self.fill_value = fill_value

    def transform(self, previous_application):
        previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

        if self.fill_missing:
            previous_application.fillna(self.fill_value, inplace=True)

        return {'previous_application': previous_application}
