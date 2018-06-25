import os

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class DataFrameByTypeSplitter(BaseTransformer):
    def __init__(self, numerical_columns, categorical_columns, timestamp_columns):
        super().__init__()
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.timestamp_columns = timestamp_columns

    def transform(self, X, y=None, **kwargs):
        outputs = {}

        if self.numerical_columns is not None:
            outputs['numerical_features'] = X[self.numerical_columns]

        if self.categorical_columns is not None:
            outputs['categorical_features'] = X[self.categorical_columns]

        if self.timestamp_columns is not None:
            outputs['timestamp_features'] = X[self.timestamp_columns]

        return outputs


class FeatureJoiner(BaseTransformer):
    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)
        outputs = dict()
        outputs['features'] = pd.concat(features, axis=1).astype(np.float32)
        outputs['feature_names'] = self._get_feature_names(features)
        outputs['categorical_features'] = self._get_feature_names(categorical_feature_list)
        return outputs

    def _get_feature_names(self, dataframes):
        feature_names = []
        for dataframe in dataframes:
            try:
                feature_names.extend(list(dataframe.columns))
            except Exception as e:
                print(e)
                feature_names.append(dataframe.name)

        return feature_names


class CategoricalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.params = kwargs
        self.encoder_class = ce.OrdinalEncoder
        self.categorical_encoder = None

    def fit(self, X, y, **kwargs):
        categorical_columns = list(X.columns)
        self.categorical_encoder = self.encoder_class(cols=categorical_columns, **self.params)
        self.categorical_encoder.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        X_ = self.categorical_encoder.transform(X)
        return {'categorical_features': X_}

    def load(self, filepath):
        self.categorical_encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.categorical_encoder, filepath)


class GroupbyAggregations(BaseTransformer):
    def __init__(self, groupby_aggregations):
        super().__init__()
        self.groupby_aggregations = groupby_aggregations

    @property
    def groupby_aggregations_names(self):
        groupby_aggregations_names = ['{}_{}_{}'.format('_'.join(spec['groupby']),
                                                        spec['agg'],
                                                        spec['select'])
                                      for spec in self.groupby_aggregations]
        return groupby_aggregations_names

    def transform(self, categorical_features, numerical_features):
        X = pd.concat([categorical_features, numerical_features], axis=1)
        for spec, groupby_aggregations_name in zip(self.groupby_aggregations, self.groupby_aggregations_names):
            group_object = X.groupby(spec['groupby'])
            X = X.merge(group_object[spec['select']]
                        .agg(spec['agg'])
                        .reset_index()
                        .rename(index=str,
                                columns={spec['select']: groupby_aggregations_name})
                        [spec['groupby'] + [groupby_aggregations_name]],
                        on=spec['groupby'],
                        how='left')

        return {'numerical_features': X[self.groupby_aggregations_names].astype(np.float32)}


class GroupbyAggregationFromFile(BaseTransformer):
    def __init__(self, filename, id_columns, groupby_aggregations):
        super().__init__()
        self.filename = filename
        self.id_columns = id_columns
        self.groupby_aggregations = groupby_aggregations

    @ property
    def groupby_aggregations_names(self):
        groupby_aggregations_names = ['{}_{}_{}_{}'.format(self.filename,
                                                           '_'.join(spec['groupby']),
                                                           spec['agg'],
                                                           spec['select'])
                                      for spec in self.groupby_aggregations]
        return groupby_aggregations_names

    def transform(self, X, file):
        for spec, groupby_aggregations_name in zip(self.groupby_aggregations, self.groupby_aggregations_names):
            group_object = file.groupby(spec['groupby'])
            X = X.merge(group_object[spec['select']]
                        .agg(spec['agg'])
                        .reset_index()
                        .rename(index=str,
                                columns={spec['select']: groupby_aggregations_name})
                        [spec['groupby'] + [groupby_aggregations_name]],
                        left_on=self.id_columns[0],
                        right_on=self.id_columns[1],
                        how='left')

        return {'numerical_features': X[self.groupby_aggregations_names].astype(np.float32)}


class ApplicationFeatures(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.application_names = ['annuity_income_percentage',
                                  'credit_to_goods_ratio',
                                  'days_employed_percentage',
                                  'ext_sources_mean',
                                  'income_credit_percentage',
                                  'income_per_person',
                                  'payment_rate']

    def transform(self, X, y=None):
        X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

        X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['ext_sources_mean'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']

        return {'numerical_features': X[self.application_names]}


class BureauFeatures(BaseTransformer):
    def __init__(self, id_columns, **kwargs):
        self.id_columns = id_columns
        self.bureau_names = ['bureau_number_of_past_loans',
                             'bureau_number_of_loan_types',
                             'bureau_average_of_past_loans_per_type',
                             'bureau_active_loans_percentage',
                             'bureau_days_credit_diff',
                             'bureau_credit_enddate_percentage',
                             'bureau_days_enddate_diff',
                             'bureau_average_enddate_future',
                             'bureau_total_customer_debt',
                             'bureau_total_customer_credit',
                             'bureau_debt_credit_ratio',
                             'bureau_total_customer_overdue',
                             'bureau_average_creditdays_prolonged'
                             ]

    def fit(self, X, bureau):
        bureau['AMT_CREDIT_SUM'].fillna(0, inplace=True)
        bureau['AMT_CREDIT_SUM_DEBT'].fillna(0, inplace=True)
        bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0, inplace=True)
        bureau['CNT_CREDIT_PROLONG'].fillna(0, inplace=True)

        bureau['bureau_number_of_past_loans'] = bureau.groupby(
            by=['SK_ID_CURR'])['DAYS_CREDIT'].agg('count').reset_index()['DAYS_CREDIT']

        bureau['bureau_number_of_loan_types'] = bureau.groupby(
            by=['SK_ID_CURR'])['CREDIT_TYPE'].agg('nunique').reset_index()['CREDIT_TYPE']

        bureau['bureau_average_of_past_loans_per_type'] = \
            bureau['bureau_number_of_past_loans'] / bureau['bureau_number_of_loan_types']

        bureau['bureau_credit_active_binary'] = bureau.apply(lambda x: int(x.CREDIT_ACTIVE != 'Closed'), axis=1)
        bureau['bureau_active_loans_percentage'] = bureau.groupby(
            by=['SK_ID_CURR'])['bureau_credit_active_binary'].agg('mean').reset_index()['bureau_credit_active_binary']

        bureau['bureau_days_credit_diff'] = bureau.groupby(
            by=['SK_ID_CURR']).apply(
            lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(drop=True)['DAYS_CREDIT']
        bureau['bureau_days_credit_diff'] *= -1
        bureau['bureau_days_credit_diff'] = bureau.groupby(by=['SK_ID_CURR'])['bureau_days_credit_diff'].diff()
        bureau['bureau_days_credit_diff'] = bureau['bureau_days_credit_diff'].fillna(0)

        bureau['bureau_credit_enddate_binary'] = bureau.apply(lambda x: int(x.DAYS_CREDIT_ENDDATE > 0), axis=1)
        bureau['bureau_credit_enddate_percentage'] = bureau.groupby(
            by=['SK_ID_CURR'])['bureau_credit_enddate_binary'].agg('mean').reset_index()['bureau_credit_enddate_binary']

        group = bureau[bureau['bureau_credit_enddate_binary'] == 1].groupby(
            by=['SK_ID_CURR']).apply(
            lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE'], ascending=True)).reset_index(drop=True)
        group['bureau_days_enddate_diff'] = group.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE'].diff()
        group['bureau_days_enddate_diff'] = group['bureau_days_enddate_diff'].fillna(0).astype('uint32')

        bureau = bureau.merge(group[['bureau_days_enddate_diff', 'SK_ID_BUREAU']], on=['SK_ID_BUREAU'], how='left')
        bureau['bureau_average_enddate_future'] = bureau.groupby(
            by=['SK_ID_CURR'])['bureau_days_enddate_diff'].agg('mean').reset_index()['bureau_days_enddate_diff']

        bureau['bureau_total_customer_debt'] = bureau.groupby(
            by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()['AMT_CREDIT_SUM_DEBT']
        bureau['bureau_total_customer_credit'] = bureau.groupby(
            by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].agg('sum').reset_index()['AMT_CREDIT_SUM']
        bureau['bureau_debt_credit_ratio'] = \
            bureau['bureau_total_customer_debt'] / bureau['bureau_total_customer_credit']

        bureau['bureau_total_customer_overdue'] = bureau.groupby(
            by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()['AMT_CREDIT_SUM_OVERDUE']
        bureau['bureau_overdue_debt_ratio'] = bureau['bureau_total_customer_overdue'] / bureau['bureau_total_customer_debt']

        bureau['bureau_average_creditdays_prolonged'] = bureau.groupby(
            by=['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].agg('mean').reset_index()['CNT_CREDIT_PROLONG']

        self.bureau_features = bureau[self.bureau_names +
                                      [self.id_columns[1]]].drop_duplicates(subset=self.id_columns[1])

        return self

    def transform(self, X, **kwargs):
        X = X.merge(self.bureau_features,
                    left_on=self.id_columns[0],
                    right_on=self.id_columns[1],
                    how='left',
                    validate='one_to_one')

        return {'numerical_features': X[self.bureau_names]}

    def load(self, filepath):
        self.bureau_features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.bureau_features, filepath)


class CreditCardBalanceFeatures(BaseTransformer):
    def __init__(self, id_columns, **kwargs):
        self.id_columns = id_columns
        self.credit_card_names = ['credit_card-number_of_loans',
                                  'credit_card-total_instalments',
                                  'credit_card-installments_per_loan',
                                  'credit_card-avg_loading_of_credit_limit',
                                  'credit_card-average_of_days_past_due',
                                  'credit_card-drawings_atm',
                                  'credit_card-drawings_total',
                                  'credit_card-cash_card_ratio',
                                  'credit_card-drawing_ratio',
                                  ]

    def fit(self, X, credit_card):
        credit_card['credit_card-number_of_loans'] = credit_card.groupby(
            by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()['SK_ID_PREV']

        credit_card['number_of_instalments'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
            'CNT_INSTALMENT_MATURE_CUM']
        credit_card['credit_card-total_instalments'] = credit_card.groupby(
            by=['SK_ID_CURR'])['number_of_instalments'].sum().reset_index()['number_of_instalments']
        credit_card['credit_card-installments_per_loan'] = (
                credit_card['credit_card-total_instalments'] / credit_card['credit_card-number_of_loans'])

        credit_card['credit_card-avg_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]
        credit_card['credit_card-avg_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR'])['credit_card-avg_loading_of_credit_limit'].agg('mean').reset_index()[
            'credit_card-avg_loading_of_credit_limit']

        credit_card['credit_card-average_of_days_past_due'] = credit_card.groupby(
            by=['SK_ID_CURR'])['SK_DPD'].agg('mean').reset_index()['SK_DPD']

        credit_card['credit_card-drawings_atm'] = credit_card.groupby(
            by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].sum().reset_index()['AMT_DRAWINGS_ATM_CURRENT']
        credit_card['credit_card-drawings_total'] = credit_card.groupby(
            by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].sum().reset_index()['AMT_DRAWINGS_CURRENT']
        credit_card['credit_card-cash_card_ratio'] = 100 * (
                credit_card['credit_card-drawings_atm'] / credit_card['credit_card-drawings_total'])
        credit_card['credit_card-cash_card_ratio'] = credit_card.groupby(
            by=['SK_ID_CURR'])['credit_card-cash_card_ratio'].agg('mean').reset_index()['credit_card-cash_card_ratio']

        # AVERAGE DRAWING PER CUSTOMER
        credit_card['credit_card-number_of_drawings'] = credit_card.groupby(
            by=['SK_ID_CURR'])['CNT_DRAWINGS_CURRENT'].sum().reset_index()['CNT_DRAWINGS_CURRENT']
        credit_card['credit_card-drawing_ratio'] = 100 * (
                credit_card['credit_card-drawings_total'] / credit_card['credit_card-number_of_drawings'])
        credit_card['credit_card-drawing_ratio'] = credit_card.groupby(
            by=['SK_ID_CURR'])['credit_card-drawing_ratio'].agg('mean').reset_index()['credit_card-drawing_ratio']

        self.credit_card_features = credit_card[self.credit_card_names +
                                                [self.id_columns[1]]].drop_duplicates(subset=self.id_columns[1])

        return self

    def transform(self, X, **kwargs):
        X = X.merge(self.credit_card_features,
                    left_on=self.id_columns[0],
                    right_on=self.id_columns[1],
                    how='left',
                    validate='one_to_one')

        return {'numerical_features': X[self.credit_card_names]}

    def load(self, filepath):
        self.credit_card_features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.credit_card_features, filepath)
