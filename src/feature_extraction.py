from copy import deepcopy

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


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
        self.categorical_columns = kwargs['categorical_columns']
        params = deepcopy(kwargs)
        params.pop('categorical_columns', None)
        self.params = params
        self.encoder_class = ce.OrdinalEncoder
        self.categorical_encoder = None

    def fit(self, X, y, **kwargs):
        X_ = X[self.categorical_columns]
        self.categorical_encoder = self.encoder_class(cols=self.categorical_columns, **self.params)
        self.categorical_encoder.fit(X_, y)
        return self

    def transform(self, X, **kwargs):
        X_ = X[self.categorical_columns]
        X_ = self.categorical_encoder.transform(X_)
        return {'categorical_features': X_}

    def load(self, filepath):
        self.categorical_encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.categorical_encoder, filepath)


class GroupbyAggregate(BaseTransformer):
    def __init__(self, groupby_aggregations):
        super().__init__()
        self.groupby_aggregations = groupby_aggregations
        self.groupby_aggregate_names = []

    def transform(self, main_table, **kwargs):
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = main_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                main_table = main_table.merge(group_object[select]
                                              .agg(agg)
                                              .reset_index()
                                              .rename(index=str,
                                                      columns={select: groupby_aggregate_name})
                                              [groupby_cols + [groupby_aggregate_name]],
                                              on=groupby_cols,
                                              how='left')
                self.groupby_aggregate_names.append(groupby_aggregate_name)
        return {'numerical_features': main_table[self.groupby_aggregate_names].astype(np.float32)}

    def _create_colname_from_specs(self, groupby_cols, agg, select):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)


class GroupbyAggregateMerge(BaseTransformer):
    def __init__(self, table_name, id_columns, groupby_aggregations):
        super().__init__()
        self.table_name = table_name
        self.id_columns = id_columns
        self.groupby_aggregations = groupby_aggregations
        self.groupby_aggregate_names = []

    def transform(self, main_table, side_table):
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = side_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                main_table = main_table.merge(group_object[select]
                                              .agg(agg)
                                              .reset_index()
                                              .rename(index=str,
                                                      columns={select: groupby_aggregate_name})
                                              [groupby_cols + [groupby_aggregate_name]],
                                              on=groupby_cols,
                                              how='left')
                self.groupby_aggregate_names.append(groupby_aggregate_name)
        return {'numerical_features': main_table[self.groupby_aggregate_names].astype(np.float32)}

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}_{}_{}_{}'.format(self.table_name, '_'.join(groupby_cols), agg, select)


class ApplicationFeatures(BaseTransformer):
    def __init__(self, categorical_columns, numerical_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.engineered_numerical_columns = ['annuity_income_percentage',
                                             'car_to_birth_ratio',
                                             'car_to_employ_ratio',
                                             'children_ratio',
                                             'credit_to_annuity_ratio',
                                             'credit_to_goods_ratio',
                                             'credit_to_income_ratio',
                                             'days_employed_percentage',
                                             'ext_sources_mean',
                                             'income_credit_percentage',
                                             'income_per_child',
                                             'income_per_person',
                                             'payment_rate',
                                             'phone_to_birth_ratio',
                                             'phone_to_employ_ratio']

    def transform(self, X, **kwargs):
        X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
        X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
        X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
        X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
        X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['ext_sources_mean'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
        X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
        X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']

        return {'numerical_features': X[self.engineered_numerical_columns + self.numerical_columns],
                'categorical_features': X[self.categorical_columns]
                }


class BureauFeatures(BaseTransformer):
    def __init__(self, id_columns, **kwargs):
        super().__init__()
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
        bureau['bureau_overdue_debt_ratio'] = bureau['bureau_total_customer_overdue'] / bureau[
            'bureau_total_customer_debt']

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
        super().__init__()
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


class ConcatFeatures(BaseTransformer):
    def transform(self, **kwargs):
        features_concat = []
        for _, feature in kwargs.items():
            feature.reset_index(drop=True, inplace=True)
            features_concat.append(feature)
        features_concat = pd.concat(features_concat, axis=1)
        return {'concatenated_features': features_concat}
