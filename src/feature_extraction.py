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
    def __init__(self, **kwargs):
        self.features = None

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove('SK_ID_CURR')
        return feature_names

    def fit(self, X, bureau):
        bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
        bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        groupby_SK_ID_CURR = bureau.groupby(by=['SK_ID_CURR'])
        features = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})

        group_object = groupby_SK_ID_CURR['DAYS_CREDIT'].agg('count').reset_index()
        group_object.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = groupby_SK_ID_CURR['CREDIT_TYPE'].agg('nunique').reset_index()
        group_object.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        features['bureau_average_of_past_loans_per_type'] = \
            features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

        group_object = groupby_SK_ID_CURR['bureau_credit_active_binary'].agg('mean').reset_index()
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
        group_object.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM'].agg('sum').reset_index()
        group_object.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        features['bureau_debt_credit_ratio'] = \
            features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']

        group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
        group_object.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        features['bureau_overdue_debt_ratio'] = \
            features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']

        group_object = groupby_SK_ID_CURR['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
        group_object.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = groupby_SK_ID_CURR['bureau_credit_enddate_binary'].agg('mean').reset_index()
        group_object.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        self.features = features
        return self

    def transform(self, X, **kwargs):
        X = X.merge(self.features,
                    left_on=['SK_ID_CURR'],
                    right_on=['SK_ID_CURR'],
                    how='left',
                    validate='one_to_one')

        return {'numerical_features': X[self.feature_names]}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)


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
