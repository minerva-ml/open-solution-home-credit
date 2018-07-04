from copy import deepcopy

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()


class FeatureJoiner(BaseTransformer):
    def __init__(self, use_nan_count=False, **kwargs):
        super().__init__()
        self.use_nan_count = use_nan_count

    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.reset_index(drop=True, inplace=True)
        features = pd.concat(features, axis=1).astype(np.float32)
        if self.use_nan_count:
            features['nan_count'] = features.isnull().sum(axis=1)

        outputs = dict()
        outputs['features'] = features
        outputs['feature_names'] = list(features.columns)
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


class GroupbyAggregateDiffs(BaseTransformer):
    def __init__(self, groupby_aggregations, use_diffs_only=False):
        super().__init__()
        self.groupby_aggregations = groupby_aggregations
        self.use_diffs_only = use_diffs_only
        self.features = []
        self.groupby_feature_names = []

    @property
    def feature_names(self):
        if self.use_diffs_only:
            return self.diff_feature_names
        else:
            return self.groupby_feature_names + self.diff_feature_names

    def fit(self, main_table, **kwargs):
        for groupby_cols, specs in self.groupby_aggregations:
            group_object = main_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                group_features = group_object[select].agg(agg).reset_index() \
                    .rename(index=str,
                            columns={select: groupby_aggregate_name})[groupby_cols + [groupby_aggregate_name]]

                self.features.append((groupby_cols, group_features))
                self.groupby_feature_names.append(groupby_aggregate_name)
        return self

    def transform(self, main_table, **kwargs):
        main_table = self._merge_grouby_features(main_table)
        main_table = self._add_diff_features(main_table)

        return {'numerical_features': main_table[self.feature_names].astype(np.float32)}

    def _merge_grouby_features(self, main_table):
        for groupby_cols, groupby_features in self.features:
            main_table = main_table.merge(groupby_features,
                                          on=groupby_cols,
                                          how='left')
        return main_table

    def _add_diff_features(self, main_table):
        self.diff_feature_names = []
        for groupby_cols, specs in self.groupby_aggregations:
            for select, agg in specs:
                if agg in ['mean', 'median', 'max', 'min']:
                    groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                    diff_feature_name = '{}_diff'.format(groupby_aggregate_name)
                    abs_diff_feature_name = '{}_abs_diff'.format(groupby_aggregate_name)

                    main_table[diff_feature_name] = main_table[select] - main_table[groupby_aggregate_name]
                    main_table[abs_diff_feature_name] = np.abs(main_table[select] - main_table[groupby_aggregate_name])

                    self.diff_feature_names.append(diff_feature_name)
                    self.diff_feature_names.append(abs_diff_feature_name)

        return main_table

    def load(self, filepath):
        params = joblib.load(filepath)
        self.features = params['features']
        self.groupby_feature_names = params['groupby_feature_names']
        return self

    def persist(self, filepath):
        params = {'features': self.features,
                  'groupby_feature_names': self.groupby_feature_names}
        joblib.dump(params, filepath)

    def _create_colname_from_specs(self, groupby_cols, agg, select):
        return '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)


class GroupbyAggregateMerge(BaseTransformer):
    def __init__(self, table_name, id_columns, groupby_aggregations):
        super().__init__()
        self.table_name = table_name
        self.id_columns = id_columns
        self.groupby_aggregations = groupby_aggregations

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove(self.id_columns[0])
        return feature_names

    def fit(self, main_table, side_table, **kwargs):
        features = pd.DataFrame({self.id_columns[0]: side_table[self.id_columns[0]].unique()})

        for groupby_cols, specs in self.groupby_aggregations:
            group_object = side_table.groupby(groupby_cols)
            for select, agg in specs:
                groupby_aggregate_name = self._create_colname_from_specs(groupby_cols, select, agg)
                features = features.merge(group_object[select]
                                          .agg(agg)
                                          .reset_index()
                                          .rename(index=str,
                                                  columns={select: groupby_aggregate_name})
                                          [groupby_cols + [groupby_aggregate_name]],
                                          on=groupby_cols,
                                          how='left')
        self.features = features
        return self

    def transform(self, main_table, side_table, **kwargs):
        main_table = main_table.merge(self.features,
                                      left_on=[self.id_columns[0]],
                                      right_on=[self.id_columns[1]],
                                      how='left',
                                      validate='one_to_one')

        return {'numerical_features': main_table[self.feature_names].astype(np.float32)}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)

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
                                             'income_credit_percentage',
                                             'income_per_child',
                                             'income_per_person',
                                             'payment_rate',
                                             'phone_to_birth_ratio',
                                             'phone_to_employ_ratio',
                                             'external_sources_weighted',
                                             'external_sources_min',
                                             'external_sources_max',
                                             'external_sources_sum',
                                             'external_sources_mean',
                                             'external_sources_nanmedian',
                                             'short_employment',
                                             'young_age']

    def transform(self, X, **kwargs):
        X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
        X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
        X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
        X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
        X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
        X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
        X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
        X['external_sources_weighted'] = X.EXT_SOURCE_1 * 2 + X.EXT_SOURCE_2 * 3 + X.EXT_SOURCE_3 * 4
        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        X['short_employment'] = (X['DAYS_EMPLOYED'] > -2000).astype(int)
        X['young_age'] = (X['DAYS_BIRTH'] > -14000).astype(int)

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

    def fit(self, X, bureau, **kwargs):
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
    def __init__(self, **kwargs):
        self.features = None

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove('SK_ID_CURR')
        return feature_names

    def fit(self, X, credit_card, **kwargs):
        static_features = self._static_features(X, credit_card, **kwargs)
        dynamic_features = self._dynamic_features(X, credit_card, **kwargs)

        self.features = pd.merge(static_features,
                                 dynamic_features,
                                 on=['SK_ID_CURR'],
                                 validate='one_to_one')
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

    def _static_features(self, X, credit_card, **kwargs):
        credit_card['number_of_instalments'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
            'CNT_INSTALMENT_MATURE_CUM']

        credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        group_object = credit_card.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].agg('nunique').reset_index()
        group_object.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = credit_card.groupby(by=['SK_ID_CURR'])['number_of_instalments'].sum().reset_index()
        group_object.rename(index=str, columns={'number_of_instalments': 'credit_card_total_instalments'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        features['credit_card_installments_per_loan'] = (
                features['credit_card_total_instalments'] / features['credit_card_number_of_loans'])

        group_object = credit_card.groupby(by=['SK_ID_CURR'])['credit_card_max_loading_of_credit_limit'].agg(
            'mean').reset_index()
        group_object.rename(index=str, columns={
            'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = credit_card.groupby(
            by=['SK_ID_CURR'])['SK_DPD'].agg('mean').reset_index()
        group_object.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
        group_object.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
        group_object.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features[
            'credit_card_drawings_total']

        return features

    def _dynamic_features(self, X, credit_card, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        credit_card_sorted = credit_card.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
        credit_card_sorted['credit_card_monthly_diff'] = credit_card_sorted.groupby(
            by='SK_ID_CURR')['AMT_BALANCE'].diff()
        group_object = credit_card_sorted.groupby(['SK_ID_CURR'])['credit_card_monthly_diff'].agg('mean').reset_index()
        group_object.rename(index=str,
                            columns={'credit_card_monthly_diff': 'credit_card_monthly_diff_mean'},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        return features


class POSCASHBalanceFeatures(BaseTransformer):
    def __init__(self, **kwargs):
        self.features = None

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove('SK_ID_CURR')
        return feature_names

    def fit(self, X, pos_cash, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': pos_cash['SK_ID_CURR'].unique()})

        pos_cash_sorted = pos_cash.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
        group_object = pos_cash_sorted.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].last().reset_index()
        group_object.rename(index=str,
                            columns={'CNT_INSTALMENT_FUTURE': 'pos_cash_remaining_installments'},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        pos_cash['is_contract_status_completed'] = pos_cash['NAME_CONTRACT_STATUS'] == 'Completed'
        group_object = pos_cash.groupby(['SK_ID_CURR'])['is_contract_status_completed'].sum().reset_index()
        group_object.rename(index=str,
                            columns={'is_contract_status_completed': 'pos_cash_completed_contracts'},
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


class ConcatFeatures(BaseTransformer):
    def transform(self, **kwargs):
        features_concat = []
        for _, feature in kwargs.items():
            feature.reset_index(drop=True, inplace=True)
            features_concat.append(feature)
        features_concat = pd.concat(features_concat, axis=1)
        return {'concatenated_features': features_concat}
