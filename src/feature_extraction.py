from copy import deepcopy
from functools import partial

import category_encoders as ce
import numpy as np
import pandas as pd
import cmath
from scipy.stats import kurtosis, iqr, skew
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from steppy.base import BaseTransformer
from steppy.utils import get_logger

from .utils import parallel_apply, safe_div, flatten_list

logger = get_logger()


class IDXMerge(BaseTransformer):
    def __init__(self, id_column, **kwargs):
        super().__init__()
        self.id_column = id_column

    def transform(self, table, features, categorical_features, **kwargs):
        merged_table = table.merge(features,
                                   left_on=self.id_column,
                                   right_on=self.id_column,
                                   how='left',
                                   validate='one_to_one')
        merged_table.drop(self.id_column, axis='columns', inplace=True)

        outputs = dict()
        outputs['features'] = merged_table
        outputs['feature_names'] = list(merged_table.columns)
        outputs['categorical_features'] = categorical_features
        return outputs


class FeatureJoiner(BaseTransformer):
    def __init__(self, id_column, use_nan_count=False, **kwargs):
        super().__init__()
        self.id_column = id_column
        self.use_nan_count = use_nan_count

    def transform(self, numerical_feature_list, categorical_feature_list, **kwargs):
        features = numerical_feature_list + categorical_feature_list
        for feature in features:
            feature.set_index(self.id_column, drop=True, inplace=True)
        features = pd.concat(features, axis=1).astype(np.float32).reset_index()
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


class FeatureConcat(BaseTransformer):
    def transform(self, features_list, feature_names_list, categorical_features_list, **kwargs):
        for feature in features_list:
            feature.reset_index(drop=True, inplace=True)

        outputs = dict()
        outputs['features'] = pd.concat(features_list, axis=1).astype(np.float32)
        outputs['feature_names'] = flatten_list(feature_names_list)
        outputs['categorical_features'] = flatten_list(categorical_features_list)
        return outputs


class CategoricalEncoder(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_class = ce.OrdinalEncoder
        self.categorical_encoder = None

    def fit(self, X, categorical_columns, **kwargs):
        self.categorical_encoder = self.encoder_class(cols=categorical_columns, **kwargs)
        self.categorical_encoder.fit(X)
        return self

    def transform(self, X, **kwargs):
        X_ = self.categorical_encoder.transform(X)
        return {'categorical_features': X_}

    def load(self, filepath):
        self.categorical_encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.categorical_encoder, filepath)


class CategoricalEncodingWrapper(BaseTransformer):
    def __init__(self, encoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.params = deepcopy(kwargs)

    def fit(self, X, y=None, cols=[], **kwargs):
        self.encoder = self.encoder(cols=cols, **self.params)
        self.encoder.fit(X, y)
        return self

    def transform(self, X, y=None, **kwargs):
        transformed = self.encoder.transform(X)
        return {'features': transformed,
                'feature_names': transformed.columns,
                'categorical_features': set(transformed.columns) - set(X.columns)}

    def persist(self, filepath):
        joblib.dump(self.encoder, filepath)

    def load(self, filepath):
        self.encoder = joblib.load(filepath)
        return self


class GroupbyAggregateDiffs(BaseTransformer):
    def __init__(self, id_column, groupby_aggregations, use_diffs_only=False, **kwargs):
        super().__init__()
        self.groupby_aggregations = groupby_aggregations
        self.use_diffs_only = use_diffs_only
        self.features = []
        self.groupby_feature_names = []
        self.id_column = id_column

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

        return {'numerical_features': main_table[[self.id_column] + self.feature_names].astype(np.float32)}

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


class GroupbyAggregate(BaseTransformer):
    def __init__(self, table_name, id_column, groupby_aggregations, **kwargs):
        super().__init__()
        self.table_name = table_name
        self.id_column = id_column
        self.groupby_aggregations = groupby_aggregations

    def fit(self, table, **kwargs):
        features = pd.DataFrame({self.id_column: table[self.id_column].unique()})

        for groupby_cols, specs in self.groupby_aggregations:
            group_object = table.groupby(groupby_cols)
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

    def transform(self, table, **kwargs):
        return {'numerical_features': self.features}

    def load(self, filepath):
        self.features = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.features, filepath)

    def _create_colname_from_specs(self, groupby_cols, select, agg):
        return '{}_{}_{}_{}'.format(self.table_name, '_'.join(groupby_cols), agg, select)


class BasicHandCraftedFeatures(BaseTransformer):
    def __init__(self, num_workers=1, **kwargs):
        self.num_workers = num_workers
        self.features = None
        self.categorical_features = None
        self.categorical_columns = []

    @property
    def feature_names(self):
        feature_names = list(self.features.columns)
        feature_names.remove('SK_ID_CURR')
        return feature_names

    def transform(self, **kwargs):
        return {'numerical_features': self.features,
                'categorical_features': self.categorical_features,
                'categorical_columns': self.categorical_columns}

    def load(self, filepath):
        self.features, self.categorical_features, self.categorical_columns = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump((self.features, self.categorical_features, self.categorical_columns), filepath)


class ApplicationFeatures(BasicHandCraftedFeatures):
    def __init__(self, categorical_columns, numerical_columns, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
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
                                             'young_age',
                                             'cnt_non_child',
                                             'child_to_non_child_ratio',
                                             'income_per_non_child',
                                             'credit_per_person',
                                             'credit_per_child',
                                             'credit_per_non_child',
                                             'ext_source_1_plus_2',
                                             'ext_source_1_plus_3',
                                             'ext_source_2_plus_3',
                                             'ext_source_1_is_nan',
                                             'ext_source_2_is_nan',
                                             'ext_source_3_is_nan',
                                             'hour_appr_process_start_radial_x',
                                             'hour_appr_process_start_radial_y',
                                             'id_renewal_days',
                                             'id_renewal_years',
                                             'id_renewal_days_issue',
                                             'id_renewal_years_issue',
                                             'id_renewal_days_delay',
                                             'id_renewal_years_delay'
                                             ]

    def fit(self, application, **kwargs):
        application['annuity_income_percentage'] = application['AMT_ANNUITY'] / application['AMT_INCOME_TOTAL']
        application['car_to_birth_ratio'] = application['OWN_CAR_AGE'] / application['DAYS_BIRTH']
        application['car_to_employ_ratio'] = application['OWN_CAR_AGE'] / application['DAYS_EMPLOYED']
        application['children_ratio'] = application['CNT_CHILDREN'] / application['CNT_FAM_MEMBERS']
        application['credit_to_annuity_ratio'] = application['AMT_CREDIT'] / application['AMT_ANNUITY']
        application['credit_to_goods_ratio'] = application['AMT_CREDIT'] / application['AMT_GOODS_PRICE']
        application['credit_to_income_ratio'] = application['AMT_CREDIT'] / application['AMT_INCOME_TOTAL']
        application['days_employed_percentage'] = application['DAYS_EMPLOYED'] / application['DAYS_BIRTH']
        application['income_credit_percentage'] = application['AMT_INCOME_TOTAL'] / application['AMT_CREDIT']
        application['income_per_child'] = application['AMT_INCOME_TOTAL'] / (1 + application['CNT_CHILDREN'])
        application['income_per_person'] = application['AMT_INCOME_TOTAL'] / application['CNT_FAM_MEMBERS']
        application['payment_rate'] = application['AMT_ANNUITY'] / application['AMT_CREDIT']
        application['phone_to_birth_ratio'] = application['DAYS_LAST_PHONE_CHANGE'] / application['DAYS_BIRTH']
        application['phone_to_employ_ratio'] = application['DAYS_LAST_PHONE_CHANGE'] / application['DAYS_EMPLOYED']
        application['external_sources_weighted'] = np.nansum(
            np.asarray([1.9, 2.1, 2.6]) * application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
        application['cnt_non_child'] = application['CNT_FAM_MEMBERS'] - application['CNT_CHILDREN']
        application['child_to_non_child_ratio'] = application['CNT_CHILDREN'] / application['cnt_non_child']
        application['income_per_non_child'] = application['AMT_INCOME_TOTAL'] / application['cnt_non_child']
        application['credit_per_person'] = application['AMT_CREDIT'] / application['CNT_FAM_MEMBERS']
        application['credit_per_child'] = application['AMT_CREDIT'] / (1 + application['CNT_CHILDREN'])
        application['credit_per_non_child'] = application['AMT_CREDIT'] / application['cnt_non_child']
        application['ext_source_1_plus_2'] = np.nansum(application[['EXT_SOURCE_1', 'EXT_SOURCE_2']], axis=1)
        application['ext_source_1_plus_3'] = np.nansum(application[['EXT_SOURCE_1', 'EXT_SOURCE_3']], axis=1)
        application['ext_source_2_plus_3'] = np.nansum(application[['EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
        application['ext_source_1_is_nan'] = np.isnan(application['EXT_SOURCE_1'])
        application['ext_source_2_is_nan'] = np.isnan(application['EXT_SOURCE_2'])
        application['ext_source_3_is_nan'] = np.isnan(application['EXT_SOURCE_3'])
        application['hour_appr_process_start_radial_x'] = application['HOUR_APPR_PROCESS_START'].apply(
            lambda x: cmath.rect(1, 2 * cmath.pi * x / 24).real)
        application['hour_appr_process_start_radial_y'] = application['HOUR_APPR_PROCESS_START'].apply(
            lambda x: cmath.rect(1, 2 * cmath.pi * x / 24).imag)
        application['id_renewal_days'] = application['DAYS_ID_PUBLISH'] - application['DAYS_BIRTH']
        application['id_renewal_years'] = (application['DAYS_ID_PUBLISH'] - application['DAYS_BIRTH']) / 366
        application['id_renewal_days_issue'] = np.vectorize(
            lambda x: max(list(set([min(x, age) for age in [0, 20 * 366, 25 * 366, 45 * 366]]) - set([x]))))(
            application['id_renewal_days'])
        application['id_renewal_years_issue'] = np.vectorize(
            lambda x: max(list(set([min(x, age) for age in [0, 20, 25, 46]]) - set([x]))))(
            application['id_renewal_years'])
        application.loc[application['id_renewal_days_issue'] <= 20 * 366, 'id_renewal_days_delay'] = -1
        application.loc[application['id_renewal_years_issue'] <= 20, 'id_renewal_years_delay'] = -1
        application.loc[application['id_renewal_days_issue'] > 20 * 366, 'id_renewal_days_delay'] = \
            application.loc[application['id_renewal_days_issue'] > 20 * 366, 'id_renewal_days'].values - \
            application.loc[application['id_renewal_days_issue'] > 20 * 366, 'id_renewal_days_issue']
        application.loc[application['id_renewal_years_issue'] > 20, 'id_renewal_years_delay'] = \
            application.loc[application['id_renewal_years_issue'] > 20, 'id_renewal_years'].values - \
            application.loc[application['id_renewal_years_issue'] > 20, 'id_renewal_years_issue']
        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            application['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        application['short_employment'] = (application['DAYS_EMPLOYED'] < -2000).astype(int)
        application['young_age'] = (application['DAYS_BIRTH'] < -14000).astype(int)

        self.features = application[['SK_ID_CURR'] + self.engineered_numerical_columns + self.numerical_columns]
        self.categorical_features = application[['SK_ID_CURR'] + self.categorical_columns]
        return self


class BureauFeatures(BasicHandCraftedFeatures):
    def __init__(self, last_k_agg_periods, last_k_agg_period_fractions, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_agg_period_fractions = last_k_agg_period_fractions

        self.num_workers = num_workers
        self.features = None

    def fit(self, bureau, **kwargs):
        bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'], ascending=False, inplace=True)
        bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
        bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        bureau['bureau_credit_type_consumer'] = (bureau['CREDIT_TYPE'] == 'Consumer credit').astype(int)
        bureau['bureau_credit_type_car'] = (bureau['CREDIT_TYPE'] == 'Car loan').astype(int)
        bureau['bureau_credit_type_mortgage'] = (bureau['CREDIT_TYPE'] == 'Mortgage').astype(int)
        bureau['bureau_credit_type_credit_card'] = (bureau['CREDIT_TYPE'] == 'Credit card').astype(int)
        bureau['bureau_credit_type_other'] = (~(bureau['CREDIT_TYPE'].isin(['Consumer credit',
                                                                            'Car loan', 'Mortgage',
                                                                            'Credit card']))).astype(int)
        bureau['bureau_unusual_currency'] = (~(bureau['CREDIT_CURRENCY'] == 'currency 1')).astype(int)

        bureau['days_credit_diff'] = bureau['DAYS_CREDIT'].diff().replace(np.nan, 0)

        features = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})
        groupby = bureau.groupby(by=['SK_ID_CURR'])

        g = groupby['DAYS_CREDIT'].agg('count').reset_index()
        g.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
        g.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
        g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['bureau_average_of_past_loans_per_type'] = \
            features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

        features['bureau_debt_credit_ratio'] = \
            features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']

        features['bureau_overdue_debt_ratio'] = \
            features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']

        func = partial(BureauFeatures.generate_features, agg_periods=self.last_k_agg_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        g = add_last_k_features_fractions(features, 'SK_ID_CURR', period_fractions=self.last_k_agg_period_fractions)
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods):
        all = BureauFeatures.all_installment_features(gr)
        agg = BureauFeatures.last_k_installment_features(gr, agg_periods)
        features = {**all, **agg}
        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return BureauFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_[gr_['DAYS_CREDIT'] >= (-1) * period]

            features = add_features_in_group(features, gr_period, 'days_credit_diff',
                                             ['sum', 'min', 'max', 'mean', 'median', 'std'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'CNT_CREDIT_PROLONG',
                                             ['sum', 'max', 'mean', 'std'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_credit_active_binary',
                                             ['sum', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_credit_type_consumer',
                                             ['sum', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_credit_type_car',
                                             ['sum', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_credit_type_credit_card',
                                             ['sum', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_credit_type_mortgage',
                                             ['sum'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_credit_type_other',
                                             ['sum', 'mean'],
                                             period_name)
        return features


class BureauBalanceFeatures(BasicHandCraftedFeatures):
    def __init__(self, last_k_agg_periods, last_k_agg_period_fractions, last_k_trend_periods, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_agg_period_fractions = last_k_agg_period_fractions
        self.last_k_trend_periods = last_k_trend_periods

        self.num_workers = num_workers
        self.features = None

    def fit(self, bureau_balance, **kwargs):
        bureau_balance['bureau_balance_dpd_level'] = bureau_balance['STATUS'].apply(self._status_to_int)
        bureau_balance['bureau_balance_status_unknown'] = (bureau_balance['STATUS'] == 'X').astype(int)
        bureau_balance['bureau_balance_no_history'] = bureau_balance['MONTHS_BALANCE'].isnull().astype(int)

        features = pd.DataFrame({'SK_ID_CURR': bureau_balance['SK_ID_CURR'].unique()})
        groupby = bureau_balance.groupby(['SK_ID_CURR'])

        func = partial(BureauBalanceFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        g = add_last_k_features_fractions(features, 'SK_ID_CURR', period_fractions=self.last_k_agg_period_fractions)
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    def _status_to_int(self, status):
        if status in ['X', 'C']:
            return 0
        if pd.isnull(status):
            return np.nan
        return int(status)

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        all = BureauBalanceFeatures.all_installment_features(gr)
        agg = BureauBalanceFeatures.last_k_installment_features(gr, agg_periods)
        trend = BureauBalanceFeatures.trend_in_last_k_installment_features(gr, trend_periods)
        features = {**all, **agg, **trend}
        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return BureauBalanceFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

            features = add_features_in_group(features, gr_period, 'bureau_balance_dpd_level',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'bureau_balance_status_unknown',
                                             ['sum', 'mean'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

            features = add_trend_feature(features, gr_period,
                                         'bureau_balance_dpd_level', '{}_period_trend_'.format(period)
                                         )
        return features


class CreditCardBalanceFeatures(BasicHandCraftedFeatures):
    def fit(self, credit_card, **kwargs):
        static_features = self._static_features(credit_card, **kwargs)
        dynamic_features = self._dynamic_features(credit_card, **kwargs)

        self.features = pd.merge(static_features,
                                 dynamic_features,
                                 on=['SK_ID_CURR'],
                                 validate='one_to_one')
        return self

    def _static_features(self, credit_card, **kwargs):
        credit_card['number_of_installments'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
            'CNT_INSTALMENT_MATURE_CUM']

        credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        groupby = credit_card.groupby(by=['SK_ID_CURR'])

        g = groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['SK_DPD'].agg('mean').reset_index()
        g.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['number_of_installments'].agg('sum').reset_index()
        g.rename(index=str, columns={'number_of_installments': 'credit_card_total_installments'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
        g.rename(index=str,
                 columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features[
            'credit_card_drawings_total']

        features['credit_card_installments_per_loan'] = (
                features['credit_card_total_installments'] / features['credit_card_number_of_loans'])

        return features

    def _dynamic_features(self, credit_card, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        credit_card_sorted = credit_card.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])
        credit_card_sorted['credit_card_monthly_diff'] = groupby['AMT_BALANCE'].diff()
        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])

        g = groupby['credit_card_monthly_diff'].agg('mean').reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        return features


class POSCASHBalanceFeatures(BasicHandCraftedFeatures):
    def __init__(self, last_k_agg_periods, last_k_trend_periods, last_k_agg_period_fractions, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_trend_periods = last_k_trend_periods
        self.last_k_agg_period_fractions = last_k_agg_period_fractions

        self.num_workers = num_workers
        self.features = None

    def fit(self, pos_cash, **kwargs):
        pos_cash['is_contract_status_completed'] = pos_cash['NAME_CONTRACT_STATUS'] == 'Completed'
        pos_cash['pos_cash_paid_late'] = (pos_cash['SK_DPD'] > 0).astype(int)
        pos_cash['pos_cash_paid_late_with_tolerance'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)

        features = pd.DataFrame({'SK_ID_CURR': pos_cash['SK_ID_CURR'].unique()})
        groupby = pos_cash.groupby(['SK_ID_CURR'])
        func = partial(POSCASHBalanceFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        g = add_last_k_features_fractions(features, 'SK_ID_CURR', period_fractions=self.last_k_agg_period_fractions)
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        one_time = POSCASHBalanceFeatures.one_time_features(gr)
        all = POSCASHBalanceFeatures.all_installment_features(gr)
        agg = POSCASHBalanceFeatures.last_k_installment_features(gr, agg_periods)
        trend = POSCASHBalanceFeatures.trend_in_last_k_installment_features(gr, trend_periods)
        last = POSCASHBalanceFeatures.last_loan_features(gr)
        features = {**one_time, **all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def one_time_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], inplace=True)
        features = {}

        features['pos_cash_remaining_installments'] = gr_['CNT_INSTALMENT_FUTURE'].tail(1)
        features['pos_cash_completed_contracts'] = gr_['is_contract_status_completed'].agg('sum')

        return features

    @staticmethod
    def all_installment_features(gr):
        return POSCASHBalanceFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD',
                                             ['mean', 'max', 'std'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                             ['mean', 'max', 'std'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

            features = add_trend_feature(features, gr_period,
                                         'SK_DPD', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'SK_DPD_DEF', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        last_installments_ids = gr_[gr_['MONTHS_BALANCE'] == gr_['MONTHS_BALANCE'].max()]['SK_ID_PREV']
        gr_ = gr_[gr_['SK_ID_PREV'].isin(last_installments_ids)]

        features = {}
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                         ['sum', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                         ['sum', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD',
                                         ['mean', 'max', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                         ['mean', 'max', 'std'],
                                         'last_loan_')

        return features


class PreviousApplicationFeatures(BasicHandCraftedFeatures):
    def __init__(self, numbers_of_applications, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.numbers_of_applications = numbers_of_applications
        self.categorical_columns = ['previous_application_approved_last_1_applications_mean',
                                    'previous_application_refused_last_1_applications_mean',
                                    'previous_application_revolving_loan_last_1_applications_mean',
                                    'previous_application_cash_loan_last_1_applications_mean',
                                    'previous_application_consumer_loan_last_1_applications_mean',
                                    'previous_application_NAME_PRODUCT_TYPE_x_sell_last_1_applications_mean',
                                    'previous_application_NAME_PRODUCT_TYPE_walk_in_last_1_applications_mean',
                                    'previous_application_NAME_PAYMENT_TYPE_bank_last_1_applications_mean',
                                    'previous_application_NAME_PAYMENT_TYPE_account_last_1_applications_mean',
                                    'previous_application_NAME_PAYMENT_TYPE_employer_last_1_applications_mean',
                                    'previous_application_channel_type']

    def fit(self, previous_application, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': previous_application['SK_ID_CURR'].unique()})
        prev_app_sorted = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

        prev_app_sorted['approved'] = (prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
        prev_app_sorted['refused'] = (prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
        prev_app_sorted['revolving_loan'] = (prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
        prev_app_sorted['cash_loan'] = (prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Cash loans').astype('int')
        prev_app_sorted['consumer_loan'] = (prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Consumer loans').astype('int')
        prev_app_sorted['credit_goods_diff'] = prev_app_sorted['AMT_CREDIT'] - prev_app_sorted['AMT_GOODS_PRICE']
        prev_app_sorted['credit_goods_ratio'] = prev_app_sorted['AMT_CREDIT'] / prev_app_sorted['AMT_GOODS_PRICE']
        prev_app_sorted['application_credit_diff'] = prev_app_sorted['AMT_APPLICATION'] - prev_app_sorted['AMT_CREDIT']
        prev_app_sorted['application_credit_ratio'] = prev_app_sorted['AMT_APPLICATION'] / prev_app_sorted['AMT_CREDIT']
        prev_app_sorted['NAME_PRODUCT_TYPE_x_sell'] = (prev_app_sorted['NAME_PRODUCT_TYPE'] == 'x-sell').astype('int')
        prev_app_sorted['NAME_PRODUCT_TYPE_walk_in'] = (prev_app_sorted['NAME_PRODUCT_TYPE'] == 'walk-in').astype('int')
        prev_app_sorted['NAME_PAYMENT_TYPE_bank'] = (
                prev_app_sorted['NAME_PAYMENT_TYPE'] == 'Cash through the bank').astype('int')
        prev_app_sorted['NAME_PAYMENT_TYPE_account'] = (
                prev_app_sorted['NAME_PAYMENT_TYPE'] == 'Non-cash from your account').astype('int')
        prev_app_sorted['NAME_PAYMENT_TYPE_employer'] = (
                prev_app_sorted['NAME_PAYMENT_TYPE'] == 'Cashless from the account of the employer').astype('int')

        prev_app_sorted_groupby = prev_app_sorted.groupby(by=['SK_ID_CURR'])

        g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted_groupby['CHANNEL_TYPE'].last().reset_index()
        g.rename(index=str, columns={'CHANNEL_TYPE': 'previous_application_channel_type'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted_groupby['refused'].mean().reset_index()
        g.rename(index=str, columns={
            'refused': 'previous_application_fraction_of_refused_applications'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = PreviousApplicationFeatures.get_last_k_credits_features(prev_app_sorted,
                                                                    numbers_of_applications=self.numbers_of_applications)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        self.features, self.categorical_features = _divide_features(features, self.categorical_columns, 'SK_ID_CURR')
        return self

    @staticmethod
    def get_last_k_credits_features(prev_app_sorted, numbers_of_applications):
        def _get_last_k_applications_feature_name(feature_name, number, suffix):
            return 'previous_application_{}_last_{}_applications_{}'.format(feature_name, number, suffix)

        features = pd.DataFrame({'SK_ID_CURR': prev_app_sorted['SK_ID_CURR'].unique()})

        feature_list = ['CNT_PAYMENT', 'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'NFLAG_INSURED_ON_APPROVAL',
                        'NAME_PRODUCT_TYPE_x_sell', 'NAME_PRODUCT_TYPE_walk_in', 'NAME_PAYMENT_TYPE_bank',
                        'NAME_PAYMENT_TYPE_account', 'NAME_PAYMENT_TYPE_employer', 'approved', 'refused',
                        'revolving_loan', 'cash_loan', 'consumer_loan', 'credit_goods_diff', 'credit_goods_ratio',
                        'application_credit_diff', 'application_credit_ratio']

        for number in numbers_of_applications:
            prev_applications_tail = prev_app_sorted.groupby('SK_ID_CURR').tail(number)
            tail_groupby = prev_applications_tail.groupby('SK_ID_CURR')
            g = tail_groupby[feature_list].agg('mean')

            g = g.rename(axis='columns', mapper=partial(_get_last_k_applications_feature_name,
                                                        number=number,
                                                        suffix='mean')).reset_index()

            features = features.merge(g, how='left', on='SK_ID_CURR')

        return features


class ApplicationPreviousApplicationFeatures(BasicHandCraftedFeatures):
    def __init__(self, numbers_of_applications=[], num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.numbers_of_applications = numbers_of_applications
        self.categorical_columns = ['application_previous_application_the_same_contract_type_last_1_applications_mean',
                                    'application_previous_application_the_same_type_suite_last_1_applications_mean',
                                    'application_previous_application_the_same_weekday_last_1_applications_mean'
                                    ]

    def fit(self, application, previous_application, **kwargs):
        features = pd.DataFrame({'SK_ID_CURR': application['SK_ID_CURR']})
        common_columns = [col for col in application.columns if col in previous_application.columns]
        applications_copy = application[common_columns].copy()

        merged_tables = previous_application[common_columns + ['DAYS_DECISION']].merge(
            applications_copy, on='SK_ID_CURR', how='right', suffixes=('_previous', '_present'))
        merged_sorted = merged_tables.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

        merged_sorted['annuity_diff'] = merged_sorted['AMT_ANNUITY_previous'] - merged_sorted['AMT_ANNUITY_present']
        merged_sorted['annuity_ratio'] = merged_sorted['AMT_ANNUITY_previous'] / merged_sorted['AMT_ANNUITY_present']
        merged_sorted['credit_diff'] = merged_sorted['AMT_CREDIT_previous'] - merged_sorted['AMT_CREDIT_present']
        merged_sorted['credit_ratio'] = merged_sorted['AMT_CREDIT_previous'] / merged_sorted['AMT_CREDIT_present']

        merged_sorted['the_same_contract_type'] = (
                merged_sorted['NAME_CONTRACT_TYPE_previous'] == merged_sorted['NAME_CONTRACT_TYPE_present']).astype(int)
        merged_sorted['the_same_weekday'] = (
                merged_sorted['WEEKDAY_APPR_PROCESS_START_previous'] == merged_sorted[
            'WEEKDAY_APPR_PROCESS_START_present']).astype(int)
        merged_sorted['hour_diff'] = merged_sorted['HOUR_APPR_PROCESS_START_previous'] - merged_sorted[
            'HOUR_APPR_PROCESS_START_present']
        merged_sorted['the_same_type_suite'] = (
                merged_sorted['NAME_TYPE_SUITE_previous'] == merged_sorted['NAME_TYPE_SUITE_present']).astype(int)
        merged_sorted['the_same_type_suite'][merged_sorted['NAME_TYPE_SUITE_previous'].isnull()] = 1

        g = ApplicationPreviousApplicationFeatures.get_last_k_credits_features(merged_sorted,
                                                                               numbers_of_applications=self.numbers_of_applications)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        self.features, self.categorical_features = _divide_features(features, self.categorical_columns, 'SK_ID_CURR')
        return self

    @staticmethod
    def get_last_k_credits_features(merged_sorted, numbers_of_applications):
        def _get_last_k_applications_feature_name(feature_name, number, suffix):
            return 'application_previous_application_{}_last_{}_applications_{}'.format(feature_name, number, suffix)

        features = pd.DataFrame({'SK_ID_CURR': merged_sorted['SK_ID_CURR'].unique()})

        feature_list = ['annuity_diff', 'annuity_ratio', 'credit_diff', 'credit_ratio', 'the_same_contract_type',
                        'the_same_type_suite', 'the_same_weekday', 'hour_diff']

        for number in numbers_of_applications:
            table_tail = merged_sorted.groupby('SK_ID_CURR').tail(number)
            tail_groupby = table_tail.groupby('SK_ID_CURR')
            g = tail_groupby[feature_list].agg('mean')

            g = g.rename(axis='columns',
                         mapper=partial(_get_last_k_applications_feature_name,
                                        number=number,
                                        suffix='mean')).reset_index()

            features = features.merge(g, how='left', on=['SK_ID_CURR'])

        return features


class InstallmentPaymentsFeatures(BasicHandCraftedFeatures):
    def __init__(self, last_k_agg_periods, last_k_agg_period_fractions, last_k_trend_periods, num_workers=1, **kwargs):
        super().__init__(num_workers=num_workers)
        self.last_k_agg_periods = last_k_agg_periods
        self.last_k_agg_period_fractions = last_k_agg_period_fractions
        self.last_k_trend_periods = last_k_trend_periods

        self.num_workers = num_workers
        self.features = None

    def fit(self, installments, **kwargs):
        installments['installment_paid_late_in_days'] = installments['DAYS_ENTRY_PAYMENT'] - installments[
            'DAYS_INSTALMENT']
        installments['installment_paid_late'] = (installments['installment_paid_late_in_days'] > 0).astype(int)
        installments['installment_paid_over_amount'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
        installments['installment_paid_over'] = (installments['installment_paid_over_amount'] > 0).astype(int)

        features = pd.DataFrame({'SK_ID_CURR': installments['SK_ID_CURR'].unique()})
        groupby = installments.groupby(['SK_ID_CURR'])

        func = partial(InstallmentPaymentsFeatures.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       trend_periods=self.last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        g = add_last_k_features_fractions(features, 'SK_ID_CURR', period_fractions=self.last_k_agg_period_fractions)
        features = features.merge(g, on='SK_ID_CURR', how='left')

        self.features = features
        return self

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        all = InstallmentPaymentsFeatures.all_installment_features(gr)
        agg = InstallmentPaymentsFeatures.last_k_installment_features(gr, agg_periods)
        trend = InstallmentPaymentsFeatures.trend_in_last_k_installment_features(gr, trend_periods)
        last = InstallmentPaymentsFeatures.last_loan_features(gr)
        features = {**all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return InstallmentPaymentsFeatures.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_[gr_['DAYS_INSTALMENT'] > (-1) * period]

            features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'iqr'],
                                             period_name)

            features = add_features_in_group(features, gr_period, 'installment_paid_late_in_days',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'kurt', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over_amount',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over',
                                             ['count', 'mean'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_[gr_['DAYS_INSTALMENT'] > (-1) * period]

            features = add_trend_feature(features, gr_period,
                                         'installment_paid_late_in_days', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'installment_paid_over_amount', '{}_period_trend_'.format(period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        last_installments_ids = gr_[gr_['DAYS_INSTALMENT'] == gr_['DAYS_INSTALMENT'].max()]['SK_ID_PREV']
        gr_ = gr_[gr_['SK_ID_PREV'].isin(last_installments_ids)]

        features = {}
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late',
                                         ['count', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over',
                                         ['count', 'mean'],
                                         'last_loan_')
        return features


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def add_last_k_features_fractions(features, id, period_fractions):
    fraction_features = features[[id]].copy()

    for short_period, long_period in period_fractions:
        short_feature_names = get_feature_names_by_period(features, short_period)
        long_feature_names = get_feature_names_by_period(features, long_period)

        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            fraction_features[fraction_feature_name] = features[short_feature] / features[long_feature]
    return fraction_features


def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


def _divide_features(features, categorical_columns, id_column):
    numerical_columns = [column for column in features.columns if column not in categorical_columns]
    return features[numerical_columns], features[[id_column] + categorical_columns]
