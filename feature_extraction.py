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
    def __init__(self, filepath, id_columns, groupby_aggregations):
        super().__init__()
        self.filename = os.path.basename(filepath).split('.')[0]
        self.file = pd.read_csv(filepath)
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

    def transform(self, X):
        for spec, groupby_aggregations_name in zip(self.groupby_aggregations, self.groupby_aggregations_names):
            group_object = self.file.groupby(spec['groupby'])
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
