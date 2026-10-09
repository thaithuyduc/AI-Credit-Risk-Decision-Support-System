"""
This module contains the main pipelines for data preprocessing and feature engineering for different datasets in the data pipeline. Each class represents a specific pipeline for a dataset, implementing the fit and transform methods to process the data accordingly.
The pipelines include:
- ApplicationTrainPipeline: Preprocesses the application training dataset, including missing value handling, categorical encoding, and feature selection.
- BureauBalancePipeline: Preprocesses the bureau balance dataset, including categorical encoding and aggregation of features.
- POSCASHPipeline: Preprocesses the POS cash dataset, including categorical encoding and aggregation of features.
- InstallmentPaymentPipeline: Preprocesses the installment payment dataset, including aggregation of features.
- CreditCardPipeline: Preprocesses the credit card dataset, including missing value handling, categorical encoding, and aggregation of features.
- BureauPipeline: Preprocesses the bureau dataset, including missing value handling and feature selection.
- PreviousApplicationPipeline: Preprocesses the previous application dataset, including missing value handling and feature selection.

These pipelines are designed to be used with scikit-learn's Pipeline and ColumnTransformer for seamless integration into data preprocessing workflows, allowing for consistent and efficient data transformation across different datasets.
"""

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from .custom_transformer import AddPrefix, RemoveFeature, FrequencyGrouping, Aggeration
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config

set_config(transform_output='pandas', display='diagram')

class ApplicationTrainPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()

        missing_ratio = X_copy.isnull().mean().astype(float).to_frame()
        missing_ratio = missing_ratio.rename(columns={0: "missing_ratio"})
        columns_remove_application_train = list(missing_ratio[missing_ratio['missing_ratio'] > 0.5].index)
        
        self.missing_ratio_feature_removed_application_train_ = RemoveFeature(columns_removed = columns_remove_application_train)
        X_copy = self.missing_ratio_feature_removed_application_train_.fit_transform(X_copy)

        obj_columns = list(X_copy.select_dtypes('object').columns)
        self.categorical_encoder_ = ColumnTransformer(
            transformers=[
                ('ordinal_encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), obj_columns)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        X_copy = self.categorical_encoder_.fit_transform(X_copy)

        all_cols = list(X_copy.columns)
        binary_columns = [col for col in all_cols if set(X_copy[col].dropna().unique()).issubset({0, 1})]
        binary_columns_mean = X_copy[binary_columns].mean()
        binary_columns_remove = list(
            binary_columns_mean[(binary_columns_mean < 0.02) |
                         (binary_columns_mean > 1 - 0.02)].index
        )

        self.binary_feature_removed_application_train_ = RemoveFeature(columns_removed = binary_columns_remove)
        X_copy = self.binary_feature_removed_application_train_.fit_transform(X_copy)

        self.application_train_add_prefix_ = AddPrefix(prefix='AT', exclude=['SK_ID_CURR'])
        X_copy = self.application_train_add_prefix_.fit_transform(X_copy)
        
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_final = self.missing_ratio_feature_removed_application_train_.transform(X_copy)
        X_final = self.categorical_encoder_.transform(X_final)
        X_final = self.binary_feature_removed_application_train_.transform(X_final)
        X_final = self.application_train_add_prefix_.transform(X_final)
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.application_train_add_prefix_.get_feature_names_out()


class BureauBalancePipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        cat_cols = list(X_copy.select_dtypes(include=['object']).columns)
        num_cols = list(X_copy.select_dtypes(include=['number']).columns)

        cat_pipeline = Pipeline([
            ('frequency_grouping', FrequencyGrouping()),
            ('one_hot_encode', OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')),
        ])
        
        self.cat_cols_transformer_ = ColumnTransformer(
            transformers=[
                ('cat_processing', cat_pipeline, cat_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        X_copy = self.cat_cols_transformer_.fit_transform(X_copy)

        cat_cols_after_onehot = [col for col in X_copy.columns if col.startswith(tuple([f"{column}_" for column in cat_cols]))]
        self.aggregate_ = Aggeration(cat_cols=cat_cols_after_onehot, num_cols=num_cols, groupby_col='SK_ID_BUREAU')
        X_copy = self.aggregate_.fit_transform(X_copy)

        self.bureau_balance_add_prefix_ = AddPrefix(prefix='BB', exclude=['SK_ID_BUREAU'])
        X_copy = self.bureau_balance_add_prefix_.fit_transform(X_copy)
        
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_final = self.cat_cols_transformer_.transform(X_copy)
        X_final = self.aggregate_.transform(X_final)
        X_final = self.bureau_balance_add_prefix_.transform(X_final)
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.bureau_balance_add_prefix_.get_feature_names_out()


class POSCASHPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        X_copy.drop(columns=['SK_ID_CURR'], inplace=True)
        cat_cols = list(X_copy.select_dtypes(include=['object']).columns)
        num_cols = list(X_copy.select_dtypes(include=['number']).columns)
        
        cat_pipeline = Pipeline([
            ('frequency_grouping', FrequencyGrouping()),
            ('one_hot_encode', OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')),
        ])
        
        self.cat_cols_transformer_ = ColumnTransformer(
            transformers=[
                ('cat_processing', cat_pipeline, cat_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        X_copy = self.cat_cols_transformer_.fit_transform(X_copy)

        cat_cols_after_onehot = [col for col in X_copy.columns if col.startswith(tuple([f"{column}_" for column in cat_cols]))]
        self.aggregate_ = Aggeration(cat_cols=cat_cols_after_onehot, num_cols=num_cols, groupby_col='SK_ID_PREV')
        X_copy = self.aggregate_.fit_transform(X_copy)
        
        self.pos_cash_add_prefix_ = AddPrefix(prefix='PC', exclude=['SK_ID_PREV'])
        X_copy = self.pos_cash_add_prefix_.fit_transform(X_copy)
        
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_copy.drop(columns=['SK_ID_CURR'], inplace=True)
        
        X_final = self.cat_cols_transformer_.transform(X_copy)
        X_final = self.aggregate_.transform(X_final)
        X_final = self.pos_cash_add_prefix_.transform(X_final)
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.pos_cash_add_prefix_.get_feature_names_out()


class InstallmentPaymentPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        X_copy.drop(columns=['SK_ID_CURR'], inplace=True)
        cat_cols = list(X_copy.select_dtypes(include=['object']).columns) # empty list
        num_cols = list(X_copy.select_dtypes(include=['number']).columns)
        
        self.aggregate_ = Aggeration(cat_cols=cat_cols, num_cols=num_cols, groupby_col='SK_ID_PREV')
        X_copy = self.aggregate_.fit_transform(X_copy)

        self.installment_payment_add_prefix_ = AddPrefix(prefix='IP', exclude=['SK_ID_PREV'])
        X_copy = self.installment_payment_add_prefix_.fit_transform(X_copy)

        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_copy.drop(columns=['SK_ID_CURR'], inplace=True)
        
        X_final = self.aggregate_.transform(X_copy)
        X_final = self.installment_payment_add_prefix_.transform(X_final)
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.installment_payment_add_prefix_.get_feature_names_out()


class CreditCardPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        X_copy.drop(columns=['SK_ID_CURR'], inplace=True)

        # auto detect features with missing ratio > 0.1, to remove them
        missing_ratio = X_copy.isnull().mean().astype(float).to_frame()
        missing_ratio = missing_ratio.rename(columns={0: "missing_ratio"})
        columns_remove_credit_card = list(missing_ratio[missing_ratio['missing_ratio'] > 0.5].index)

        self.feature_removed_credit_card_ = RemoveFeature(columns_removed = columns_remove_credit_card)
        X_copy = self.feature_removed_credit_card_.fit_transform(X_copy)

        cat_cols = list(X_copy.select_dtypes(include=['object']).columns)
        num_cols = list(X_copy.select_dtypes(include=['number']).columns)

        cat_pipeline = Pipeline([
            ('frequency_grouping', FrequencyGrouping()),
            ('one_hot_encode', OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')),
        ])
        
        self.cat_cols_transformer_ = ColumnTransformer(
            transformers=[
                ('cat_processing', cat_pipeline, cat_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        X_copy = self.cat_cols_transformer_.fit_transform(X_copy)

        cat_cols_after_onehot = [col for col in X_copy.columns if col.startswith(tuple([f"{column}_" for column in cat_cols]))]
        self.aggregate_ = Aggeration(cat_cols=cat_cols_after_onehot, num_cols=num_cols, groupby_col='SK_ID_PREV')
        X_copy = self.aggregate_.fit_transform(X_copy)

        self.credit_card_add_prefix_ = AddPrefix(prefix='CC', exclude=['SK_ID_PREV'])
        X_copy = self.credit_card_add_prefix_.fit_transform(X_copy)
        
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_copy.drop(columns=['SK_ID_CURR'], inplace=True)
        
        X_final = self.feature_removed_credit_card_.transform(X_copy)
        X_final = self.cat_cols_transformer_.transform(X_final)
        X_final = self.aggregate_.transform(X_final)
        X_final = self.credit_card_add_prefix_.transform(X_final)
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.credit_card_add_prefix_.get_feature_names_out()


class BureauPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()

        # auto detect features with missing ratio > 0.1, to remove them
        missing_ratio = X_copy.isnull().mean().astype(float).to_frame()
        missing_ratio = missing_ratio.rename(columns={0: "missing_ratio"})
        columns_remove_bureau = list(missing_ratio[missing_ratio['missing_ratio'] > 0.5].index)

        self.feature_removed_bureau_ = RemoveFeature(columns_removed = columns_remove_bureau)
        X_copy = self.feature_removed_bureau_.fit_transform(X_copy)

        self.bureau_add_prefix_ = AddPrefix(prefix='B', exclude=['SK_ID_CURR', 'SK_ID_BUREAU'])
        X_copy = self.bureau_add_prefix_.fit_transform(X_copy)
        
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        
        X_final = self.feature_removed_bureau_.transform(X_copy)
        X_final = self.bureau_add_prefix_.transform(X_final)
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.bureau_add_prefix_.get_feature_names_out()


class PreviousApplicationPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()

        # auto detect features with missing ratio > 0.1, to remove them
        missing_ratio = X_copy.isnull().mean().astype(float).to_frame()
        missing_ratio = missing_ratio.rename(columns={0: "missing_ratio"})
        columns_remove_previous_application = list(missing_ratio[missing_ratio['missing_ratio'] > 0.5].index)
        
        self.feature_removed_previous_application_ = RemoveFeature(columns_removed = columns_remove_previous_application)
        X_copy = self.feature_removed_previous_application_.fit_transform(X_copy)

        self.previous_application_add_prefix_ = AddPrefix(prefix='PA', exclude=['SK_ID_CURR', 'SK_ID_PREV'])
        X_copy = self.previous_application_add_prefix_.fit_transform(X_copy)
        
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        
        X_final = self.feature_removed_previous_application_.transform(X_copy)
        X_final = self.previous_application_add_prefix_.transform(X_final)
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.previous_application_add_prefix_.get_feature_names_out()