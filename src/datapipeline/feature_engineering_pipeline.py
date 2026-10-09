"""
This module contains the FeatureEngineeringPipeline class, which is a custom transformer for feature engineering in a machine learning pipeline. The pipeline performs the following steps:
1. Removes columns with a high percentage of missing values (above a specified threshold).
2. Removes binary features that are highly imbalanced (above a specified threshold).
3. Applies RobustScaler to scale numerical features.
4. Imputes missing values with a constant value (0).

The class is designed to be used with scikit-learn's Pipeline and ColumnTransformer for seamless integration into data preprocessing workflows.
"""

import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from .custom_transformer import RemoveFeature

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold: float = 0.1, imbalance_threshold: float = 0.02):
        self.missing_threshold = missing_threshold
        self.imbalance_threshold = imbalance_threshold

    def fit(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()

        # Bước 1: loại cột có tỷ lệ missing quá cao
        missing_ratio = X_copy.isnull().mean()
        columns_to_remove = list(missing_ratio[missing_ratio > self.missing_threshold].index)
        self.feature_remove_ = RemoveFeature(columns_removed=columns_to_remove)
        # X_copy = self.feature_remove_.fit_transform(X_copy)

        # auto detect binary features having ratio < 0.1 or ratio > 1 - 0.1, to remove them
        all_cols = list(X_copy.columns)
        binary_columns = [col for col in all_cols if set(X_copy[col].dropna().unique()).issubset({0, 1})]
        binary_columns_mean = X_copy[binary_columns].mean()
        binary_columns_remove = list(
            binary_columns_mean[(binary_columns_mean < self.imbalance_threshold) |
                         (binary_columns_mean > 1 - self.imbalance_threshold)].index
        )

        self.binary_feature_remove_ = RemoveFeature(columns_removed=binary_columns_remove)
        # X_copy = self.binary_feature_remove_.fit_transform(X_copy)

        non_binary_columns = [col for col in all_cols if col not in binary_columns]
        self.robust_scaler_ = RobustScaler()
        X_copy = self.robust_scaler_.fit_transform(X_copy)
        
        self.imputer_ = SimpleImputer(strategy='constant', fill_value=0)
        X_copy = self.imputer_.fit_transform(X_copy)

        self.modified_columns_ = [re.sub(r'[^\w]+', '_', str(col)) for col in X_copy.columns]
        # X_copy.columns = self.modified_columns_

        return self

    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()

        X_final = X_copy
        # X_final = self.feature_remove_.transform(X_final)
        # X_final = self.binary_feature_remove_.transform(X_final)
        X_final = self.robust_scaler_.transform(X_final)
        X_final = self.imputer_.transform(X_final)
        X_final.columns = self.modified_columns_
        
        return X_final

    def get_feature_names_out(self, input_features=None):
        return self.feature_remove_.get_feature_names_out()