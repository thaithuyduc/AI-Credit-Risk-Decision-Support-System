"""
This module contains custom transformers for data preprocessing in a machine learning pipeline. The transformers include:
- AddPrefix: Adds a specified prefix to column names, excluding certain columns.
- RemoveFeature: Removes specified columns from the dataset.
- Aggeration: Performs aggregation on categorical and numerical columns based on a specified groupby column.
- FrequencyGrouping: Groups infrequent categories in categorical columns into an 'Others' category based on a specified frequency threshold.

These transformers are designed to be used with scikit-learn's Pipeline and ColumnTransformer for seamless integration into data preprocessing workflows.
"""

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output='pandas', display='diagram')

class AddPrefix(BaseEstimator, TransformerMixin):
    def __init__(self, prefix:str, exclude:list[str]):
        """
        Used to add prefix to columns' name
        
        Args:
            prefix: prefix added to columns' name
            exclude: columns that don't need prefix
        """
        self.prefix = prefix
        self.exclude = exclude

    def fit(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        self.new_cols_name_ = [
            col if col in self.exclude else f'{self.prefix}_{col}'
            for col in X_copy.columns
        ]
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_copy.columns = self.new_cols_name_
        return X_copy

    def get_feature_names_out(self, input_features=None): # must have this method
        # must have 'input_features' parameter
        return self.new_cols_name_


class RemoveFeature(BaseEstimator, TransformerMixin):
    def __init__(self, columns_removed:list[str]):
        """
        columns_removed: columns which will be removed from dataset
        """
        self.columns_removed = columns_removed

    def fit(self, X:pd.DataFrame, y=None):
        missing = set(self.columns_removed) - set(X.columns) # check if columns's name removed is misspelled
        if missing:
            raise ValueError(f"Các cột sau không tồn tại trong dữ liệu: {missing}")
        
        # Giữ nguyên thứ tự cột gốc, chỉ loại bỏ cột cần xóa
        self.columns_selected_ = [
            col for col in X.columns if col not in self.columns_removed
        ]
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        X_new = X_copy[self.columns_selected_]
        return X_new

    def get_feature_names_out(self, input_features=None): # must have this method
        # must have 'input_features' parameter
        return self.columns_selected_


class Aggeration(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols:list[str], num_cols:list[str], groupby_col:str):
        """
        Used to perform aggregation by specific columns, num_cols and cat_cols will have differenct aggregation
        
        Args:
            cat_cols: categorical column's name
            num_cols: numerical column's name
            groupby_col: specific column's name used for groupby to perform aggregation
        """
        if groupby_col in num_cols:
            num_cols.remove(groupby_col)

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.groupby_col = groupby_col

    def _build_agg_dict(self):
        agg_dict = {}
        for col in self.cat_cols:
            agg_dict[col] = ['mean'] # ['sum', 'mean']
        for col in self.num_cols:
            agg_dict[col] = ['mean', 'max'] # ['min', 'max', 'mean']
        return agg_dict
    
    def fit(self, X:pd.DataFrame, y=None):
        agg_dict = self._build_agg_dict()
        self.new_cols_name_ = [self.groupby_col] + [
            f'{col}_{stat.lower()}' for col, stats in agg_dict.items() for stat in stats
        ]
        return self

    def transform(self, X:pd.DataFrame):
        X_copy = X.copy()
        agg_dict = self._build_agg_dict()

        X_agg = X_copy.groupby(self.groupby_col).agg(agg_dict)
        X_agg.columns = self.new_cols_name_[1:]
        X_agg.reset_index(inplace=True)
        
        return X_agg

    def get_feature_names_out(self, input_features=None): # must have this method
        # must have 'input_features' parameter
        return self.new_cols_name_


class FrequencyGrouping(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.999999):
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y=None):
        X_copy = X.copy()
        self.dict_categories_ = {}
        columns = X_copy.columns

        for col in columns:
            col: str
            frequency = X_copy[col].value_counts(normalize=True)
            cum_frequency = np.cumsum(frequency)

            kept_categories = frequency[cum_frequency <= self.threshold].index.tolist()
            self.dict_categories_[col] = kept_categories

        return self
    
    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()
        
        for col, kept_categories in self.dict_categories_.items():
            col: str
            kept_categories: list
            X_copy[col] = X_copy[col].apply(lambda x: x if x in kept_categories else 'Others')

        return X_copy