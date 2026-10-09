"""
This file contains the MergePipeline class, which is responsible for merging and transforming multiple datasets related to credit risk analysis. 
The class provides methods for fitting and transforming the data, including one-hot encoding and aggregation of categorical and numerical features. 
It also maintains a dictionary of fitted transformers for reuse during transformation.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .custom_transformer import FrequencyGrouping, Aggeration
from .main_pipelines import BureauBalancePipeline, InstallmentPaymentPipeline, POSCASHPipeline, CreditCardPipeline, PreviousApplicationPipeline, BureauPipeline, ApplicationTrainPipeline
from sklearn import set_config

set_config(transform_output='pandas', display='diagram')

class MergePipeline:
    def __init__(self):
        self.bureau_balance_pipeline = BureauBalancePipeline()
        self.installment_payment_pipeline = InstallmentPaymentPipeline()
        self.pos_cash_pipeline = POSCASHPipeline()
        self.credit_card_pipeline = CreditCardPipeline()
        self.previous_application_pipeline = PreviousApplicationPipeline()
        self.bureau_pipeline = BureauPipeline()
        self.application_pipeline = ApplicationTrainPipeline()

        # Lưu các transformer/aggregator theo tên nhóm, thay vì tạo nhiều thuộc tính rời rạc
        self.group_transformers_ = {}

    def _encode_and_aggregate_fit(self, df: pd.DataFrame, group_name: str, groupby_col: str = 'SK_ID_CURR'):
        """
        Dùng khi fit: one-hot encode + aggregate cho 1 nhóm bảng (bureau hoặc previous_application),
        đồng thời lưu lại các transformer đã fit để dùng lại lúc transform.
        """
        cat_cols = list(df.select_dtypes(include=['object']).columns)
        num_cols = list(df.select_dtypes(include=['number']).columns)

        cat_pipeline = Pipeline([
            ('frequency_grouping', FrequencyGrouping()),
            ('one_hot_encode', OneHotEncoder(sparse_output=False, drop='if_binary', handle_unknown='ignore')),
        ])
        
        cat_transformer = ColumnTransformer(
            transformers=[
                ('cat_processing', cat_pipeline, cat_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False,
        )
        df_encoded = cat_transformer.fit_transform(df)

        cat_cols_after_onehot = [
            col for col in df_encoded.columns
            if col.startswith(tuple(f"{c}_" for c in cat_cols))
        ]

        aggregator = Aggeration(cat_cols=cat_cols_after_onehot, num_cols=num_cols, groupby_col=groupby_col)
        df_agg = aggregator.fit_transform(df_encoded)

        # Lưu lại để transform() dùng sau này
        self.group_transformers_[group_name] = {
            'cat_transformer': cat_transformer,
            'aggregator': aggregator,
        }

        return df_agg

    def _encode_and_aggregate_transform(self, df: pd.DataFrame, group_name: str):
        """Dùng khi transform: áp dụng lại transformer đã fit cho nhóm tương ứng."""
        transformers = self.group_transformers_[group_name]
        df_encoded = transformers['cat_transformer'].transform(df)
        df_agg = transformers['aggregator'].transform(df_encoded)
        return df_agg

    def fit(self, bureau_balance:pd.DataFrame, POS_CASH_balance:pd.DataFrame, application_train:pd.DataFrame,
             credit_card_balance:pd.DataFrame, previous_application:pd.DataFrame, bureau:pd.DataFrame, installments_payments:pd.DataFrame):

        bureau_balance_transform = self.bureau_balance_pipeline.fit_transform(bureau_balance)
        installments_payments_transform = self.installment_payment_pipeline.fit_transform(installments_payments)
        POS_CASH_balance_transform = self.pos_cash_pipeline.fit_transform(POS_CASH_balance)
        credit_card_balance_transform = self.credit_card_pipeline.fit_transform(credit_card_balance)
        previous_application_transform = self.previous_application_pipeline.fit_transform(previous_application)
        bureau_transform = self.bureau_pipeline.fit_transform(bureau)
        application_train_transform = self.application_pipeline.fit_transform(application_train)

        # --- Nhóm bureau ---
        merge_bureau = bureau_transform.merge(bureau_balance_transform, on='SK_ID_BUREAU', how='left')
        merge_bureau.drop(columns=['SK_ID_BUREAU'], inplace=True)
        merge_bureau = self._encode_and_aggregate_fit(merge_bureau, group_name='bureau')

        # --- Nhóm previous_application ---
        merge_prev = previous_application_transform.merge(credit_card_balance_transform, on='SK_ID_PREV', how='left')
        merge_prev = merge_prev.merge(POS_CASH_balance_transform, on='SK_ID_PREV', how='left')
        merge_prev = merge_prev.merge(installments_payments_transform, on='SK_ID_PREV', how='left')
        merge_prev.drop(columns=['SK_ID_PREV'], inplace=True)
        merge_prev = self._encode_and_aggregate_fit(merge_prev, group_name='previous_application')

        # --- Merge cuối cùng ---
        final_dataset = application_train_transform.merge(merge_prev, on='SK_ID_CURR', how='left')
        final_dataset = final_dataset.merge(merge_bureau, on='SK_ID_CURR', how='left')
        # self.final_dataset_ = final_dataset

        return self

    def transform(self, bureau_balance:pd.DataFrame, POS_CASH_balance:pd.DataFrame, application_train:pd.DataFrame,
                   credit_card_balance:pd.DataFrame, previous_application:pd.DataFrame, bureau:pd.DataFrame, installments_payments:pd.DataFrame):

        bureau_balance_transform = self.bureau_balance_pipeline.transform(bureau_balance)
        installments_payments_transform = self.installment_payment_pipeline.transform(installments_payments)
        POS_CASH_balance_transform = self.pos_cash_pipeline.transform(POS_CASH_balance)
        credit_card_balance_transform = self.credit_card_pipeline.transform(credit_card_balance)
        previous_application_transform = self.previous_application_pipeline.transform(previous_application)
        bureau_transform = self.bureau_pipeline.transform(bureau)
        application_train_transform = self.application_pipeline.transform(application_train)

        merge_bureau = bureau_transform.merge(bureau_balance_transform, on='SK_ID_BUREAU', how='left')
        merge_bureau.drop(columns=['SK_ID_BUREAU'], inplace=True)
        merge_bureau = self._encode_and_aggregate_transform(merge_bureau, group_name='bureau')

        merge_prev = previous_application_transform.merge(credit_card_balance_transform, on='SK_ID_PREV', how='left')
        merge_prev = merge_prev.merge(POS_CASH_balance_transform, on='SK_ID_PREV', how='left')
        merge_prev = merge_prev.merge(installments_payments_transform, on='SK_ID_PREV', how='left')
        merge_prev.drop(columns=['SK_ID_PREV'], inplace=True)
        merge_prev = self._encode_and_aggregate_transform(merge_prev, group_name='previous_application')

        final_dataset = application_train_transform.merge(merge_prev, on='SK_ID_CURR', how='left')
        final_dataset = final_dataset.merge(merge_bureau, on='SK_ID_CURR', how='left')

        return final_dataset