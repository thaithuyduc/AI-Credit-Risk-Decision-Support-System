"""
This module provides a class for local interpretation of machine learning models using SHAP (SHapley Additive exPlanations) values. 
The `LocalInterpretation` class allows for the analysis of individual predictions by computing SHAP values for the features of a given input instance. 
It provides insights into how each feature contributes to the model's prediction, enabling a better understanding of the model's decision-making process. 
The class also includes methods to retrieve the top contributing features along with their SHAP values, the baseline value, and the predicted probability for the positive class.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import shap

class LocalInterpretation:
    def __init__(self, background_data:pd.DataFrame, estimator, optimal_threshold:float=0.5):
        self.optimal_threshold = optimal_threshold
        if isinstance(estimator, CalibratedClassifierCV):
            self.model = estimator
            
            def predict_proba_positive(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=background_data.columns)
                return self.model.predict_proba(X)[:, 1]

            summary_background_data = shap.utils.sample(background_data, 100, random_state=42)
            self.explainer = shap.KernelExplainer(predict_proba_positive, summary_background_data, link="identity")
            
        elif isinstance(estimator, Pipeline):
            self.model = estimator[-1]
            self.transformer = estimator[:-1]
            
            background_transformed = self.transformer.transform(background_data)
            self.explainer = shap.TreeExplainer(model=self.model, model_output="probability", feature_perturbation='interventional', data=background_transformed)
        
        else:
            raise TypeError("estimator phải là Pipeline hoặc CalibratedClassifierCV")
        
        self.baseline_value = self.explainer.expected_value
        
    def get_top_features_with_shap_values(self, X_raw:pd.DataFrame, top_features:int=10):
        # X_raw is DataFrame that only has one row
        np.random.seed(42)
        if isinstance(self.explainer, shap.KernelExplainer):
            shap_values = self.explainer.shap_values(X_raw, nsamples=500)
            prob_positive = self.model.predict_proba(X_raw)[:, 1][0]
            shap_df = pd.DataFrame({'feature': X_raw.columns, 'shap_value': shap_values[0], 'abs_shap': np.abs(shap_values[0])})

        elif isinstance(self.explainer, shap.TreeExplainer):
            X_transformed = self.transformer.transform(X_raw)
            shap_values = self.explainer.shap_values(X_transformed)
            prob_positive = self.model.predict_proba(X_transformed)[:, 1][0]
            shap_df = pd.DataFrame({'feature': X_transformed.columns, 'shap_value': shap_values[0], 'abs_shap': np.abs(shap_values[0])})
        
        late_payment = 'True' if prob_positive >= self.optimal_threshold else 'False'
        top = shap_df.sort_values('abs_shap', ascending=False).head(top_features).reset_index(drop=True)

        top_features = top['feature'].values
        top_shap_values = np.round(top['shap_value'].values, 4)
        top_features_dict = dict(zip(top_features, top_shap_values))
        
        result = {
            'baseline value': np.round(self.baseline_value, 4),
            'probability for positive class': np.round(prob_positive, 4),
            'late payment?': late_payment,
            'top feature attribution': top_features_dict
        }

        return result