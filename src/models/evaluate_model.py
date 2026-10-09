"""
This module contains the `TestPerformance` class, which is designed to evaluate the performance of machine learning models. It provides methods to compute various performance metrics based on predicted probabilities or binary predictions. The class can handle multiple models and metrics, allowing for flexible evaluation of model performance.
The `model_performance` method computes the specified metrics for each model in the provided dictionary of estimators, returning the results in a pandas DataFrame for easy analysis and comparison.
"""

import pandas as pd
from sklearn.metrics import f1_score

class TestPerformance:
    def __init__(self, X: pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def model_performance(self, estimators_dict:dict, use_prob:bool, threshold:float=0.5, metrics_fn:dict[str, tuple]={'F1 Score':(f1_score, {})}):
        """
        use_prob: True if metrics_fn requires estimated probabilities, False otherwise
        threshold: only used when use_prob = False
        """

        # Initialize result container
        data = {
            'Model': list(estimators_dict.keys())
        }
    
        # Initialize metric columns
        for metric_name in metrics_fn:
            data[metric_name] = []
    
        # Compute predictions & metrics
        for name, estimator in estimators_dict.items():
            y_prob_positive = estimator.predict_proba(self.x)[:, 1]
            
            if use_prob == True:
                y = y_prob_positive
            else:
                y_pred = (y_prob_positive >= threshold).astype(int)
                y = y_pred
    
            for metric_name, (metric_fn, kwargs) in metrics_fn.items():
                score = metric_fn(self.y, y, **kwargs)
                data[metric_name].append(score)

        return pd.DataFrame(data)