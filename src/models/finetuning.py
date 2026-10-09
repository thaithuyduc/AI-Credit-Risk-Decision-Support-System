"""
This module contains classes to finetune machine learning models using grid search and random search cross-validation.
The `GridSearch` class provides methods to perform hyperparameter tuning for a given estimator using either grid search or random search cross-validation.
It allows for the specification of scoring metrics, cross-validation strategies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold, LeaveOneOut, TimeSeriesSplit, PredefinedSplit, train_test_split
from typing import Literal

class GridSearch:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def get_results(self, cv_results:dict, best_index:np.int64, scoring:dict) -> pd.DataFrame:

        # Get all the results from the CV and put them in a df
        cv_results = pd.DataFrame(cv_results)
    
        # Isolate the row of the df with the max(mean f1 score)
        best_estimator_results = cv_results.iloc[best_index]
    
        # Extract accuracy, precision, recall, and f1 score from that row        
        table = {score: [] for score in scoring}
        table['type'] = ['train', 'test']
        
        for score in scoring:
            for prefix in ['train', 'test']:
                table[score].append(best_estimator_results[f'mean_{prefix}_{score}'])
                
        # Create table of results
        table = pd.DataFrame(table)
        index_of_test = table[table['type']=='test'].index
        table.loc[index_of_test, 'type'] = 'validate'
        
        return table
    
    def grid_search_cv(self, estimator, params_grid:dict, scoring:dict, 
                        cv: Literal['KFold', 'StratifiedKFold', 'Predefined', 'TimeSeries'],
                       refit:str, fit_params:dict|None=None) -> tuple[dict, pd.DataFrame]:
        
        if cv == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        
        elif cv == 'KFold':
            cv = KFold(n_splits=3, random_state=42, shuffle=True)
        
        elif cv == 'LeaveOneOut':
            cv = LeaveOneOut()
        
        elif cv == 'TimeSeries':
            cv = TimeSeriesSplit(n_splits=5)
        
        elif cv == 'Predefined':
            X_train, X_val, y_train, y_val = train_test_split(self.x, self.y, test_size=0.2, 
                                                          stratify=self.y, random_state=42)
            X_val: pd.DataFrame

            split_index = [0 if x in X_val.index else -1 for x in self.x.index]
            cv = PredefinedSplit(split_index)
        
        search = GridSearchCV(estimator=estimator, param_grid=params_grid, scoring=scoring, 
                              cv=cv, refit=refit, n_jobs=-1, return_train_score=True)
        
        if fit_params is None:
            fit_params = {}
        
        search.fit(self.x, self.y, **fit_params)
        best_index = search.best_index_
        cv_results = search.cv_results_
        
        table = self.get_results(cv_results, best_index, scoring)
        
        best_model = {'best_estimator': search.best_estimator_,
                    'best_params': search.best_params_,}
        
        return best_model, table
    
    def random_search_cv(self, estimator, param_distributions:dict, scoring:dict,
                         cv:Literal['KFold', 'StratifiedKFold', 'Predefined', 'TimeSeries'], 
                         refit:str, fit_params:dict|None=None) -> tuple[dict, pd.DataFrame]:
        
        if cv == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        
        elif cv == 'KFold':
            cv = KFold(n_splits=3, random_state=42, shuffle=True)

        elif cv == 'LeaveOneOut':
            cv = LeaveOneOut()
        
        elif cv == 'TimeSeries':
            cv = TimeSeriesSplit(n_splits=5)
        
        elif cv == 'Predefined':
            X_train, X_val, y_train, y_val = train_test_split(self.x, self.y, test_size=0.2, 
                                                          stratify=self.y, random_state=42)
            
            split_index = [0 if x in X_val.index else -1 for x in self.x.index]
            cv = PredefinedSplit(split_index)

        search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, 
                                    n_iter=10, n_jobs=-1, return_train_score=True,
                                    scoring=scoring, cv=cv, refit=refit)

        if fit_params is None:
            fit_params = {}
        
        search.fit(self.x, self.y, **fit_params)
        best_index = search.best_index_
        cv_results = search.cv_results_
        
        table = self.get_results(cv_results, best_index, scoring)
        
        best_model = {'best_estimator': search.best_estimator_,
                     'best_params': search.best_params_,}
        
        return best_model, table