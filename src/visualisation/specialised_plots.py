"""
This module contains specialised plots for visualising model performance, particularly calibration curves for binary classification models. 
- The `CalibrationPlot` class provides methods to compute and plot calibration curves, as well as calculate metrics such as Brier score, log loss, and Expected Calibration Error (ECE).
- The `ConfusionMatrix` class provides methods to compute and plot confusion matrices for binary classification models, allowing for visual assessment of model performance in terms of true positives, false positives, true negatives, and false negatives.
- The `PrecisionRecallCurve` class provides methods to compute and plot precision-recall curves for possible thresholds, along with F-beta scores, for binary classification models.
- The `ROCCurve` class provides methods to compute and plot ROC curves for binary classification models, along with Youden's J statistic at the optimal decision threshold.
- The `KSStatistic` class provides methods to compute and plot the empirical cumulative distribution functions (ECDFs) for positive and negative classes, along with the Kolmogorov-Smirnov (KS) statistic at the optimal decision threshold.

These visualisations are useful for evaluating the reliability of predicted probabilities and the overall performance of classification models
"""

import math
from sklearn.metrics import brier_score_loss, log_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import itertools
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve

class CalibrationPlot:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def expected_calibration_error(self, y_true, y_prob, n_bins: int = 15) -> float:
        """
        Classic binned Expected Calibration Error (ECE).
        Fully self-contained - does not rely on calibration_curve output.
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
    
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
    
        # Important: values exactly equal to 1.0 would go to bin n_bins → clip them
        binids = np.clip(binids, 0, n_bins - 1)
    
        bin_counts = np.bincount(binids, minlength=n_bins)
        bin_true   = np.bincount(binids, weights=y_true, minlength=n_bins)
        bin_prob   = np.bincount(binids, weights=y_prob, minlength=n_bins)
    
        non_empty = bin_counts > 0
    
        # Avoid division by zero
        avg_true = np.zeros(n_bins)
        avg_prob = np.zeros(n_bins)
        avg_true[non_empty] = bin_true[non_empty] / bin_counts[non_empty]
        avg_prob[non_empty] = bin_prob[non_empty] / bin_counts[non_empty]
    
        weights = bin_counts[non_empty] / len(y_prob)
        ece = np.sum(weights * np.abs(avg_true[non_empty] - avg_prob[non_empty]))
        return float(ece)
    
    def plot_calibration_curve(self, estimators_dict:dict, plots_each_row:int):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 12), ) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))

        for coor, (estimator_name, estimator) in zip(coordinates, estimators_dict.items()):
            n_bins=15
            
            y_prob = estimator.predict_proba(self.x)[:, 1] # predicted probabilities for the positive class (class 1)
            prob_true, prob_pred = calibration_curve(self.y, y_prob, n_bins=n_bins, strategy='uniform')
            
            brier = brier_score_loss(self.y, y_prob)
            logloss = log_loss(self.y, y_prob)
            ece = self.expected_calibration_error(prob_true, prob_pred, y_prob, n_bins)

            axes[coor].plot(prob_pred, prob_true, marker='o', label=f'{estimator_name}\'s calibration')
            axes[coor].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
            axes[coor].set_xlabel("Mean Predicted Probability", fontsize=11)
            axes[coor].set_ylabel("True Positive Rate", fontsize=11)
            axes[coor].set_title(f"{estimator_name}\n Brier: {brier:.3f} | Log Loss: {logloss:.3f} | ECE: {ece:.3f}", fontsize=13, pad=12)
            axes[coor].legend(loc="best")
            # axes[coor].set_ylim([0, 1])
            axes[coor].spines["top"].set_visible(False)
            axes[coor].spines["right"].set_visible(False)
            axes[coor].grid()

        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor])

        plt.tight_layout()


class ConfusionMatrix:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def plot_confusion_matrix(self, estimators_dict:dict, plots_each_row:int, threshold:float=0.5):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8), ) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
        my_cmap = 'viridis'
        
        for coor, (estimator_name, estimator) in zip(coordinates, estimators_dict.items()):
            y_prob = estimator.predict_proba(self.x)[:, 1] # get probabilities for positive class
            y_pred = (y_prob >= threshold).astype(int)
            
            cm = confusion_matrix(self.y, y_pred,  labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['Negative', 'Positive'])
          
            disp.plot(values_format='', ax=axes[coor], text_kw={'size': 20, 'weight':'bold'},
                     cmap=my_cmap) # `values_format=''` suppresses scientific notation
            axes[coor].set_title(f'{estimator_name}', fontsize=16)

        plt.tight_layout()
        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor])

        # return fig, axes


class PrecisionRecallCurve:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def plot_precision_recall_with_thresholds(self, estimators_dict:dict, plots_each_row:int):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8),) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
        optimal_threshold = 0.3441317362531742

        for coor, (estimator_name, estimator) in zip(coordinates, estimators_dict.items()):
            y_score = estimator.predict_proba(self.x)[:, 1] # certainty measure for the positive class (class 1)
            precision, recall, thresholds = precision_recall_curve(self.y, y_score)

            # calculate F-beta (beta = 2)
            beta = 2
            beta2 = beta ** 2
            fbeta = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
            
            axes[coor].axvline(x=0.5, color="red", linestyle=":", linewidth=2, label=f"{estimator_name}'s Default Threshold")
            axes[coor].axvline(x=optimal_threshold, color="orange", linestyle="--", linewidth=2, label=f"{estimator_name}'s Optimised Threshold")
            
            axes[coor].plot(thresholds, precision[:-1], "b--", label="Precision")
            axes[coor].plot(thresholds, recall[:-1], "g-", label="Recall")
            axes[coor].plot(thresholds, fbeta[:-1], "m-", linewidth=2, label="F-beta (β=2)")
            axes[coor].set_xlabel("Threshold")
            axes[coor].set_title(estimator_name)
            axes[coor].legend(loc="best")
            axes[coor].set_ylim([0, 1])
            axes[coor].grid()

        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor])

        return fig, axes


class ROCCurve:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y
    
    def plot_roc_curve_with_youden_j(self, estimators_dict:dict, plots_each_row:int):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8),) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
        optimal_threshold = 0.3441317362531742
        
        for coor, (estimator_name, estimator) in zip(coordinates, estimators_dict.items()):
            y_score = estimator.predict_proba(self.x)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y, y_score)
            
            close_optimal = np.argmin(np.abs(thresholds - optimal_threshold))
            fpr_optimal = fpr[close_optimal]
            tpr_optimal = tpr[close_optimal]

            # calculate Youden'J statistic
            j = tpr_optimal - fpr_optimal

            axes[coor].vlines(x=fpr_optimal, ymin=fpr_optimal, ymax=tpr_optimal, color="black", linestyle="-", linewidth=2.5, label=f"Youden's J = {j:.3f}")
            axes[coor].plot(fpr, tpr, label=f"{estimator_name}'s ROC Curve")
            axes[coor].set_title('Youden\'J Statistic at Optimal Decision Threshold')
            axes[coor].set_xlabel("FPR")
            axes[coor].set_ylabel("TPR (recall)")
            axes[coor].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "b--", label=f"Chance Diagonal")
            axes[coor].legend(loc='best')
            axes[coor].grid()

        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor])

        return fig, axes


class KSStatistic:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def plot_ecdf_with_ks_statistic(self, estimators_dict:dict, plots_each_row:int):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8),) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))
        optimal_threshold = 0.3441317362531742

        for coor, (estimator_name, estimator) in zip(coordinates, estimators_dict.items()):
            y_prob = estimator.predict_proba(self.x)[:, 1] # get probabilities for positive class
            prob_for_positive = y_prob[self.y == 1]
            prob_for_negative = y_prob[self.y == 0]

            scores_pos_sorted = np.sort(prob_for_positive)
            scores_neg_sorted = np.sort(prob_for_negative)

            ecdf_pos = np.arange(1, len(scores_pos_sorted)+1) / len(scores_pos_sorted)
            ecdf_neg = np.arange(1, len(scores_neg_sorted)+1) / len(scores_neg_sorted)
            
            _, _, thresholds = roc_curve(self.y, y_prob)
            
            ecdf_pos_at_th = np.searchsorted(scores_pos_sorted, thresholds, side='right') / len(scores_pos_sorted)
            ecdf_neg_at_th = np.searchsorted(scores_neg_sorted, thresholds, side='right') / len(scores_neg_sorted)

            diff = np.abs(ecdf_pos_at_th - ecdf_neg_at_th)

            # used when already have optimal_threshold and find ks-statistic corresponding to this threshold 
            optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            optimal_diff = diff[optimal_idx]
            # optimal_threshold = thresholds[optimal_idx]

            # used when finding optimal threshold that can maximise ks-statistic
            # ks_statistic = np.max(diff)
            # ks_index = np.argmax(diff)
            # ks_threshold = thresholds[ks_index]

            # ECDF lines (step functions)
            axes[coor].step(scores_pos_sorted, ecdf_pos, where='post', color='tab:orange', linewidth=2, label='ECDF Positive')
            axes[coor].step(scores_neg_sorted, ecdf_neg, where='post', color='tab:blue', linewidth=2, label='ECDF Negative')

            # Vertical line at KS threshold
            axes[coor].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'KS threshold = {optimal_threshold:.3f}')
            axes[coor].vlines(x=optimal_threshold, ymin=min(ecdf_pos_at_th[optimal_idx], ecdf_neg_at_th[optimal_idx]),\
                      ymax=max(ecdf_pos_at_th[optimal_idx], ecdf_neg_at_th[optimal_idx]), colors='purple', linewidth=3, label=f'KS = {optimal_diff:.3f}')

            axes[coor].set_xlabel("Predicted Probability")
            axes[coor].set_ylabel("Cumulative Probability (ECDF)")
            axes[coor].set_title("Kolmogorov-Smirnov Statistic")
            axes[coor].legend(loc='best')
            axes[coor].grid(True, alpha=0.3)
        
        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor]) 

        return fig, axes