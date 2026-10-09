"""
This module contains utility functions and classes for optimising decision thresholds in binary classification models based on business costs. 
The `DecisionThresholdOptimisation` class provides methods to compute and visualise the total business cost associated with different decision thresholds for multiple models. 
It allows for the identification of the optimal decision threshold that minimises the total business cost, taking into account the costs of false positives and false negatives.
"""

from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import math
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from typing import Literal
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.frozen import FrozenEstimator

class DecisionThresholdOptimisation:
    def __init__(self, X:pd.DataFrame, y:pd.Series):
        self.x = X
        self.y = y

    def plot_based_on_business_cost(self, estimators_dict:dict, plots_each_row:int):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 8), ) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))

        C_FP = 1.0   # Chi phí từ chối nhầm khách tốt (chi phí cơ hội)
        C_FN = 10.0  # Chi phí duyệt nhầm khách vỡ nợ (mất vốn)
        
        for coor, (estimator_name, estimator) in zip(coordinates, estimators_dict.items()):
            total_costs = []
            y_prob = estimator.predict_proba(self.x)[:, 1]
            _, _, thresholds = roc_curve(self.y, y_prob)

            for t in thresholds:
              # Phân lớp dựa trên threshold hiện tại
              y_pred_t = (y_prob >= t).astype(int)
            
              # Lấy các thành phần từ Confusion Matrix
              tn, fp, fn, tp = confusion_matrix(self.y, y_pred_t).ravel()
            
              # Tính tổng chi phí
              cost = (fp * C_FP) + (fn * C_FN)
              total_costs.append(cost)

            # Tìm threshold và cost cho mức chi phí thấp nhất
            optimal_idx = np.argmin(total_costs)
            optimal_threshold = thresholds[optimal_idx]
            optimal_cost = total_costs[optimal_idx]

            print(f"{estimator_name} - Optimised Decision Threshold: {optimal_threshold} - Optimal Cost: {optimal_cost}")
            
            axes[coor].plot(thresholds, total_costs, label="Total Cost", color="navy", linewidth=2,)
            axes[coor].set_xlabel("Decision Threshold", fontsize=12)
            axes[coor].set_ylabel("Total Business Cost", fontsize=12)
            axes[coor].legend(loc='best')
            
            y_offset = (max(total_costs) - min(total_costs)) * 0.15
            x_offset = 0.05 if optimal_threshold < 0.7 else -0.2
            
            axes[coor].annotate(
                f"({optimal_threshold:.5f}, {optimal_cost:.1f})",
                xy=(optimal_threshold, optimal_cost),  # Tọa độ mũi tên trỏ tới (x, y)
                xytext=(
                    optimal_threshold + x_offset,
                    optimal_cost + y_offset,
                ),  # Tọa độ đặt chữ
                arrowprops=dict(
                    facecolor="red", edgecolor="red", shrink=0.05, width=1.5, headwidth=8
                ),
                fontsize=11,
                fontweight="bold",
                color="darkred",
            )
            
            axes[coor].set_title(f'Total Cost by {estimator_name}\'s Decision Threshold\n C_FP = 1 | C_FN = 10', fontsize=16)

        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.6)
        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor])

        # return fig, axes


class ImproveModelCalibration:
    def __init__(self, X_val:pd.DataFrame, y_val:pd.Series, X_test:pd.DataFrame, y_test:pd.Series,
                 method:Literal['sigmoid', 'isotonic', 'temperature']):
        self.x_val = X_val
        self.y_val = y_val
        self.x_test = X_test
        self.y_test = y_test
        self.method = method

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
    
    def calibration_plot_before_after_calibrating(self, estimators_dict:dict, plots_each_row:int):
        num_plots = len(estimators_dict)
        number_rows = math.ceil(num_plots / plots_each_row)
        
        fig, axes = plt.subplots(nrows=number_rows, ncols=plots_each_row, figsize=(16, 12), ) # sharey=True
        axes = np.array(axes).reshape(number_rows, plots_each_row)
        coordinates = list(itertools.product(range(number_rows), range(plots_each_row)))

        calibrated_estimators_dict = {}

        for coor, (estimator_name, base_estimator) in zip(coordinates, estimators_dict.items()):
            n_bins=15

            # estimated probabilites for positive class from base estimator
            y_prob_base = base_estimator.predict_proba(self.x_test)[:, 1] # predicted probabilities for the positive class (class 1)
            prob_true_base, prob_pred_base = calibration_curve(self.y_test, y_prob_base, n_bins=n_bins, strategy='uniform')
            
            # estimated probabilites for positive class from calibrated estimator
            calibrated_estimator = CalibratedClassifierCV(
                estimator=FrozenEstimator(base_estimator), #FrozenEstimator is important when we have fine-tuned model and just want to improve its calibration
                method=self.method,
                n_jobs=-1,
                cv=None,
            )
            calibrated_estimator.fit(self.x_val, self.y_val)
            calibrated_estimators_dict[f'Calibrated {estimator_name}'] = calibrated_estimator
            
            y_prob_calibrated = calibrated_estimator.predict_proba(self.x_test)[:, 1]
            prob_true_calibrated, prob_pred_calibrated = calibration_curve(self.y_test, y_prob_calibrated, n_bins=n_bins, strategy='uniform')

            # compute metrics using base estimator
            brier_base = brier_score_loss(self.y_test, y_prob_base)
            logloss_base = log_loss(self.y_test, y_prob_base)
            ece_base = self.expected_calibration_error(self.y_test, y_prob_base, n_bins)

            # compute metrics using calibrated estimator
            brier_calibrated = brier_score_loss(self.y_test, y_prob_calibrated)
            logloss_calibrated = log_loss(self.y_test, y_prob_calibrated)
            ece_calibrated = self.expected_calibration_error(self.y_test, y_prob_calibrated, n_bins)            

            axes[coor].plot(prob_pred_base, prob_true_base, color='cyan', marker='o', label=f'{estimator_name}\'s calibration before calibrating')
            axes[coor].plot(prob_pred_calibrated, prob_true_calibrated, color='red', marker='s', label=f'{estimator_name}\'s calibration after calibrating')
            axes[coor].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated', linewidth=1.8)
            axes[coor].set_xlabel("Mean Predicted Probability", fontsize=12)
            axes[coor].set_ylabel("True Positive Rate", fontsize=12)
            
            # Keep a short title
            axes[coor].set_title(estimator_name, fontsize=16, pad=12)
            
            # Metrics text box
            metrics_text = (
                f"Before Calibrating -> Brier: {brier_base:.3f} | LogLoss: {logloss_base:.3f} | ECE: {ece_base:.3f}\n"
                f"After Calibrating -> Brier: {brier_calibrated:.3f} | LogLoss: {logloss_calibrated:.3f} | ECE: {ece_calibrated:.3f}"
            )

            axes[coor].text(
                0.05, 0.95, metrics_text,
                transform=axes[coor].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            )
            
            axes[coor].legend(loc="best", fontsize=14)
            # axes[coor].set_ylim([0, 1])
            axes[coor].spines["top"].set_visible(False)
            axes[coor].spines["right"].set_visible(False)
            axes[coor].grid()

        # hide any unused axes
        for coor in coordinates[num_plots:]:
            fig.delaxes(axes[coor])

        plt.tight_layout()

        return calibrated_estimators_dict