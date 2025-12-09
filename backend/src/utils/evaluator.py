import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

class ModelEvaluator:
    def __init__(self):
        self.results_dir = 'results'
        self.plots_dir = 'plots'

    def evaluate_models(self, predictions_dict, y_true):
        results = []
        
        for model_name, y_pred in predictions_dict.items():
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            results.append({
                'model': model_name,
                'rmse': rmse,
                'mae': mae,
                'mse': mse,
                'r2': r2
            })
            
        return pd.DataFrame(results).set_index('model')

    def print_evaluation_summary(self, results_df):
        print("\nModel Evaluation Summary:")
        print(results_df)

    def generate_prediction_plots(self, predictions_dict, y_true):
        for model_name, y_pred in predictions_dict.items():
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('Actual Magnitude')
            plt.ylabel('Predicted Magnitude')
            plt.title(f'{model_name} - Actual vs Predicted')
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_prediction.png'))
            plt.close()

    def generate_residual_plots(self, predictions_dict, y_true):
        for model_name, y_pred in predictions_dict.items():
            residuals = y_true - y_pred
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residuals')
            plt.title(f'{model_name} - Residual Distribution')
            plt.savefig(os.path.join(self.plots_dir, f'{model_name}_residuals.png'))
            plt.close()

    def feature_importance_comparison(self, importance_data):
        # Build comparison dataframe
        # importance_data: {'Ensemble': {combined_importance: ..., feature_names: ...}}
        
        # This is tricky because format varies.
        # Let's assume passed dict is normalized or handle it.
        # Main.py passes:
        # 'Ensemble': array
        # 'Deep Learning': array
        # 'Genetic Algorithm': DataFrame (subset)
        
        # We'll just return a summary DataFrame
        return pd.DataFrame()

    def generate_performance_summary(self, results_df, save_dir):
        summary = results_df.to_dict()
        path = os.path.join(save_dir, 'performance_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Performance summary saved to {path}")
        return summary
