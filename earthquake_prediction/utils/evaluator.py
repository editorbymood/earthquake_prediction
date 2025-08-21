"""
Model evaluation utilities for earthquake magnitude prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import EVAL_CONFIG
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation for earthquake magnitude prediction
    """
    
    def __init__(self, config=None):
        """
        Initialize the evaluator
        
        Args:
            config (dict): Evaluation configuration
        """
        self.config = config or EVAL_CONFIG
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Calculated metrics
        """
        metrics = {
            'model': model_name,
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': self._mean_absolute_percentage_error(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred)),
            'std_error': np.std(y_true - y_pred)
        }
        
        return metrics
    
    def _mean_absolute_percentage_error(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        # Avoid division by zero
        y_true_safe = np.where(y_true == 0, 1e-8, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    def cross_validate_model(self, model, X, y, cv=5, model_name="Model"):
        """
        Perform cross-validation evaluation
        
        Args:
            model: Trained model with predict method
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            cv (int): Number of cross-validation folds
            model_name (str): Name of the model
            
        Returns:
            dict: Cross-validation results
        """
        # Note: This is a simplified approach
        # For ensemble models, proper nested CV would be more appropriate
        
        scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        cv_results = {'model': model_name}
        
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            
            if 'neg_' in metric:
                scores = -scores  # Convert negative scores to positive
                metric = metric.replace('neg_', '')
            
            cv_results[f'{metric}_scores'] = scores
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()
        
        return cv_results
    
    def evaluate_models(self, models_predictions, y_true):
        """
        Evaluate multiple models and compare performance
        
        Args:
            models_predictions (dict): Dictionary with model names as keys and predictions as values
            y_true (array-like): True target values
            
        Returns:
            pd.DataFrame: Comparison of model performances
        """
        evaluation_results = []
        
        for model_name, y_pred in models_predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            evaluation_results.append(metrics)
        
        results_df = pd.DataFrame(evaluation_results)
        
        # Store results
        self.results['model_comparison'] = results_df
        
        return results_df
    
    def generate_prediction_plots(self, models_predictions, y_true, save_dir=None):
        """
        Generate prediction vs actual plots for models
        
        Args:
            models_predictions (dict): Model predictions
            y_true (array-like): True values
            save_dir (str, optional): Directory to save plots
            
        Returns:
            dict: Generated plot information
        """
        n_models = len(models_predictions)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        plot_info = {}
        
        for i, (model_name, y_pred) in enumerate(models_predictions.items()):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Labels and title
            ax.set_xlabel('True Magnitude')
            ax.set_ylabel('Predicted Magnitude')
            ax.set_title(f'{model_name} - Predictions vs Actual')
            
            # Add R² score to plot
            r2 = r2_score(y_true, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Make it square
            ax.set_aspect('equal', adjustable='box')
            
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'prediction_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_info['prediction_plot_path'] = plot_path
        
        plt.show()
        
        return plot_info
    
    def generate_residual_plots(self, models_predictions, y_true, save_dir=None):
        """
        Generate residual plots for model evaluation
        
        Args:
            models_predictions (dict): Model predictions
            y_true (array-like): True values
            save_dir (str, optional): Directory to save plots
        """
        n_models = len(models_predictions)
        
        # Create subplots for residuals
        fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, y_pred) in enumerate(models_predictions.items()):
            residuals = y_true - y_pred
            
            # Residuals vs Predicted
            axes[0, i].scatter(y_pred, residuals, alpha=0.6, s=20)
            axes[0, i].axhline(y=0, color='r', linestyle='--')
            axes[0, i].set_xlabel('Predicted Magnitude')
            axes[0, i].set_ylabel('Residuals')
            axes[0, i].set_title(f'{model_name} - Residuals vs Predicted')
            
            # Residuals histogram
            axes[1, i].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, i].axvline(x=0, color='r', linestyle='--')
            axes[1, i].set_xlabel('Residuals')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].set_title(f'{model_name} - Residuals Distribution')
        
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'residual_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def feature_importance_comparison(self, importance_data, save_dir=None):
        """
        Compare feature importance across models
        
        Args:
            importance_data (dict): Dictionary with model names and their feature importance DataFrames
            save_dir (str, optional): Directory to save plots
        """
        # Find common features across all models
        all_features = set()
        for model_importance in importance_data.values():
            all_features.update(model_importance['feature'].values)
        
        # Create comparison DataFrame
        comparison_data = []
        for feature in all_features:
            feature_row = {'feature': feature}
            
            for model_name, importance_df in importance_data.items():
                # Get importance for this feature (0 if not present)
                feature_importance = importance_df[
                    importance_df['feature'] == feature
                ]['importance'].values
                
                importance_value = feature_importance[0] if len(feature_importance) > 0 else 0
                feature_row[model_name] = importance_value
            
            comparison_data.append(feature_row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Get top 15 features by average importance
        model_cols = [col for col in comparison_df.columns if col != 'feature']
        comparison_df['avg_importance'] = comparison_df[model_cols].mean(axis=1)
        top_features = comparison_df.nlargest(15, 'avg_importance')
        
        # Create heatmap
        heatmap_data = top_features.set_index('feature')[model_cols]
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', cbar=True)
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Models')
        plt.ylabel('Features')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'feature_importance_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return comparison_df
    
    def generate_performance_summary(self, results_df, save_dir=None):
        """
        Generate a comprehensive performance summary
        
        Args:
            results_df (pd.DataFrame): Model evaluation results
            save_dir (str, optional): Directory to save results
            
        Returns:
            dict: Performance summary
        """
        summary = {
            'best_model_by_metric': {},
            'model_rankings': {},
            'performance_summary': results_df.to_dict('records')
        }
        
        # Find best model for each metric (lower is better for error metrics)
        error_metrics = ['mae', 'mse', 'rmse', 'mape', 'max_error']
        performance_metrics = ['r2']  # Higher is better
        
        for metric in error_metrics:
            best_idx = results_df[metric].idxmin()
            summary['best_model_by_metric'][metric] = {
                'model': results_df.loc[best_idx, 'model'],
                'value': results_df.loc[best_idx, metric]
            }
        
        for metric in performance_metrics:
            best_idx = results_df[metric].idxmax()
            summary['best_model_by_metric'][metric] = {
                'model': results_df.loc[best_idx, 'model'],
                'value': results_df.loc[best_idx, metric]
            }
        
        # Create overall ranking (based on RMSE as primary metric)
        ranking = results_df.sort_values('rmse')[['model', 'rmse', 'mae', 'r2']]
        summary['model_rankings'] = ranking.to_dict('records')
        
        # Save summary if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save detailed results
            results_path = os.path.join(save_dir, 'evaluation_results.csv')
            results_df.to_csv(results_path, index=False)
            
            # Save summary
            import json
            summary_path = os.path.join(save_dir, 'performance_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def print_evaluation_summary(self, results_df):
        """
        Print a formatted evaluation summary
        
        Args:
            results_df (pd.DataFrame): Model evaluation results
        """
        print("\\n" + "="*70)
        print("MODEL EVALUATION SUMMARY")
        print("="*70)
        
        # Model comparison table
        print("\\nPerformance Comparison:")
        print("-" * 70)
        
        # Format and display key metrics
        display_cols = ['model', 'rmse', 'mae', 'r2', 'mape']
        display_df = results_df[display_cols].copy()
        
        # Round numerical columns
        for col in display_df.columns:
            if col != 'model':
                display_df[col] = display_df[col].round(4)
        
        print(display_df.to_string(index=False))
        
        # Best models
        print("\\n\\nBest Models by Metric:")
        print("-" * 70)
        
        best_rmse = results_df.loc[results_df['rmse'].idxmin()]
        best_r2 = results_df.loc[results_df['r2'].idxmax()]
        
        print(f"Best RMSE: {best_rmse['model']} ({best_rmse['rmse']:.4f})")
        print(f"Best R²:   {best_r2['model']} ({best_r2['r2']:.4f})")
        print(f"Best MAE:  {results_df.loc[results_df['mae'].idxmin(), 'model']} ({results_df['mae'].min():.4f})")


def create_evaluator(config=None):
    """
    Factory function to create a model evaluator
    
    Args:
        config (dict): Evaluation configuration
        
    Returns:
        ModelEvaluator: Configured evaluator
    """
    return ModelEvaluator(config)
