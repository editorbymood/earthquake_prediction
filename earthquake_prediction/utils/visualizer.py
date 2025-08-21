"""
Visualization utilities for earthquake magnitude prediction project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


class EarthquakeVisualizer:
    """
    Visualization utilities for earthquake magnitude prediction analysis
    """
    
    def __init__(self, style='seaborn'):
        """
        Initialize the visualizer
        
        Args:
            style (str): Matplotlib style to use
        """
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for plots
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_data_distribution(self, df, target_column='magnitude', save_path=None):
        """
        Plot data distribution and basic statistics
        
        Args:
            df (pd.DataFrame): Dataset to visualize
            target_column (str): Target column name
            save_path (str, optional): Path to save the plot
        """
        # Create a copy and preprocess for visualization
        df_viz = df.copy()
        
        # Handle datetime columns
        datetime_columns = ['timestamp'] if 'timestamp' in df_viz.columns else []
        for col in datetime_columns:
            if col in df_viz.columns:
                # Convert to datetime if it's a string
                if df_viz[col].dtype == 'object':
                    df_viz[col] = pd.to_datetime(df_viz[col])
                
                # Extract useful datetime features
                df_viz[f'{col}_year'] = df_viz[col].dt.year
                df_viz[f'{col}_month'] = df_viz[col].dt.month
                
                # Drop original datetime column
                df_viz = df_viz.drop(columns=[col])
        
        # Handle categorical columns
        categorical_columns = df_viz.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        
        for col in categorical_columns:
            # Use label encoding for categorical features
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_viz[col] = le.fit_transform(df_viz[col].astype(str))
        
        # Select only numeric columns for correlation analysis
        numeric_df = df_viz.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_df.columns if col != target_column]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target distribution
        axes[0, 0].hist(numeric_df[target_column], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Earthquake Magnitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Earthquake Magnitudes')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(numeric_df[target_column])
        axes[0, 1].set_ylabel('Earthquake Magnitude')
        axes[0, 1].set_title('Magnitude Distribution (Box Plot)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature correlation with target (top 10)
        if len(feature_cols) > 0:
            correlations = numeric_df[feature_cols + [target_column]].corr()[target_column].abs().sort_values(ascending=False)
            top_corr = correlations.head(11)[1:]  # Exclude self-correlation
            
            axes[1, 0].barh(range(len(top_corr)), top_corr.values)
            axes[1, 0].set_yticks(range(len(top_corr)))
            axes[1, 0].set_yticklabels(top_corr.index, fontsize=8)
            axes[1, 0].set_xlabel('Absolute Correlation with Magnitude')
            axes[1, 0].set_title('Top 10 Features by Correlation')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No numeric features\nfor correlation analysis', 
                           ha='center', va='center')
            axes[1, 0].set_title('Feature Correlations')
        
        # Dataset statistics
        stats_text = f"""Dataset Statistics:
Samples: {len(df):,}
Features: {len(df.columns)-1}
Numeric Features: {len(feature_cols)}
Target Range: {numeric_df[target_column].min():.2f} - {numeric_df[target_column].max():.2f}
Target Mean: {numeric_df[target_column].mean():.2f}
Target Std: {numeric_df[target_column].std():.2f}
Missing Values: {df.isnull().sum().sum()}"""
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Dataset Overview')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_genetic_algorithm_evolution(self, evolution_history, save_path=None):
        """
        Plot genetic algorithm evolution progress
        
        Args:
            evolution_history (list): GA evolution history
            save_path (str, optional): Path to save the plot
        """
        if not evolution_history:
            print("No evolution history available.")
            return
        
        generations = [gen['generation'] for gen in evolution_history]
        best_fitness = [gen['best_fitness'] for gen in evolution_history]
        avg_fitness = [gen['avg_fitness'] for gen in evolution_history]
        worst_fitness = [gen['worst_fitness'] for gen in evolution_history]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(generations, best_fitness, 'g-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitness, 'b--', label='Average Fitness', alpha=0.7)
        plt.plot(generations, worst_fitness, 'r:', label='Worst Fitness', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Fitness (MSE)')
        plt.title('Genetic Algorithm Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Improvement over generations
        plt.subplot(1, 2, 2)
        improvement = [(best_fitness[0] - f) / best_fitness[0] * 100 for f in best_fitness]
        plt.plot(generations, improvement, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Improvement (%)')
        plt.title('Fitness Improvement Over Generations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_selection_results(self, selected_features, all_features, importance_df=None, save_path=None):
        """
        Visualize feature selection results
        
        Args:
            selected_features (list): List of selected feature names
            all_features (list): List of all available features
            importance_df (pd.DataFrame, optional): Feature importance data
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature selection summary
        n_selected = len(selected_features)
        n_total = len(all_features)
        n_not_selected = n_total - n_selected
        
        # Pie chart
        axes[0].pie([n_selected, n_not_selected], 
                   labels=[f'Selected ({n_selected})', f'Not Selected ({n_not_selected})'],
                   autopct='%1.1f%%', startangle=90,
                   colors=['lightgreen', 'lightcoral'])
        axes[0].set_title('Feature Selection Results')
        
        # Selected features bar chart
        if importance_df is not None and 'importance' in importance_df.columns:
            # Show importance of selected features
            selected_importance = importance_df[
                importance_df['feature'].isin(selected_features)
            ].sort_values('importance', ascending=True)
            
            axes[1].barh(range(len(selected_importance)), selected_importance['importance'])
            axes[1].set_yticks(range(len(selected_importance)))
            axes[1].set_yticklabels(selected_importance['feature'], fontsize=8)
            axes[1].set_xlabel('Importance Score')
            axes[1].set_title('Importance of Selected Features')
        else:
            # Just show selected features
            axes[1].barh(range(len(selected_features)), [1] * len(selected_features))
            axes[1].set_yticks(range(len(selected_features)))
            axes[1].set_yticklabels(selected_features, fontsize=8)
            axes[1].set_xlabel('Selected')
            axes[1].set_title('Selected Features')
        
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, results_df, save_path=None):
        """
        Create comprehensive model comparison visualizations
        
        Args:
            results_df (pd.DataFrame): Model evaluation results
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        models = results_df['model'].values
        
        # RMSE comparison
        rmse_values = results_df['rmse'].values
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars1 = axes[0, 0].bar(models, rmse_values, color=colors)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Root Mean Square Error by Model')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # R² comparison
        r2_values = results_df['r2'].values
        bars2 = axes[0, 1].bar(models, r2_values, color=colors)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('R² Score by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, r2_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # MAE comparison
        mae_values = results_df['mae'].values
        bars3 = axes[1, 0].bar(models, mae_values, color=colors)
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Mean Absolute Error by Model')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars3, mae_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Multi-metric radar chart
        metrics = ['rmse', 'mae', 'r2']
        
        # Normalize metrics for radar chart (scale 0-1)
        normalized_data = {}
        for metric in metrics:
            values = results_df[metric].values
            if metric == 'r2':
                # For R², higher is better, so normalize as is
                normalized_data[metric] = values
            else:
                # For error metrics, lower is better, so invert
                normalized_data[metric] = 1 - (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(models):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            axes[1, 1].fill(angles, values, alpha=0.1, color=colors[i])
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(['RMSE (inv)', 'MAE (inv)', 'R²'])
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Multi-Metric Model Comparison\\n(Higher is Better)')
        axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history_data, save_path=None):
        """
        Plot deep learning model training history
        
        Args:
            history_data (dict): Training history data
            save_path (str, optional): Path to save the plot
        """
        if not history_data:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = history_data['epochs']
        
        # Training loss
        axes[0].plot(epochs, history_data['loss'], 'b-', label='Training Loss', linewidth=2)
        if history_data['val_loss']:
            axes[0].plot(epochs, history_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training MAE
        axes[1].plot(epochs, history_data['mae'], 'b-', label='Training MAE', linewidth=2)
        if history_data['val_mae']:
            axes[1].plot(epochs, history_data['val_mae'], 'r-', label='Validation MAE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_feature_importance(self, importance_data, save_path=None):
        """
        Create interactive feature importance plot using Plotly
        
        Args:
            importance_data (dict): Dictionary with model names and importance DataFrames
            save_path (str, optional): Path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        # Combine all importance data
        combined_data = []
        for model_name, importance_df in importance_data.items():
            for _, row in importance_df.head(10).iterrows():  # Top 10 features per model
                combined_data.append({
                    'model': model_name,
                    'feature': row['feature'],
                    'importance': row['importance']
                })
        
        df = pd.DataFrame(combined_data)
        
        # Create interactive bar plot
        fig = px.bar(df, x='importance', y='feature', color='model',
                    orientation='h', barmode='group',
                    title='Feature Importance Comparison Across Models',
                    labels={'importance': 'Importance Score', 'feature': 'Features'})
        
        fig.update_layout(
            height=600,
            font=dict(size=12),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig
    
    def create_interactive_model_performance(self, results_df, save_path=None):
        """
        Create interactive model performance dashboard
        
        Args:
            results_df (pd.DataFrame): Model evaluation results
            save_path (str, optional): Path to save the plot
            
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'R² Score Comparison', 
                          'MAE Comparison', 'Multi-Metric Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scatterpolar"}]]
        )
        
        models = results_df['model'].values
        colors = px.colors.qualitative.Set3[:len(models)]
        
        # RMSE bar chart
        fig.add_trace(
            go.Bar(x=models, y=results_df['rmse'], name='RMSE', 
                  marker_color=colors, showlegend=False),
            row=1, col=1
        )
        
        # R² bar chart
        fig.add_trace(
            go.Bar(x=models, y=results_df['r2'], name='R²', 
                  marker_color=colors, showlegend=False),
            row=1, col=2
        )
        
        # MAE bar chart
        fig.add_trace(
            go.Bar(x=models, y=results_df['mae'], name='MAE', 
                  marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        # Radar chart for multi-metric comparison
        metrics = ['RMSE', 'MAE', 'R²']
        for i, model in enumerate(models):
            # Normalize values for radar chart
            rmse_norm = 1 - (results_df.iloc[i]['rmse'] - results_df['rmse'].min()) / (results_df['rmse'].max() - results_df['rmse'].min())
            mae_norm = 1 - (results_df.iloc[i]['mae'] - results_df['mae'].min()) / (results_df['mae'].max() - results_df['mae'].min())
            r2_norm = results_df.iloc[i]['r2']
            
            fig.add_trace(
                go.Scatterpolar(
                    r=[rmse_norm, mae_norm, r2_norm],
                    theta=metrics,
                    fill='toself',
                    name=model,
                    line_color=colors[i]
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update polar subplot
        fig.update_polars(radialaxis_range=[0, 1], row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig
    
    def save_all_plots(self, data_dict, output_dir=None):
        """
        Generate and save all visualization plots
        
        Args:
            data_dict (dict): Dictionary containing all necessary data for plotting
            output_dir (str, optional): Directory to save plots
        """
        if output_dir is None:
            output_dir = self.plots_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        plots_generated = []
        
        # Data distribution
        if 'dataframe' in data_dict:
            plot_path = os.path.join(output_dir, 'data_distribution.png')
            self.plot_data_distribution(data_dict['dataframe'], save_path=plot_path)
            plots_generated.append(plot_path)
        
        # GA evolution
        if 'ga_evolution' in data_dict:
            plot_path = os.path.join(output_dir, 'ga_evolution.png')
            self.plot_genetic_algorithm_evolution(data_dict['ga_evolution'], save_path=plot_path)
            plots_generated.append(plot_path)
        
        # Feature selection
        if 'selected_features' in data_dict and 'all_features' in data_dict:
            plot_path = os.path.join(output_dir, 'feature_selection.png')
            self.plot_feature_selection_results(
                data_dict['selected_features'], 
                data_dict['all_features'],
                data_dict.get('feature_importance'),
                save_path=plot_path
            )
            plots_generated.append(plot_path)
        
        # Model comparison
        if 'evaluation_results' in data_dict:
            plot_path = os.path.join(output_dir, 'model_comparison.png')
            self.plot_model_comparison(data_dict['evaluation_results'], save_path=plot_path)
            plots_generated.append(plot_path)
        
        # Training history
        if 'training_history' in data_dict:
            plot_path = os.path.join(output_dir, 'training_history.png')
            self.plot_training_history(data_dict['training_history'], save_path=plot_path)
            plots_generated.append(plot_path)
        
        # Interactive plots
        if 'importance_data' in data_dict:
            plot_path = os.path.join(output_dir, 'interactive_feature_importance.html')
            self.create_interactive_feature_importance(data_dict['importance_data'], save_path=plot_path)
            plots_generated.append(plot_path)
        
        if 'evaluation_results' in data_dict:
            plot_path = os.path.join(output_dir, 'interactive_performance_dashboard.html')
            self.create_interactive_model_performance(data_dict['evaluation_results'], save_path=plot_path)
            plots_generated.append(plot_path)
        
        print(f"\\nGenerated {len(plots_generated)} visualization plots in '{output_dir}':")
        for plot in plots_generated:
            print(f"  - {os.path.basename(plot)}")
        
        return plots_generated
    
    def create_earthquake_map(self, df, save_path=None):
        """
        Create interactive earthquake map using the EarthquakeMapVisualizer
        
        Args:
            df (pd.DataFrame): Earthquake data with lat, lon, magnitude columns
            save_path (str, optional): Path to save the map
            
        Returns:
            str: Path to saved map file
        """
        try:
            from .earthquake_map import EarthquakeMapVisualizer
            
            # Initialize map visualizer
            map_viz = EarthquakeMapVisualizer()
            
            # Convert DataFrame to map-compatible format
            map_data = self._prepare_data_for_mapping(df)
            
            if len(map_data) > 0:
                # Create interactive map
                earthquake_map = map_viz.create_interactive_map(map_data)
                
                # Save map
                if save_path is None:
                    save_path = os.path.join(self.plots_dir, 'earthquake_map.html')
                
                earthquake_map.save(save_path)
                print(f"✅ Interactive earthquake map saved to {save_path}")
                
                return save_path
            else:
                print("⚠️ No valid data for mapping")
                return None
                
        except ImportError:
            print("❌ EarthquakeMapVisualizer not available. Install folium and streamlit.")
            return None
        except Exception as e:
            print(f"❌ Error creating earthquake map: {str(e)}")
            return None
    
    def create_live_earthquake_dashboard(self, save_dir=None):
        """
        Create live earthquake dashboard with real-time data
        
        Args:
            save_dir (str, optional): Directory to save dashboard files
            
        Returns:
            dict: Dictionary with paths to generated files
        """
        try:
            from .earthquake_map import EarthquakeMapVisualizer
            
            # Initialize map visualizer
            map_viz = EarthquakeMapVisualizer()
            
            print("🌍 Fetching live earthquake data...")
            # Fetch live earthquake data
            earthquake_data = map_viz.fetch_live_earthquake_data(
                days=7, 
                min_magnitude=4.0, 
                max_results=200
            )
            
            # Create comprehensive dashboard
            if save_dir is None:
                save_dir = self.plots_dir
            
            dashboard_path = os.path.join(save_dir, 'live_earthquake_dashboard.html')
            dashboard = map_viz.create_comprehensive_dashboard(
                earthquake_data, 
                save_path=dashboard_path
            )
            
            # Generate statistics
            stats = map_viz.get_earthquake_statistics(earthquake_data)
            
            print(f"\n📊 Live Earthquake Data Statistics:")
            print(f"Total earthquakes: {stats.get('total_earthquakes', 0)}")
            print(f"Max magnitude: {stats.get('magnitude_stats', {}).get('max', 0):.1f}")
            print(f"Time span: {stats.get('time_range', {}).get('span_days', 0)} days")
            
            return {
                'dashboard': dashboard,
                'stats': stats,
                'data': earthquake_data
            }
            
        except ImportError:
            print("❌ EarthquakeMapVisualizer not available. Install required dependencies.")
            return None
        except Exception as e:
            print(f"❌ Error creating live dashboard: {str(e)}")
            return None
    
    def _prepare_data_for_mapping(self, df):
        """
        Prepare DataFrame for earthquake mapping
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Map-compatible DataFrame
        """
        map_data = df.copy()
        
        # Check for required columns
        required_cols = ['latitude', 'longitude', 'magnitude']
        if not all(col in map_data.columns for col in required_cols):
            print(f"❌ Missing required columns for mapping: {required_cols}")
            print(f"Available columns: {list(map_data.columns)}")
            return pd.DataFrame()
        
        # Add time column if missing
        if 'time' not in map_data.columns:
            map_data['time'] = pd.date_range(
                start='2023-01-01', 
                periods=len(map_data), 
                freq='H'
            )
        
        # Add place column if missing
        if 'place' not in map_data.columns:
            map_data['place'] = 'Synthetic Location'
        
        # Add depth column if missing
        if 'depth_km' not in map_data.columns:
            # Use existing depth_km or default
            map_data['depth_km'] = map_data.get('depth_km', 10.0)
        
        # Filter out invalid coordinates
        map_data = map_data[
            (map_data['latitude'].between(-90, 90)) &
            (map_data['longitude'].between(-180, 180)) &
            (map_data['magnitude'] > 0)
        ]
        
        return map_data
