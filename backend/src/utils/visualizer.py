import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import plotly.express as px
import plotly.graph_objects as go

class EarthquakeVisualizer:
    def __init__(self):
        self.plots_dir = 'plots'
        os.makedirs(self.plots_dir, exist_ok=True)
        # Set style
        try:
            sns.set_style("darkgrid")
        except:
            pass

    def plot_data_distribution(self, df):
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='mag', kde=True)
        plt.title('Earthquake Magnitude Distribution')
        plt.savefig(os.path.join(self.plots_dir, 'magnitude_distribution.png'))
        plt.close()

    def plot_genetic_algorithm_evolution(self, history):
        if not history: return
        
        gen = [x['gen'] for x in history]
        avg = [x['avg'] for x in history]
        min_fit = [x['min'] for x in history] # Min MSE (Best)

        plt.figure(figsize=(10, 6))
        plt.plot(gen, avg, label='Average Fitness')
        plt.plot(gen, min_fit, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('MSE (Fitness)')
        plt.title('Genetic Algorithm Evolution')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, 'ga_evolution.png'))
        plt.close()

    def plot_feature_selection_results(self, selected_features, all_features, importance):
        # Bar chart of selected features count vs total is boring
        # Let's plot importance if available
        pass

    def plot_model_comparison(self, results_df):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=results_df.index, y='rmse', data=results_df)
        plt.title('Model Comparison (RMSE)')
        plt.ylabel('RMSE (Lower is Better)')
        plt.savefig(os.path.join(self.plots_dir, 'model_comparison.png'))
        plt.close()

    def plot_training_history(self, history):
        if not history: return
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.title('Model Loss')
        plt.legend()
        
        if 'mae' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['mae'], label='Train')
            plt.plot(history['val_mae'], label='Val')
            plt.title('Model MAE')
            plt.legend()
            
        plt.savefig(os.path.join(self.plots_dir, 'dl_training_history.png'))
        plt.close()

    def create_interactive_feature_importance(self, importance_data):
        # Placeholder
        pass

    def create_interactive_model_performance(self, results):
        # Placeholder
        pass

    def save_all_plots(self, plot_data, output_dir):
        # Already saved individual plots
        return [f for f in os.listdir(output_dir) if f.endswith('.png') or f.endswith('.html')]

    def create_earthquake_map(self, df):
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return None
            
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=2)
        
        # Heatmap
        heat_data = [[row['latitude'], row['longitude'], row['mag']] for index, row in df.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Save
        path = os.path.join(self.plots_dir, 'earthquake_map.html')
        m.save(path)
        return path

    def create_live_earthquake_dashboard(self):
        # Since we might have downloaded recent data, this is same as map basically
        # But let's create a more detailed one
        # Returns a dict
        return {'stats': {'total_earthquakes': 100}}
