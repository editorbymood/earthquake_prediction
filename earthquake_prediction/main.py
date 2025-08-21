"""
Main execution script for Earthquake Magnitude Prediction with Feature Optimization

This script orchestrates the complete pipeline:
1. Data loading/generation
2. Genetic algorithm feature selection
3. Model training (ensemble and deep learning)
4. Evaluation and comparison
5. Visualization and reporting
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from config import (
    DATA_CONFIG, GA_CONFIG, ENSEMBLE_CONFIG, DL_CONFIG, 
    EVAL_CONFIG, FEATURE_NAMES, PATHS
)

# Import modules
from utils.data_loader import EarthquakeDataLoader
from utils.evaluator import ModelEvaluator
from utils.visualizer import EarthquakeVisualizer
from models.genetic_optimizer import run_genetic_feature_selection
from models.ensemble_model import create_ensemble_model
from models.deep_learning_model import create_deep_learning_model


def create_output_directories():
    """Create necessary output directories"""
    directories = ['results', 'plots', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("EARTHQUAKE MAGNITUDE PREDICTION WITH FEATURE OPTIMIZATION")
    print("="*80)
    print("Using Genetic Algorithm + Ensemble Models + Deep Learning")
    print()
    
    # Create output directories
    create_output_directories()
    
    # Initialize components
    data_loader = EarthquakeDataLoader()
    evaluator = ModelEvaluator()
    visualizer = EarthquakeVisualizer()
    
    # Step 1: Load and prepare data
    print("\\n" + "="*60)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*60)
    
    # Load data (will generate synthetic data if none exists)
    df = data_loader.load_data()
    
    # Display dataset information
    feature_info = data_loader.get_feature_info(df)
    print(f"\\nDataset Information:")
    print(f"  - Shape: {feature_info['data_shape']}")
    print(f"  - Features: {feature_info['n_features']}")
    print(f"  - Missing values: {feature_info['missing_values']}")
    print(f"  - Magnitude range: {feature_info['magnitude_stats']['min']:.2f} - {feature_info['magnitude_stats']['max']:.2f}")
    
    # Prepare data for modeling
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data(df)
    
    # Visualize data distribution
    visualizer.plot_data_distribution(df)
    
    # Step 2: Genetic Algorithm Feature Selection
    print("\\n" + "="*60)
    print("STEP 2: GENETIC ALGORITHM FEATURE SELECTION")
    print("="*60)
    
    start_time = time.time()
    
    # Run genetic algorithm for feature selection
    ga_results = run_genetic_feature_selection(X_train, y_train, GA_CONFIG)
    
    ga_time = time.time() - start_time
    print(f"\\nGenetic Algorithm completed in {ga_time:.2f} seconds")
    
    # Extract selected features
    selected_features = ga_results['selected_features']
    feature_importance_ga = ga_results['feature_importance']
    
    # Visualize GA evolution and feature selection
    visualizer.plot_genetic_algorithm_evolution(ga_results['evolution_history'])
    visualizer.plot_feature_selection_results(
        selected_features, 
        list(X_train.columns), 
        feature_importance_ga
    )
    
    # Filter data to selected features
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Step 3: Train Models
    print("\\n" + "="*60)
    print("STEP 3: MODEL TRAINING")
    print("="*60)
    
    models = {}
    model_results = {}
    
    # Train Ensemble Model (RandomForest + XGBoost + Stacking)
    print("\\n3.1 Training Ensemble Model...")
    start_time = time.time()
    
    ensemble_model = create_ensemble_model(ENSEMBLE_CONFIG)
    ensemble_results = ensemble_model.fit(X_train_selected, y_train, X_val_selected, y_val)
    models['Ensemble'] = ensemble_model
    model_results['Ensemble'] = ensemble_results
    
    ensemble_time = time.time() - start_time
    print(f"Ensemble training completed in {ensemble_time:.2f} seconds")
    
    # Train Deep Learning Model (MLP)
    print("\\n3.2 Training Deep Learning Model...")
    start_time = time.time()
    
    dl_model = create_deep_learning_model(DL_CONFIG)
    dl_results = dl_model.fit(X_train_selected, y_train, X_val_selected, y_val)
    models['Deep Learning'] = dl_model
    model_results['Deep Learning'] = dl_results
    
    dl_time = time.time() - start_time
    print(f"Deep Learning training completed in {dl_time:.2f} seconds")
    
    # Step 4: Model Evaluation and Comparison
    print("\\n" + "="*60)
    print("STEP 4: MODEL EVALUATION AND COMPARISON")
    print("="*60)
    
    # Make predictions on test set
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_test_selected)
    
    # Evaluate all models
    evaluation_results = evaluator.evaluate_models(predictions, y_test)
    
    # Print evaluation summary
    evaluator.print_evaluation_summary(evaluation_results)
    
    # Generate evaluation plots
    evaluator.generate_prediction_plots(predictions, y_test)
    evaluator.generate_residual_plots(predictions, y_test)
    
    # Step 5: Feature Importance Analysis
    print("\\n" + "="*60)
    print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    importance_data = {}
    
    # Ensemble feature importance
    ensemble_importance = ensemble_model.get_feature_importance()
    importance_data['Ensemble'] = ensemble_importance['combined_importance']
    
    # Deep Learning feature importance (approximation)
    dl_importance = dl_model.get_feature_importance_approximation(X_test_selected)
    importance_data['Deep Learning'] = dl_importance
    
    # GA feature importance
    importance_data['Genetic Algorithm'] = feature_importance_ga[feature_importance_ga['selected'] == True]
    
    # Compare feature importance across models
    importance_comparison = evaluator.feature_importance_comparison(importance_data)
    
    # Step 6: Visualization and Reporting
    print("\\n" + "="*60)
    print("STEP 6: VISUALIZATION AND REPORTING")
    print("="*60)
    
    # Create comprehensive visualizations
    visualizer.plot_model_comparison(evaluation_results)
    
    # Plot deep learning training history
    dl_history = dl_model.get_training_history()
    if dl_history:
        visualizer.plot_training_history(dl_history)
    
    # Create interactive plots
    visualizer.create_interactive_feature_importance(importance_data)
    visualizer.create_interactive_model_performance(evaluation_results)
    
    # Save all plots first to get the plots_generated list
    plot_data = {
        'dataframe': df,
        'ga_evolution': ga_results['evolution_history'],
        'selected_features': selected_features,
        'all_features': list(X_train.columns),
        'feature_importance': feature_importance_ga,
        'evaluation_results': evaluation_results,
        'training_history': dl_history,
        'importance_data': importance_data
    }
    
    plots_generated = visualizer.save_all_plots(plot_data, 'plots')
    
    # Create earthquake map visualizations
    print("\n🗺️ Creating Interactive Maps...")
    earthquake_map_path = visualizer.create_earthquake_map(df)
    if earthquake_map_path:
        plots_generated.append(earthquake_map_path)
    
    # Create live earthquake dashboard
    print("\n🌍 Creating Live Earthquake Dashboard...")
    live_dashboard = visualizer.create_live_earthquake_dashboard()
    if live_dashboard:
        print(f"✅ Live dashboard created with {live_dashboard['stats'].get('total_earthquakes', 0)} real earthquakes")
    
    # Step 7: Save Results and Models
    print("\\n" + "="*60)
    print("STEP 7: SAVING RESULTS AND MODELS")
    print("="*60)
    
    # Save models
    ensemble_model.save_model('models/ensemble_model.pkl')
    dl_model.save_model('models/deep_learning_model.h5')
    
    # Generate and save performance summary
    performance_summary = evaluator.generate_performance_summary(evaluation_results, 'results')
    
    # Save feature selection results
    import pandas as pd
    import json
    
    # Save selected features
    pd.DataFrame({
        'feature': selected_features,
        'selected': True
    }).to_csv('results/selected_features.csv', index=False)
    
    # Save GA results
    with open('results/ga_results.json', 'w') as f:
        json.dump({
            'selected_features': ga_results['selected_features'],
            'best_fitness': ga_results['best_fitness'],
            'n_selected_features': ga_results['n_selected_features']
        }, f, indent=2)
    
    # Final Summary
    print("\\n" + "="*80)
    print("PROJECT COMPLETION SUMMARY")
    print("="*80)
    
    print(f"\\n🎯 FEATURE SELECTION:")
    print(f"   - Original features: {len(X_train.columns)}")
    print(f"   - Selected features: {len(selected_features)} ({len(selected_features)/len(X_train.columns)*100:.1f}%)")
    print(f"   - GA best fitness (MSE): {ga_results['best_fitness']:.4f}")
    
    print(f"\\n🤖 MODEL PERFORMANCE:")
    best_model = evaluation_results.loc[evaluation_results['rmse'].idxmin()]
    print(f"   - Best model: {best_model['model']}")
    print(f"   - Best RMSE: {best_model['rmse']:.4f}")
    print(f"   - Best MAE: {best_model['mae']:.4f}")
    print(f"   - Best R²: {best_model['r2']:.4f}")
    
    print(f"\\n💾 OUTPUTS GENERATED:")
    print(f"   - Models saved: 2")
    print(f"   - Plots generated: {len(plots_generated)}")
    print(f"   - Results files: 4")
    
    print(f"\\n📊 FILES CREATED:")
    print("   Models:")
    print("     - models/ensemble_model.pkl")
    print("     - models/deep_learning_model.h5")
    print("   Results:")
    print("     - results/evaluation_results.csv")
    print("     - results/performance_summary.json")
    print("     - results/selected_features.csv")
    print("     - results/ga_results.json")
    print("   Plots:")
    for plot in plots_generated:
        print(f"     - {plot}")
    
    print(f"\\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    # Set up environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    
    try:
        main()
    except Exception as e:
        print(f"\\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\\nPlease check the error above and ensure all dependencies are installed.")
        print("Run: pip install -r requirements.txt")
