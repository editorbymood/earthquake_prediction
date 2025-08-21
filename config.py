"""
Configuration file for earthquake magnitude prediction project
"""

# Data Generation Parameters
DATA_CONFIG = {
    'n_samples': 8000,  # Increased dataset size for better training
    'random_state': 42,
    'noise_level': 0.1,
    'test_size': 0.2,
    'val_size': 0.2,
    'location_focus': 'new_delhi',  # Focus location for enhanced data
    'use_enhanced_generator': True  # Use enhanced data generator by default
}

# Genetic Algorithm Parameters
GA_CONFIG = {
    'population_size': 100,
    'n_generations': 50,
    'crossover_prob': 0.7,
    'mutation_prob': 0.2,
    'tournament_size': 3,
    'min_features': 5,
    'max_features': 15,
    'cv_folds': 3,
    'random_state': 42
}

# Ensemble Model Parameters
ENSEMBLE_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'meta_learner': {
        'alpha': 1.0,
        'random_state': 42
    }
}

# Deep Learning Model Parameters
DL_CONFIG = {
    'hidden_layers': [256, 128, 64],
    'dropout_rate': 0.3,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'validation_split': 0.2,
    'random_state': 42
}

# Evaluation Parameters
EVAL_CONFIG = {
    'cv_folds': 5,
    'metrics': ['mae', 'mse', 'rmse', 'r2'],
    'plot_predictions': True,
    'save_results': True
}

# Feature Names for Synthetic Data
FEATURE_NAMES = [
    'latitude', 'longitude', 'depth_km', 'distance_to_fault_km',
    'p_wave_velocity', 's_wave_velocity', 'density', 'young_modulus',
    'poisson_ratio', 'stress_drop', 'focal_mechanism', 'seismic_moment',
    'rupture_length', 'rupture_width', 'slip_rate', 'crustal_thickness',
    'heat_flow', 'gravity_anomaly', 'magnetic_anomaly', 'topography_elevation'
]

# Output Paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'results_dir': 'results',
    'plots_dir': 'plots'
}
