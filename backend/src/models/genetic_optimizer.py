import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from config import GA_CONFIG

# Check if deap is available
try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    print("Warning: DEAP not found. Using fallback feature selection.")

def eval_features(individual, X, y, estimator, cv_folds):
    """Fitness function for GA"""
    # Select features based on individual (binary mask)
    cols = [i for i, val in enumerate(individual) if val == 1]
    
    # Penalize if too few features
    if len(cols) < GA_CONFIG.get('min_features', 2):
        return float('inf'),
    
    X_selected = X.iloc[:, cols]
    
    # Calculate negative MSE (since we want to minimize error, but sklearn returns neg_mse)
    # Actually cross_val_score with 'neg_mean_squared_error' returns negative values.
    # Higher is better (closer to 0).
    # But DEAP usually minimizes.
    # Let's use RMSE.
    scores = cross_val_score(estimator, X_selected, y, cv=cv_folds, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    return mse,

def run_genetic_feature_selection(X_train, y_train, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Genetic Algorithm to select optimal features.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training target
        config: Configuration dictionary
        
    Returns:
        Dictionary with results
    """
    feature_names = list(X_train.columns)
    n_features = len(feature_names)
    
    print(f"Starting Genetic Algorithm on {n_features} features...")
    
    if not HAS_DEAP:
        # Fallback: Select all numerical features or simple correlation based
        # For now, just return all
        return {
            'selected_features': feature_names,
            'feature_importance': pd.DataFrame({'feature': feature_names, 'selected': True}),
            'evolution_history': [],
            'best_fitness': 0.0,
            'n_selected_features': n_features
        }

    # Setup DEAP
    # Create types: FitnessMin, Individual (list)
    # We must wrap creation in try/except to avoid 'Class already exists' error in Jupyter/Re-runs, 
    # but in script it's fine.
    try:
        del creator.FitnessMin
        del creator.Individual
    except AttributeError:
        pass
        
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Attribute generator: random 0 or 1
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Estimator for evaluation (Found Random Forest to be robust)
    estimator = RandomForestRegressor(
        n_estimators=10, # Low estimators for speed
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Operator registration
    toolbox.register("evaluate", eval_features, X=X_train, y=y_train, estimator=estimator, cv_folds=2)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=config.get('tournament_size', 3))
    
    # GA Parameters
    pop = toolbox.population(n=config.get('population_size', 20))
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run Algorithm
    print(f"Running GA for {config.get('n_generations', 10)} generations...")
    pop, log = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=config.get('crossover_prob', 0.5), 
        mutpb=config.get('mutation_prob', 0.2), 
        ngen=config.get('n_generations', 10), 
        stats=stats, 
        halloffame=hof, 
        verbose=True
    )
    
    # Process results
    best_ind = hof[0]
    selected_indices = [i for i, val in enumerate(best_ind) if val == 1]
    selected_features = [feature_names[i] for i in selected_indices]
    
    # Ensure at least 'min_features' are selected, if not, fill with others
    min_feats = config.get('min_features', 1)
    if len(selected_features) < min_feats:
        # Add random features to meet minimum
        remaining = [f for f in feature_names if f not in selected_features]
        needed = min_feats - len(selected_features)
        selected_features.extend(remaining[:needed])
        
    print(f"GA selected {len(selected_features)} features.")
    
    # Create importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'selected': [f in selected_features for f in feature_names]
    })
    
    return {
        'selected_features': selected_features,
        'feature_importance': feature_importance,
        'evolution_history': log,
        'best_fitness': best_ind.fitness.values[0],
        'n_selected_features': len(selected_features)
    }
