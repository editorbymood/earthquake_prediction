"""
Genetic Algorithm for feature selection in earthquake magnitude prediction
"""

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import random
from config import GA_CONFIG


class GeneticFeatureSelector:
    """
    Genetic Algorithm for selecting optimal features for earthquake magnitude prediction
    """
    
    def __init__(self, X_train, y_train, config=None):
        """
        Initialize the genetic algorithm feature selector
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            config (dict): GA configuration parameters
        """
        self.X_train = X_train
        self.y_train = y_train
        self.config = config or GA_CONFIG
        self.feature_names = list(X_train.columns)
        self.n_features = len(self.feature_names)
        
        # Setup DEAP framework
        self._setup_deap()
        
        # Best individual found
        self.best_individual = None
        self.best_fitness = float('inf')
        self.evolution_history = []
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MSE
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        
        # Attribute generator: binary representation for feature selection
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_bool, self.n_features)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, 
                             tournsize=self.config['tournament_size'])
    
    def _evaluate_individual(self, individual):
        """
        Evaluate an individual by training a model with selected features
        
        Args:
            individual (list): Binary representation of feature selection
            
        Returns:
            tuple: Fitness score (MSE)
        """
        # Get selected features
        selected_features = [i for i, bit in enumerate(individual) if bit == 1]
        
        # Ensure minimum number of features
        if len(selected_features) < self.config['min_features']:
            return (float('inf'),)
        
        # Ensure maximum number of features
        if len(selected_features) > self.config['max_features']:
            return (float('inf'),)
        
        # Select features from training data
        feature_cols = [self.feature_names[i] for i in selected_features]
        X_selected = self.X_train[feature_cols]
        
        # Train a simple RandomForest for evaluation
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=self.config['random_state'],
            n_jobs=1  # Use single job for stability in GA
        )
        
        # Cross-validation score
        cv_scores = cross_val_score(
            rf, X_selected, self.y_train, 
            cv=self.config['cv_folds'], 
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        # Return negative MSE (since we want to minimize)
        mse = -cv_scores.mean()
        return (mse,)
    
    def evolve(self):
        """
        Run the genetic algorithm to find optimal feature subset
        
        Returns:
            dict: Results containing best features and evolution history
        """
        print("Starting genetic algorithm feature selection...")
        
        # Initialize population
        population = self.toolbox.population(n=self.config['population_size'])
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution parameters
        CXPB = self.config['crossover_prob']
        MUTPB = self.config['mutation_prob']
        NGEN = self.config['n_generations']
        
        print(f"Generation 0: Best fitness = {min(fitnesses)[0]:.4f}")
        
        # Evolution loop
        for gen in range(1, NGEN + 1):
            # Select the next generation individuals
            offspring = self.toolbox.select(population, len(population))
            
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Track best individual
            current_best = tools.selBest(population, 1)[0]
            current_best_fitness = current_best.fitness.values[0]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = current_best.copy()
            
            # Store evolution history
            generation_stats = {
                'generation': gen,
                'best_fitness': current_best_fitness,
                'avg_fitness': np.mean([ind.fitness.values[0] for ind in population]),
                'worst_fitness': max(ind.fitness.values[0] for ind in population)
            }
            self.evolution_history.append(generation_stats)
            
            # Print progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {current_best_fitness:.4f}")
        
        # Get final results
        selected_indices = [i for i, bit in enumerate(self.best_individual) if bit == 1]
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        results = {
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'best_fitness': self.best_fitness,
            'n_selected_features': len(selected_features),
            'evolution_history': self.evolution_history,
            'feature_selection_binary': self.best_individual
        }
        
        print(f"\nGenetic Algorithm completed!")
        print(f"Best fitness (MSE): {self.best_fitness:.4f}")
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        return results
    
    def get_feature_importance_ga(self):
        """
        Get feature importance based on GA selection frequency
        
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if not self.evolution_history:
            print("No evolution history available. Run evolve() first.")
            return None
        
        # This is a simplified version - in a full implementation,
        # you would track feature selection frequency across generations
        importance_scores = []
        for i, feature in enumerate(self.feature_names):
            if self.best_individual and self.best_individual[i] == 1:
                importance_scores.append(1.0)
            else:
                importance_scores.append(0.0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores,
            'selected': [bool(score) for score in importance_scores]
        }).sort_values('importance', ascending=False)
        
        return importance_df


def run_genetic_feature_selection(X_train, y_train, config=None):
    """
    Convenience function to run genetic algorithm feature selection
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        config (dict): GA configuration
        
    Returns:
        dict: Feature selection results
    """
    selector = GeneticFeatureSelector(X_train, y_train, config)
    results = selector.evolve()
    
    # Add feature importance
    importance_df = selector.get_feature_importance_ga()
    results['feature_importance'] = importance_df
    
    return results
