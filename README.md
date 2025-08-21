# Earthquake Magnitude Prediction with Feature Optimization

A complete machine learning project that uses Genetic Algorithms to optimize feature selection for earthquake magnitude prediction, comparing ensemble methods (RandomForest + XGBoost) with deep learning (MLP) models.

## Features

- **Genetic Algorithm Feature Selection**: Automatically selects optimal features for prediction
- **Ensemble Model**: Stacked RandomForest and XGBoost regression
- **Deep Learning Model**: Multi-layer perceptron with TensorFlow/Keras
- **Comprehensive Evaluation**: Cross-validation, feature importance analysis, and performance metrics
- **Data Visualization**: Interactive plots and analysis charts

## Project Structure

```
earthquake_prediction/
├── data/
│   └── earthquake_data.csv
├── models/
│   ├── __init__.py
│   ├── genetic_optimizer.py
│   ├── ensemble_model.py
│   └── deep_learning_model.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluator.py
│   └── visualizer.py
├── main.py
├── config.py
└── requirements.txt
```

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Generate synthetic earthquake data
2. Apply genetic algorithm for feature optimization
3. Train ensemble and deep learning models
4. Compare performance and generate visualizations

## Models

### Ensemble Model
- **RandomForest**: Handles non-linear patterns and feature interactions
- **XGBoost**: Gradient boosting for enhanced accuracy
- **Stacking**: Meta-learner combines predictions

### Deep Learning Model
- **MLP Architecture**: Multiple hidden layers with dropout
- **Adaptive Learning**: Early stopping and learning rate scheduling
- **Regularization**: Dropout and batch normalization

## Results

The project outputs:
- Feature importance rankings
- Model performance comparisons
- Cross-validation results
- Prediction accuracy metrics
- Visualization plots

## Configuration

Modify `config.py` to adjust:
- Genetic algorithm parameters
- Model hyperparameters
- Data generation settings
- Evaluation metrics
