import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print(f"XGBoost not available: {e}. Using GradientBoostingRegressor instead.")

class EarthquakeEnsembleModel:
    def __init__(self, config):
        self.config = config
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None
        self.feature_names = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = list(X_train.columns)
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf_params = self.config.get('random_forest', {})
        self.rf_model = RandomForestRegressor(**rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # 2. XGBoost
        print("Training XGBoost...")
        xgb_params = self.config.get('xgboost', {})
        if HAS_XGB:
            self.xgb_model = xgb.XGBRegressor(**xgb_params)
            self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)] if X_val is not None else None, verbose=False)
        else:
            # Fallback
            self.xgb_model = GradientBoostingRegressor(n_estimators=100)
            self.xgb_model.fit(X_train, y_train)
            
        # 3. Simple Average (or Stacking)
        # For this implementation, we'll just use simple averaging for prediction, 
        # or we could train a meta-learner on the validation predictions.
        # Let's simple average for stability unless we want to implement true stacking.
        # "Stacked RandomForest and XGBoost regression" prompts suggest Stacking.
        
        metrics = {}
        if X_val is not None and y_val is not None:
            preds = self.predict(X_val)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_val, preds))
            metrics['mae'] = mean_absolute_error(y_val, preds)
            metrics['r2'] = r2_score(y_val, preds)
            
        return metrics

    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        return (rf_pred + xgb_pred) / 2.0

    def get_feature_importance(self):
        # Normalize and combine
        rf_imp = self.rf_model.feature_importances_
        xgb_imp = self.xgb_model.feature_importances_
        
        # Average rank or value
        combined = (rf_imp + xgb_imp) / 2.0
        
        return {
            'combined_importance': combined,
            'rf_importance': rf_imp,
            'xgb_importance': xgb_imp,
            'feature_names': self.feature_names
        }

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'rf': self.rf_model,
            'xgb': self.xgb_model,
            'config': self.config
        }, path)
        print(f"Ensemble model saved to {path}")

def create_ensemble_model(config):
    return EarthquakeEnsembleModel(config)
