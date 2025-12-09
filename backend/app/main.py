import sys
import os
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.data_loader import EarthquakeDataLoader
from utils.weather_loader import WeatherLoader
from models.ensemble_model import EarthquakeEnsembleModel
import joblib

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models and Data
model = None
data_loader = EarthquakeDataLoader()
weather_loader = WeatherLoader()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../src/models/ensemble_model.pkl')
        # Check if saved model exists
        if os.path.exists(model_path):
            logger.info("Loading saved model...")
            # Note: The saved file is a dict with 'rf' and 'xgb' models, not the class instance itself usually
            # But the save_model in ensemble_model.py dumps a dict.
            # We need to reconstruct or wrap it. 
            # Actually, let's just load the dict and use it manually or reconstruct logic
            loaded_data = joblib.load(model_path)
            # We can create a wrapper or just use the loaded objects directly if we replicate prediction logic
            class ModelWrapper:
                def __init__(self, data):
                    self.rf = data['rf']
                    self.xgb = data['xgb']
                def predict(self, X):
                    rf_pred = self.rf.predict(X)
                    xgb_pred = self.xgb.predict(X)
                    return (rf_pred + xgb_pred) / 2.0
            
            model = ModelWrapper(loaded_data)
            logger.info("Model loaded successfully.")
        else:
            logger.warning("No model found at startup. Predictions will fail until model is trained.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    # Cleanup if needed

app = FastAPI(title="SeismicAI API", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    depth: float
    # Add other features if they are used by the model
    # For simplicity, we'll auto-fill others or accept them optionally
    timestamp: Optional[float] = None
    gap: Optional[float] = 0.0
    dmin: Optional[float] = 0.0
    rms: Optional[float] = 0.0

@app.get("/")
async def root():
    return {"message": "SeismicAI API is running"}

@app.get("/api/recent-earthquakes")
async def get_recent_earthquakes(limit: int = 100):
    """Get recent earthquake data"""
    try:
        df = data_loader.load_data()
        # Ensure date format is serializable
        df['time'] = df['time'].astype(str)
        
        # Sort by time desc
        df = df.sort_values('time', ascending=False)
        
        return df.head(limit).to_dict(orient='records')
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_earthquake(input_data: PredictionInput):
    """Predict magnitude based on parameters"""
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input vector
        # We need to know the feature names expected by the model.
        # The loaded model (RF) has feature_names_in_
        
        feature_names = model.rf.feature_names_in_
        
        # Construct input dict
        data = input_data.model_dump()
        
        # Add derived features if missing
        if input_data.timestamp is None:
            import time
            data['timestamp'] = time.time()
            
        # Create DataFrame
        df_input = pd.DataFrame([data])
        
        # Fill missing columns with 0
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0.0
                
        # Reorder to match model
        X = df_input[feature_names]
        
        prediction = model.predict(X)[0]
        
        return {
            "predicted_magnitude": float(prediction),
            "confidence": "High", # Placeholder
            "input_summary": data
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-performance")
async def get_model_performance():
    """Get saved model performance metrics"""
    results_path = os.path.join(os.path.dirname(__file__), '../results/performance_summary.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {"error": "Performance data not available"}

@app.get("/api/weather")
async def get_weather(lat: Optional[float] = None, lon: Optional[float] = None):
    """Get weather data for a location"""
    try:
        if lat is not None and lon is not None:
            return weather_loader.get_weather(lat, lon)
        return weather_loader.get_weather_for_default_location()
    except Exception as e:
        logger.error(f"Weather error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
