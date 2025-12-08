import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from io import StringIO
from sklearn.model_selection import train_test_split
from config import DATA_CONFIG

class EarthquakeDataLoader:
    def __init__(self):
        self.data_dir = 'data'
        self.data_file = os.path.join(self.data_dir, 'earthquake_data.csv')
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self):
        """Load data from CSV or download if not exists"""
        if not os.path.exists(self.data_file):
            print("Data file not found. Downloading from USGS...")
            try:
                self.download_usgs_data()
            except Exception as e:
                print(f"Failed to download data: {e}")
                print("Generating synthetic data as fallback...")
                return self.generate_synthetic_data()
        
        try:
            df = pd.read_csv(self.data_file)
            print(f"Loaded {len(df)} records from {self.data_file}")
            
            # Simple validation
            required_cols = ['time', 'latitude', 'longitude', 'mag', 'depth']
            if not all(col in df.columns for col in required_cols):
                print("Missing required columns. Re-downloading...")
                self.download_usgs_data()
                df = pd.read_csv(self.data_file)
                
            return df
        except Exception as e:
            print(f"Error reading data: {e}")
            return self.generate_synthetic_data()

    def download_usgs_data(self):
        """Download earthquake data from USGS API for the last 5 years"""
        # Endpoint
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
        # Parameters
        end_time = datetime.now()
        start_time = end_time - timedelta(days=365*5) # Last 5 years
        
        params = {
            'format': 'csv',
            'starttime': start_time.strftime('%Y-%m-%d'),
            'endtime': end_time.strftime('%Y-%m-%d'),
            'minmagnitude': 4.5, # Increased to keep within limit
            'limit': 20000,
            'orderby': 'time'
        }
        
        print(f"Requesting data from USGS ({start_time.date()} to {end_time.date()})...")
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Check if we got valid CSV content
            if "time,latitude,longitude" in response.text[:200]:
                with open(self.data_file, 'w') as f:
                    f.write(response.text)
                print("Download successful.")
            else:
                raise Exception("Invalid API response format")
        else:
            raise Exception(f"API Request failed: {response.status_code}")

    def generate_synthetic_data(self):
        """Fallback: Generate synthetic data"""
        print("Generating synthetic earthquake data...")
        n_samples = DATA_CONFIG.get('n_samples', 1000)
        
        data = {
            'time': pd.date_range(end=datetime.now(), periods=n_samples, freq='4h'),
            'latitude': np.random.uniform(-90, 90, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'depth': np.random.uniform(0, 700, n_samples),
            'mag': np.random.normal(3.5, 1.0, n_samples).clip(0, 10),
            # Add some 'geophysical' columns to match expected features somewhat
            'gap': np.random.uniform(0, 360, n_samples),
            'dmin': np.random.uniform(0, 10, n_samples),
            'rms': np.random.uniform(0, 2, n_samples),
        }
        return pd.DataFrame(data)

    def prepare_data(self, df):
        """Preprocess data and split into train/val/test"""
        # Convert time
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            # Feature Engineering
            df['timestamp'] = df['time'].astype('int64') // 10**9
            df['month'] = df['time'].dt.month
            df['day_of_week'] = df['time'].dt.dayofweek
            df['hour'] = df['time'].dt.hour
        
        # Handle missing values in important columns
        df['depth'] = df['depth'].fillna(df['depth'].mean())
        if 'mag' in df.columns:
            df = df.dropna(subset=['mag'])
        
        self.target_col = 'mag'
        
        # Select numeric columns only for X
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_col)
            
        X = df[numeric_cols]
        y = df[self.target_col]
        
        # Replace NaNs in X
        X = X.fillna(0)
        
        # Split
        test_size = DATA_CONFIG.get('test_size', 0.2)
        val_size = DATA_CONFIG.get('val_size', 0.2)
        
        # First split: Train + Val vs Test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: Train vs Val
        # Adjust val_size relative to the remaining data
        relative_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=relative_val_size, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_info(self, df):
        """Return dataset statistics"""
        return {
            'data_shape': df.shape,
            'n_features': df.shape[1],
            'missing_values': df.isnull().sum().sum(),
            'magnitude_stats': {
                'min': df['mag'].min(),
                'max': df['mag'].max(),
                'mean': df['mag'].mean()
            }
        }
