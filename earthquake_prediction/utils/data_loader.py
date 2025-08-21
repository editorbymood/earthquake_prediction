"""
Data loading and generation utilities for earthquake magnitude prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from config import DATA_CONFIG, FEATURE_NAMES, PATHS

# Import the enhanced data generator
try:
    from utils.enhanced_data_generator import NewDelhiEarthquakeDataGenerator
    ENHANCED_GENERATOR_AVAILABLE = True
except ImportError:
    ENHANCED_GENERATOR_AVAILABLE = False


class EarthquakeDataLoader:
    """
    Data loader for earthquake magnitude prediction with synthetic data generation
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = FEATURE_NAMES
        
    def generate_synthetic_data(self, n_samples=5000, random_state=42, noise_level=0.1):
        """
        Generate synthetic earthquake data based on realistic geological parameters
        
        Args:
            n_samples (int): Number of samples to generate
            random_state (int): Random seed for reproducibility
            noise_level (float): Amount of noise to add to the target
            
        Returns:
            pd.DataFrame: Generated earthquake dataset
        """
        np.random.seed(random_state)
        
        # Generate features with realistic ranges
        data = {}
        
        # Geographic features
        data['latitude'] = np.random.uniform(-60, 60, n_samples)  # Global latitude range
        data['longitude'] = np.random.uniform(-180, 180, n_samples)  # Global longitude range
        data['depth_km'] = np.random.exponential(15, n_samples)  # Most earthquakes are shallow
        data['distance_to_fault_km'] = np.random.exponential(5, n_samples)  # Distance to nearest fault
        
        # Seismic wave properties
        data['p_wave_velocity'] = np.random.normal(6.5, 1.0, n_samples)  # km/s
        data['s_wave_velocity'] = np.random.normal(3.5, 0.8, n_samples)  # km/s
        
        # Rock properties
        data['density'] = np.random.normal(2.7, 0.3, n_samples)  # g/cm³
        data['young_modulus'] = np.random.normal(70, 15, n_samples)  # GPa
        data['poisson_ratio'] = np.random.normal(0.25, 0.05, n_samples)
        
        # Earthquake source parameters
        data['stress_drop'] = np.random.exponential(3, n_samples)  # MPa
        data['focal_mechanism'] = np.random.uniform(0, 360, n_samples)  # Strike angle
        data['seismic_moment'] = np.random.lognormal(15, 2, n_samples)  # N⋅m
        
        # Rupture characteristics
        data['rupture_length'] = np.random.exponential(10, n_samples)  # km
        data['rupture_width'] = np.random.exponential(5, n_samples)  # km
        data['slip_rate'] = np.random.exponential(2, n_samples)  # mm/year
        
        # Geological features
        data['crustal_thickness'] = np.random.normal(35, 10, n_samples)  # km
        data['heat_flow'] = np.random.normal(65, 20, n_samples)  # mW/m²
        data['gravity_anomaly'] = np.random.normal(0, 50, n_samples)  # mGal
        data['magnetic_anomaly'] = np.random.normal(0, 1000, n_samples)  # nT
        data['topography_elevation'] = np.random.normal(500, 1000, n_samples)  # m
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate realistic earthquake magnitude based on features
        magnitude = self._calculate_magnitude(df, noise_level)
        df['magnitude'] = magnitude
        
        return df
    
    def _calculate_magnitude(self, df, noise_level):
        """
        Calculate earthquake magnitude based on physical relationships
        
        Args:
            df (pd.DataFrame): DataFrame with earthquake features
            noise_level (float): Amount of noise to add
            
        Returns:
            np.ndarray: Calculated magnitudes
        """
        # Base magnitude from seismic moment (Hanks-Kanamori relation)
        moment_magnitude = (2/3) * (np.log10(df['seismic_moment']) - 16.1)
        
        # Adjustments based on other factors
        depth_factor = -0.01 * df['depth_km']  # Deeper = slightly smaller magnitude
        stress_factor = 0.1 * np.log(df['stress_drop'] + 1)  # Higher stress = larger magnitude
        fault_distance_factor = -0.05 * np.log(df['distance_to_fault_km'] + 1)  # Closer to fault = larger
        
        # Combine factors
        magnitude = (moment_magnitude + depth_factor + stress_factor + 
                    fault_distance_factor + np.random.normal(0, noise_level, len(df)))
        
        # Constrain to realistic range (1.0 to 9.5)
        magnitude = np.clip(magnitude, 1.0, 9.5)
        
        return magnitude
    
    def load_data(self, file_path=None, use_enhanced_generator=True):
        """
        Load earthquake data from file or generate synthetic data
        
        Args:
            file_path (str, optional): Path to earthquake data file
            use_enhanced_generator (bool): Use enhanced New Delhi generator if available
            
        Returns:
            pd.DataFrame: Earthquake dataset
        """
        # Check for enhanced data file first
        enhanced_data_path = os.path.join(PATHS['data_dir'], 'enhanced_earthquake_data.csv')
        
        if file_path and os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
        elif os.path.exists(enhanced_data_path):
            print(f"Loading enhanced earthquake data from {enhanced_data_path}")
            return pd.read_csv(enhanced_data_path)
        else:
            print("Generating earthquake data...")
            
            # Use enhanced generator if available and requested
            if use_enhanced_generator and ENHANCED_GENERATOR_AVAILABLE:
                print("🌟 Using Enhanced New Delhi Earthquake Data Generator")
                generator = NewDelhiEarthquakeDataGenerator(
                    random_state=DATA_CONFIG.get('random_state', 42)
                )
                
                # Generate enhanced dataset
                df = generator.generate_enhanced_dataset(
                    n_samples=DATA_CONFIG.get('n_samples', 5000)
                )
                df = generator.add_realistic_noise_and_uncertainties(df)
                
                # Save enhanced data
                os.makedirs(PATHS['data_dir'], exist_ok=True)
                output_path = enhanced_data_path
                df.to_csv(output_path, index=False)
                print(f"✅ Enhanced synthetic data saved to {output_path}")
                
                return df
            else:
                print("📊 Using Basic Earthquake Data Generator")
                # Use basic generator as fallback
                generation_params = {
                    'n_samples': DATA_CONFIG.get('n_samples', 5000),
                    'random_state': DATA_CONFIG.get('random_state', 42),
                    'noise_level': DATA_CONFIG.get('noise_level', 0.1)
                }
                df = self.generate_synthetic_data(**generation_params)
                
                # Save basic synthetic data
                os.makedirs(PATHS['data_dir'], exist_ok=True)
                output_path = os.path.join(PATHS['data_dir'], 'earthquake_data.csv')
                df.to_csv(output_path, index=False)
                print(f"Basic synthetic data saved to {output_path}")
                
                return df
    
    def prepare_data(self, df, target_column='magnitude'):
        """
        Prepare data for machine learning by splitting and scaling
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Name of target column
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Handle datetime columns by extracting useful features
        datetime_columns = ['timestamp'] if 'timestamp' in df.columns else []
        for col in datetime_columns:
            if col in df.columns:
                # Convert to datetime if it's a string
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col])
                
                # Extract useful datetime features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_dayofyear'] = df[col].dt.dayofyear
                df[f'{col}_weekday'] = df[col].dt.weekday
                
                # Drop original datetime column
                df = df.drop(columns=[col])
        
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_columns:
            categorical_columns.remove(target_column)
        
        for col in categorical_columns:
            # Use label encoding for categorical features
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Ensure all features are numeric
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        print(f"Features after preprocessing: {len(X.columns)}")
        print(f"Feature types: {X.dtypes.value_counts().to_dict()}")
        
        # Split data
        test_size = DATA_CONFIG.get('test_size', 0.2)
        val_size = DATA_CONFIG.get('val_size', 0.2)
        random_state = DATA_CONFIG.get('random_state', 42)
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Second split: train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Fit scaler on training data and transform all sets
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames with feature names
        feature_names = X.columns.tolist()
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
        
        print(f"Data split: Train={len(X_train_scaled)}, Val={len(X_val_scaled)}, Test={len(X_test_scaled)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def get_feature_info(self, df):
        """
        Get information about features in the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            dict: Feature information
        """
        info = {
            'n_features': len(df.columns) - 1,  # Excluding target
            'feature_names': [col for col in df.columns if col != 'magnitude'],
            'data_shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'magnitude_stats': df['magnitude'].describe() if 'magnitude' in df.columns else None
        }
        
        return info
