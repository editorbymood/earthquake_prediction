"""
Enhanced Earthquake Data Generator for New Delhi Region
Generates realistic seismic data based on actual geological characteristics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, Any

class NewDelhiEarthquakeDataGenerator:
    """Generate realistic earthquake data for New Delhi region"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # New Delhi region geographical bounds
        self.delhi_bounds = {
            'lat_min': 28.40,    # South Delhi boundary
            'lat_max': 28.88,    # North Delhi boundary  
            'lon_min': 76.84,    # West Delhi boundary
            'lon_max': 77.34,    # East Delhi boundary
        }
        
        # Extended NCR region for broader seismic context
        self.ncr_bounds = {
            'lat_min': 27.50,    # Includes Mathura, Agra region
            'lat_max': 29.50,    # Includes Panipat, Karnal region
            'lon_min': 76.00,    # Includes Alwar region
            'lon_max': 78.00,    # Includes Bulandshahr region
        }
        
        # Real fault systems near Delhi
        self.fault_systems = {
            'Delhi_Ridge_Fault': {'lat': 28.65, 'lon': 77.20, 'depth': 15, 'activity': 0.7},
            'Mathura_Fault': {'lat': 27.50, 'lon': 77.67, 'depth': 20, 'activity': 0.5},
            'Moradabad_Fault': {'lat': 28.84, 'lon': 78.78, 'depth': 25, 'activity': 0.6},
            'Sohna_Fault': {'lat': 28.25, 'lon': 77.07, 'depth': 18, 'activity': 0.4},
            'Gurgaon_Fault': {'lat': 28.46, 'lon': 77.03, 'depth': 12, 'activity': 0.3}
        }
        
        # Geological layers in NCR region
        self.geological_layers = {
            'Quaternary_Alluvium': {'depth_range': (0, 50), 'density': 1.8, 'velocity_p': 2.5, 'velocity_s': 1.2},
            'Upper_Siwalik': {'depth_range': (50, 200), 'density': 2.3, 'velocity_p': 4.2, 'velocity_s': 2.4},
            'Middle_Siwalik': {'depth_range': (200, 500), 'density': 2.5, 'velocity_p': 5.8, 'velocity_s': 3.3},
            'Delhi_Supergroup': {'depth_range': (500, 2000), 'density': 2.7, 'velocity_p': 6.2, 'velocity_s': 3.6},
            'Basement_Gneiss': {'depth_range': (2000, 5000), 'density': 2.9, 'velocity_p': 6.8, 'velocity_s': 3.9}
        }
    
    def generate_realistic_coordinates(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic latitude and longitude coordinates around New Delhi"""
        
        # 60% Delhi region, 40% broader NCR region
        delhi_samples = int(0.6 * n_samples)
        ncr_samples = n_samples - delhi_samples
        
        # Delhi region coordinates
        delhi_lats = np.random.uniform(
            self.delhi_bounds['lat_min'], 
            self.delhi_bounds['lat_max'], 
            delhi_samples
        )
        delhi_lons = np.random.uniform(
            self.delhi_bounds['lon_min'], 
            self.delhi_bounds['lon_max'], 
            delhi_samples
        )
        
        # NCR region coordinates
        ncr_lats = np.random.uniform(
            self.ncr_bounds['lat_min'], 
            self.ncr_bounds['lat_max'], 
            ncr_samples
        )
        ncr_lons = np.random.uniform(
            self.ncr_bounds['lon_min'], 
            self.ncr_bounds['lon_max'], 
            ncr_samples
        )
        
        # Combine coordinates
        latitudes = np.concatenate([delhi_lats, ncr_lats])
        longitudes = np.concatenate([delhi_lons, ncr_lons])
        
        return latitudes, longitudes
    
    def calculate_distance_to_nearest_fault(self, lat: float, lon: float) -> float:
        """Calculate distance to nearest known fault system"""
        min_distance = float('inf')
        
        for fault_name, fault_data in self.fault_systems.items():
            # Haversine distance approximation for short distances
            lat_diff = lat - fault_data['lat']
            lon_diff = lon - fault_data['lon']
            distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Convert to km
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def generate_depth_characteristics(self, lat: float, lon: float, fault_distance: float) -> Dict[str, float]:
        """Generate depth and associated geological characteristics"""
        
        # Depth influenced by proximity to faults and geological setting
        base_depth = 5.0  # Shallow crustal earthquakes common in region
        fault_influence = max(0, 20 - fault_distance)  # Closer to faults = potentially deeper
        
        # Most earthquakes in NCR are shallow (0-30 km)
        if np.random.random() < 0.7:  # 70% shallow earthquakes
            depth = np.random.exponential(8) + 1  # 1-30 km range
        elif np.random.random() < 0.9:  # 20% intermediate
            depth = np.random.uniform(30, 70)
        else:  # 10% deeper
            depth = np.random.uniform(70, 150)
        
        depth = min(depth, 200)  # Maximum depth constraint for region
        
        # Determine geological layer
        layer_properties = None
        for layer_name, properties in self.geological_layers.items():
            if properties['depth_range'][0] <= depth <= properties['depth_range'][1]:
                layer_properties = properties
                break
        
        if layer_properties is None:
            layer_properties = self.geological_layers['Basement_Gneiss']
        
        return {
            'depth': depth,
            'layer_properties': layer_properties
        }
    
    def generate_realistic_datetime(self, n_samples: int) -> pd.Series:
        """Generate realistic earthquake occurrence times"""
        
        # Generate timestamps over last 50 years with varying frequency
        end_date = datetime.now()
        start_date = end_date - timedelta(days=50*365)
        
        # Create realistic temporal distribution
        timestamps = []
        
        for _ in range(n_samples):
            # Random date within range
            random_days = np.random.randint(0, (end_date - start_date).days)
            random_seconds = np.random.randint(0, 24*60*60)
            
            earthquake_time = start_date + timedelta(days=random_days, seconds=random_seconds)
            timestamps.append(earthquake_time)
        
        return pd.Series(timestamps).sort_values().reset_index(drop=True)
    
    def generate_seismic_parameters(self, depth: float, fault_distance: float, 
                                  layer_properties: Dict) -> Dict[str, float]:
        """Generate realistic seismic parameters based on geology"""
        
        # P and S wave velocities based on geological layer
        p_velocity = layer_properties['velocity_p'] + np.random.normal(0, 0.3)
        s_velocity = layer_properties['velocity_s'] + np.random.normal(0, 0.2)
        
        # Ensure realistic Vp/Vs ratio (typically 1.6-1.8 for crustal rocks)
        if p_velocity / s_velocity < 1.5:
            s_velocity = p_velocity / 1.7
        elif p_velocity / s_velocity > 2.0:
            s_velocity = p_velocity / 1.8
        
        # Density from geological layer with variation
        density = layer_properties['density'] + np.random.normal(0, 0.1)
        
        # Young's modulus and Poisson ratio (realistic ranges for rocks)
        young_modulus = np.random.uniform(20, 100) * 1e9  # GPa
        poisson_ratio = np.random.uniform(0.15, 0.35)
        
        # Stress drop (typical values 0.1-10 MPa)
        stress_drop = 10 ** np.random.uniform(-1, 1)  # 0.1 to 10 MPa
        
        return {
            'p_wave_velocity': max(1.5, p_velocity),
            's_wave_velocity': max(0.8, s_velocity),
            'density': max(1.5, density),
            'young_modulus': young_modulus,
            'poisson_ratio': poisson_ratio,
            'stress_drop': stress_drop
        }
    
    def generate_rupture_parameters(self, magnitude: float, stress_drop: float) -> Dict[str, float]:
        """Generate rupture parameters based on magnitude and stress drop"""
        
        # Seismic moment from magnitude (Hanks-Kanamori relation)
        seismic_moment = 10 ** (1.5 * magnitude + 9.1)  # N⋅m
        
        # Rupture area scaling (Wells & Coppersmith, 1994)
        log_area = magnitude - 3.49  # km²
        rupture_area = 10 ** log_area
        
        # Rupture length and width with aspect ratio considerations
        aspect_ratio = np.random.uniform(1.5, 3.0)  # Length/Width ratio
        rupture_length = np.sqrt(rupture_area * aspect_ratio)
        rupture_width = rupture_area / rupture_length
        
        # Average displacement from seismic moment and area
        rigidity = 30e9  # Pa (typical for crustal rocks)
        average_displacement = seismic_moment / (rigidity * rupture_area * 1e6)
        
        return {
            'seismic_moment': seismic_moment,
            'rupture_length': rupture_length,
            'rupture_width': rupture_width,
            'average_displacement': average_displacement
        }
    
    def generate_focal_mechanism(self, fault_distance: float) -> Dict[str, float]:
        """Generate realistic focal mechanism based on regional tectonics"""
        
        # NCR region has predominantly reverse and strike-slip faulting
        mechanism_types = {
            'reverse': 0.4,      # Compressive regime
            'strike_slip': 0.35, # Transform faulting
            'normal': 0.15,      # Minor extension
            'oblique': 0.1       # Combined mechanisms
        }
        
        mechanism = np.random.choice(
            list(mechanism_types.keys()),
            p=list(mechanism_types.values())
        )
        
        if mechanism == 'reverse':
            strike = np.random.uniform(0, 360)
            dip = np.random.uniform(30, 60)
            rake = np.random.uniform(60, 120)
        elif mechanism == 'strike_slip':
            strike = np.random.uniform(0, 360)
            dip = np.random.uniform(70, 90)
            rake = np.random.choice([np.random.uniform(-30, 30), 
                                   np.random.uniform(150, 210)])
        elif mechanism == 'normal':
            strike = np.random.uniform(0, 360)
            dip = np.random.uniform(45, 75)
            rake = np.random.uniform(-120, -60)
        else:  # oblique
            strike = np.random.uniform(0, 360)
            dip = np.random.uniform(35, 75)
            rake = np.random.uniform(-150, 150)
        
        return {
            'focal_mechanism_strike': strike,
            'focal_mechanism_dip': dip,
            'focal_mechanism_rake': rake,
            'mechanism_type': mechanism
        }
    
    def generate_regional_parameters(self, lat: float, lon: float) -> Dict[str, float]:
        """Generate regional geological and geophysical parameters"""
        
        # Crustal thickness varies across NCR (28-45 km)
        if self.delhi_bounds['lat_min'] <= lat <= self.delhi_bounds['lat_max']:
            crustal_thickness = np.random.normal(35, 3)  # Delhi region
        else:
            crustal_thickness = np.random.normal(40, 5)  # Broader NCR
        
        # Heat flow (typical values for stable continental crust)
        heat_flow = np.random.normal(45, 10)  # mW/m²
        
        # Gravity anomaly (Bouguer anomaly)
        gravity_anomaly = np.random.normal(-20, 30)  # mGal
        
        # Magnetic anomaly 
        magnetic_anomaly = np.random.normal(0, 500)  # nT
        
        # Topography elevation (Delhi is relatively flat)
        if self.delhi_bounds['lat_min'] <= lat <= self.delhi_bounds['lat_max']:
            elevation = np.random.normal(220, 50)  # Delhi elevation ~200-300m
        else:
            elevation = np.random.normal(250, 100)  # NCR regional variation
        
        return {
            'crustal_thickness': max(25, min(50, crustal_thickness)),
            'heat_flow': max(20, min(80, heat_flow)),
            'gravity_anomaly': gravity_anomaly,
            'magnetic_anomaly': magnetic_anomaly,
            'topography_elevation': max(100, min(500, elevation))
        }
    
    def generate_magnitude(self, depth: float, fault_distance: float) -> float:
        """Generate realistic magnitude following Gutenberg-Richter distribution"""
        
        # Regional parameters for NCR (low to moderate seismicity)
        b_value = 0.9  # Slightly lower b-value for intraplate region
        
        # Magnitude distribution heavily weighted toward smaller events
        if np.random.random() < 0.6:  # 60% micro earthquakes
            magnitude = np.random.uniform(1.0, 3.0)
        elif np.random.random() < 0.85:  # 25% small earthquakes
            magnitude = np.random.uniform(3.0, 4.5)
        elif np.random.random() < 0.95:  # 10% moderate earthquakes
            magnitude = np.random.uniform(4.5, 6.0)
        else:  # 5% larger earthquakes (rare for region)
            magnitude = np.random.uniform(6.0, 7.0)
        
        # Slight adjustment based on depth and fault proximity
        depth_factor = 0.1 * (depth / 50)  # Deeper events can be slightly larger
        fault_factor = -0.05 * (fault_distance / 50)  # Closer to faults slightly larger
        
        magnitude += depth_factor + fault_factor
        
        return max(1.0, min(7.5, magnitude))
    
    def generate_enhanced_dataset(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate complete enhanced earthquake dataset"""
        
        print(f"Generating {n_samples} realistic earthquake records for New Delhi region...")
        
        # Generate coordinates
        latitudes, longitudes = self.generate_realistic_coordinates(n_samples)
        
        # Generate timestamps
        timestamps = self.generate_realistic_datetime(n_samples)
        
        # Initialize data storage
        data = []
        
        for i in range(n_samples):
            if i % 500 == 0:
                print(f"Generated {i}/{n_samples} records...")
            
            lat, lon = latitudes[i], longitudes[i]
            timestamp = timestamps.iloc[i]
            
            # Calculate fault distance
            fault_distance = self.calculate_distance_to_nearest_fault(lat, lon)
            
            # Generate depth and geological characteristics
            depth_info = self.generate_depth_characteristics(lat, lon, fault_distance)
            depth = depth_info['depth']
            layer_props = depth_info['layer_properties']
            
            # Generate magnitude first (needed for rupture parameters)
            magnitude = self.generate_magnitude(depth, fault_distance)
            
            # Generate seismic parameters
            seismic_params = self.generate_seismic_parameters(depth, fault_distance, layer_props)
            
            # Generate rupture parameters
            rupture_params = self.generate_rupture_parameters(magnitude, seismic_params['stress_drop'])
            
            # Generate focal mechanism
            focal_params = self.generate_focal_mechanism(fault_distance)
            
            # Generate regional parameters
            regional_params = self.generate_regional_parameters(lat, lon)
            
            # Calculate additional derived parameters
            slip_rate = rupture_params['average_displacement'] / np.random.uniform(0.1, 10)  # mm/year
            
            # Compile record
            record = {
                # Basic location and time
                'timestamp': timestamp,
                'latitude': lat,
                'longitude': lon,
                'depth_km': depth,
                'magnitude': magnitude,
                
                # Fault and geological
                'distance_to_fault_km': fault_distance,
                'geological_layer': list(self.geological_layers.keys())[
                    list(self.geological_layers.values()).index(layer_props)
                ],
                
                # Seismic wave properties
                'p_wave_velocity': seismic_params['p_wave_velocity'],
                's_wave_velocity': seismic_params['s_wave_velocity'],
                'density': seismic_params['density'],
                'young_modulus': seismic_params['young_modulus'],
                'poisson_ratio': seismic_params['poisson_ratio'],
                'stress_drop': seismic_params['stress_drop'],
                
                # Rupture characteristics
                'seismic_moment': rupture_params['seismic_moment'],
                'rupture_length': rupture_params['rupture_length'],
                'rupture_width': rupture_params['rupture_width'],
                'average_displacement': rupture_params['average_displacement'],
                'slip_rate': slip_rate,
                
                # Focal mechanism
                'focal_mechanism_strike': focal_params['focal_mechanism_strike'],
                'focal_mechanism_dip': focal_params['focal_mechanism_dip'],
                'focal_mechanism_rake': focal_params['focal_mechanism_rake'],
                'mechanism_type': focal_params['mechanism_type'],
                
                # Regional characteristics
                'crustal_thickness': regional_params['crustal_thickness'],
                'heat_flow': regional_params['heat_flow'],
                'gravity_anomaly': regional_params['gravity_anomaly'],
                'magnetic_anomaly': regional_params['magnetic_anomaly'],
                'topography_elevation': regional_params['topography_elevation'],
                
                # Additional temporal features
                'year': timestamp.year,
                'month': timestamp.month,
                'day_of_year': timestamp.timetuple().tm_yday,
                'hour': timestamp.hour,
            }
            
            data.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some derived features
        df['vp_vs_ratio'] = df['p_wave_velocity'] / df['s_wave_velocity']
        df['energy_magnitude'] = 0.67 * df['magnitude'] + 2.9  # Energy magnitude
        df['moment_magnitude'] = (2/3) * (np.log10(df['seismic_moment']) - 9.1)  # Mw calculation
        df['fault_density'] = 1 / (df['distance_to_fault_km'] + 1)  # Inverse distance to fault
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n✅ Generated {len(df)} earthquake records!")
        print(f"📊 Magnitude range: {df['magnitude'].min():.2f} - {df['magnitude'].max():.2f}")
        print(f"📍 Geographic coverage: {df['latitude'].min():.3f}°N to {df['latitude'].max():.3f}°N")
        print(f"📅 Time range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        
        return df
    
    def add_realistic_noise_and_uncertainties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic measurement uncertainties and noise"""
        
        # Add location uncertainties (GPS accuracy)
        df['latitude'] += np.random.normal(0, 0.0001, len(df))  # ~10m uncertainty
        df['longitude'] += np.random.normal(0, 0.0001, len(df))
        
        # Add depth uncertainties (larger for deeper events)
        depth_uncertainty = 0.05 + 0.01 * df['depth_km']
        df['depth_km'] += np.random.normal(0, depth_uncertainty)
        df['depth_km'] = df['depth_km'].clip(lower=0.1)
        
        # Add magnitude uncertainties
        df['magnitude'] += np.random.normal(0, 0.1, len(df))
        df['magnitude'] = df['magnitude'].clip(lower=0.5, upper=8.0)
        
        return df

# Example usage and testing
if __name__ == "__main__":
    generator = NewDelhiEarthquakeDataGenerator()
    
    # Generate enhanced dataset
    enhanced_df = generator.generate_enhanced_dataset(n_samples=5000)
    enhanced_df = generator.add_realistic_noise_and_uncertainties(enhanced_df)
    
    # Save to CSV
    enhanced_df.to_csv('data/enhanced_earthquake_data.csv', index=False)
    
    print("\n📈 Dataset Summary:")
    print(enhanced_df.describe())
    
    print(f"\n🎯 Target Distribution:")
    print(enhanced_df['magnitude'].value_counts(bins=10).sort_index())
