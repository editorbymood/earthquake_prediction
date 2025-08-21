"""
Demo Script for Earthquake Map Visualization
This script demonstrates the earthquake map features with live data
"""

import os
import sys
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.earthquake_map import EarthquakeMapVisualizer
import pandas as pd


def main():
    """
    Main demo function showcasing earthquake map capabilities
    """
    print("🌍 Earthquake Map Visualization Demo")
    print("=" * 60)
    
    # Initialize the map visualizer
    map_viz = EarthquakeMapVisualizer()
    
    # Demo 1: Fetch live earthquake data
    print("\n📡 Demo 1: Fetching Live Earthquake Data")
    print("-" * 40)
    
    try:
        # Fetch live data from USGS (last 7 days, magnitude 4.0+)
        earthquake_data = map_viz.fetch_live_earthquake_data(
            days=7, 
            min_magnitude=4.0, 
            max_results=100
        )
        
        print(f"✅ Successfully loaded {len(earthquake_data)} earthquakes")
        
        # Display sample data
        if len(earthquake_data) > 0:
            print("\n📊 Sample of the data:")
            print(earthquake_data[['magnitude', 'place', 'time', 'latitude', 'longitude']].head())
        
    except Exception as e:
        print(f"❌ Error loading live data: {e}")
        print("📊 Using sample data instead...")
        earthquake_data = map_viz._generate_sample_earthquake_data()
    
    # Demo 2: Create interactive map
    print(f"\n🗺️ Demo 2: Creating Interactive Map")
    print("-" * 40)
    
    # Create the interactive map
    earthquake_map = map_viz.create_interactive_map(earthquake_data)
    
    # Save map to HTML file
    map_filename = "earthquake_live_map.html"
    earthquake_map.save(map_filename)
    print(f"✅ Interactive map saved to: {map_filename}")
    
    # Demo 3: Generate statistics
    print(f"\n📈 Demo 3: Earthquake Statistics")
    print("-" * 40)
    
    stats = map_viz.get_earthquake_statistics(earthquake_data)
    
    if stats:
        print(f"Total Earthquakes: {stats['total_earthquakes']}")
        print(f"Magnitude Range: {stats['magnitude_stats']['min']:.1f} - {stats['magnitude_stats']['max']:.1f}")
        print(f"Average Magnitude: {stats['magnitude_stats']['mean']:.1f}")
        print(f"Time Span: {stats['time_range']['span_days']} days")
        print(f"Latest Event: {stats['time_range']['latest']}")
        
        print("\n📊 Magnitude Categories:")
        for category, count in stats['magnitude_ranges'].items():
            if count > 0:
                print(f"  - {category}: {count}")
    
    # Demo 4: Create comprehensive dashboard
    print(f"\n🎯 Demo 4: Creating Comprehensive Dashboard")
    print("-" * 40)
    
    # Create all visualizations
    dashboard = map_viz.create_comprehensive_dashboard(
        earthquake_data, 
        save_path="earthquake_dashboard.html"
    )
    
    print(f"✅ Dashboard created with {len(dashboard)} components:")
    print("  - Interactive map with multiple layers")
    print("  - Timeline of earthquake magnitudes")
    print("  - Depth vs magnitude analysis")
    print("  - Magnitude distribution histogram")
    print("  - Global geographic distribution")
    
    # Demo 5: Display file outputs
    print(f"\n📁 Generated Files:")
    print("-" * 40)
    
    output_files = [
        "earthquake_live_map.html",
        "earthquake_dashboard_map.html",
        "earthquake_dashboard_timeline.html",
        "earthquake_dashboard_depth_magnitude.html",
        "earthquake_dashboard_magnitude_dist.html",
        "earthquake_dashboard_geo_dist.html"
    ]
    
    for filename in output_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"  ✅ {filename} ({file_size:.1f} KB)")
        else:
            print(f"  ❌ {filename} (not found)")
    
    # Demo 6: Integration suggestions
    print(f"\n🔧 Integration with Your Earthquake Prediction Project:")
    print("-" * 40)
    print("1. Add map visualization to your main.py:")
    print("   from utils.earthquake_map import EarthquakeMapVisualizer")
    print("   map_viz = EarthquakeMapVisualizer()")
    print("   map_viz.create_comprehensive_dashboard(your_data)")
    
    print("\n2. Launch the Streamlit web app:")
    print("   streamlit run earthquake_map_app.py")
    
    print("\n3. Integrate with your existing visualizer:")
    print("   Add map methods to utils/visualizer.py")
    
    print("\n4. Use live data for real-time predictions:")
    print("   Fetch live data and apply your trained models")
    
    # Final summary
    print(f"\n🎉 Demo Complete!")
    print("=" * 60)
    print("Open the generated HTML files in your web browser to explore the interactive visualizations.")
    print("For the full web dashboard experience, install streamlit-folium and run:")
    print("  pip install streamlit-folium")
    print("  streamlit run earthquake_map_app.py")
    

def integration_example():
    """
    Example of how to integrate the map with existing project data
    """
    print("\n🔗 Integration Example with Project Data")
    print("-" * 50)
    
    # Load existing project data if available
    try:
        from utils.data_loader import EarthquakeDataLoader
        
        data_loader = EarthquakeDataLoader()
        df = data_loader.load_data()
        
        print(f"✅ Loaded project data: {len(df)} samples")
        
        # Convert to map-compatible format
        if all(col in df.columns for col in ['latitude', 'longitude', 'magnitude']):
            # Add required timestamp column if missing
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            
            # Add required place column if missing
            if 'place' not in df.columns:
                df['place'] = f"Synthetic Location"
            
            # Add depth column if missing
            if 'depth_km' not in df.columns:
                df['depth_km'] = df.get('depth_km', 10.0)  # Default depth
            
            # Create map with project data
            map_viz = EarthquakeMapVisualizer()
            project_map = map_viz.create_interactive_map(df.head(100))  # Limit for demo
            project_map.save("project_data_map.html")
            
            print("✅ Created map using your project's synthetic data")
            print("📁 Saved to: project_data_map.html")
            
        else:
            print("ℹ️ Project data doesn't have required columns for mapping")
            print("   Required: latitude, longitude, magnitude")
            print(f"   Available: {list(df.columns)}")
            
    except Exception as e:
        print(f"ℹ️ Could not load project data: {e}")
        print("   This is normal if you haven't run the main project yet")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Run integration example
    integration_example()
    
    print(f"\n🚀 Next Steps:")
    print("1. Open the generated HTML files to explore the maps")
    print("2. Install additional dependencies for full functionality:")
    print("   pip install streamlit-folium")
    print("3. Launch the web dashboard:")
    print("   streamlit run earthquake_map_app.py")
    print("4. Integrate the map features into your existing project workflow")
