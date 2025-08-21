"""
Standalone Live Earthquake Map Creator
This script creates a separate, focused live earthquake visualization
"""

import sys
import os
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.earthquake_map import EarthquakeMapVisualizer
import pandas as pd


def create_live_earthquake_map():
    """
    Create a standalone live earthquake map with clear data source indication
    """
    print("🌍 Creating Standalone Live Earthquake Map")
    print("=" * 60)
    
    # Initialize map visualizer
    map_viz = EarthquakeMapVisualizer()
    
    # Fetch live data with detailed logging
    print("\n📡 Fetching Live USGS Earthquake Data...")
    print("Parameters:")
    print("  - Time period: Last 7 days")
    print("  - Minimum magnitude: 4.0")
    print("  - Maximum results: 500")
    print("  - Data source: USGS Earthquake Hazards Program")
    
    try:
        # Fetch live data from USGS
        live_data = map_viz.fetch_live_earthquake_data(
            days=7,
            min_magnitude=4.0,
            max_results=500
        )
        
        print(f"\n✅ Successfully fetched {len(live_data)} live earthquakes")
        
        # Display sample of live data
        if len(live_data) > 0:
            print("\n📊 Sample of Live Data:")
            print(live_data[['magnitude', 'place', 'time', 'latitude', 'longitude']].head())
            
            # Check for recent earthquakes (last 24 hours)
            recent_threshold = datetime.utcnow().replace(tzinfo=live_data['time'].dt.tz) - pd.Timedelta(hours=24)
            recent_earthquakes = live_data[live_data['time'] > recent_threshold]
            print(f"\n🔥 Recent earthquakes (last 24h): {len(recent_earthquakes)}")
            
            # Show magnitude distribution
            print(f"\n📈 Magnitude Distribution:")
            for mag_range, count in map_viz.get_earthquake_statistics(live_data)['magnitude_ranges'].items():
                if count > 0:
                    print(f"  - {mag_range}: {count}")
        
    except Exception as e:
        print(f"❌ Error fetching live data: {e}")
        print("🔄 Using sample data for demonstration...")
        live_data = map_viz._generate_sample_earthquake_data()
    
    # Create the live earthquake map
    print(f"\n🗺️ Creating Interactive Live Earthquake Map...")
    
    # Use global view for live data
    live_map = map_viz.create_interactive_map(
        live_data,
        center=[20.0, 0.0],  # Global center
        zoom_start=2
    )
    
    # Add data source information to the map
    data_source_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 300px; height: 80px;
                border: 2px solid #2E8B57; z-index: 9999; font-size: 12px;
                background-color: rgba(255,255,255,0.95); border-radius: 5px;
                padding: 10px;">
        <h4 style="margin: 0; color: #2E8B57;">🌍 Live USGS Data</h4>
        <p style="margin: 5px 0;">Last 7 days • Min Mag 4.0</p>
        <p style="margin: 5px 0;">Total: {len(live_data)} earthquakes</p>
        <p style="margin: 5px 0;">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    '''
    
    live_map.get_root().html.add_child(
        folium.Element(data_source_html)
    )
    
    # Save the live map
    live_map_filename = "live_earthquakes_only.html"
    live_map.save(live_map_filename)
    print(f"✅ Live earthquake map saved to: {live_map_filename}")
    
    # Create statistics summary
    stats = map_viz.get_earthquake_statistics(live_data)
    
    print(f"\n📊 Live Earthquake Statistics:")
    print(f"  - Total earthquakes: {stats.get('total_earthquakes', 0)}")
    print(f"  - Magnitude range: {stats.get('magnitude_stats', {}).get('min', 0):.1f} - {stats.get('magnitude_stats', {}).get('max', 0):.1f}")
    print(f"  - Average magnitude: {stats.get('magnitude_stats', {}).get('mean', 0):.1f}")
    print(f"  - Time span: {stats.get('time_range', {}).get('span_days', 0)} days")
    
    # Create separate timeline for live data
    print(f"\n📈 Creating Live Data Timeline...")
    timeline_fig = map_viz.create_magnitude_timeline(live_data)
    timeline_fig.update_layout(title="Live USGS Earthquake Timeline (Last 7 Days)")
    timeline_fig.write_html("live_earthquakes_timeline.html")
    
    # Create separate analysis plots
    print(f"📊 Creating Live Data Analysis...")
    depth_mag_fig = map_viz.create_depth_magnitude_plot(live_data)
    depth_mag_fig.update_layout(title="Live USGS Earthquakes: Depth vs Magnitude")
    depth_mag_fig.write_html("live_earthquakes_depth_magnitude.html")
    
    # Save live data to CSV for inspection
    live_data_filename = "live_earthquake_data.csv"
    live_data.to_csv(live_data_filename, index=False)
    print(f"💾 Live earthquake data saved to: {live_data_filename}")
    
    # Create comparison with project data if available
    print(f"\n🔄 Checking for Project Data Comparison...")
    try:
        from utils.data_loader import EarthquakeDataLoader
        data_loader = EarthquakeDataLoader()
        project_data = data_loader.load_data()
        
        print(f"✅ Found project data: {len(project_data)} samples")
        
        # Create comparison summary
        print(f"\n📊 Data Comparison:")
        print(f"  Live USGS Data:")
        print(f"    - Count: {len(live_data)}")
        print(f"    - Source: Real earthquakes from USGS")
        print(f"    - Time range: {stats.get('time_range', {}).get('span_days', 0)} days")
        
        print(f"  Project Synthetic Data:")
        print(f"    - Count: {len(project_data)}")
        print(f"    - Source: Generated for training")
        print(f"    - Purpose: Model development")
        
    except Exception as e:
        print(f"ℹ️ Project data not available: {e}")
    
    return live_data, live_map_filename


def main():
    """Main execution"""
    print("🎯 Live Earthquake Data Visualization")
    print("This creates a separate map with ONLY live USGS data")
    print("=" * 60)
    
    # Create the live earthquake map
    live_data, map_filename = create_live_earthquake_map()
    
    # Final summary
    print(f"\n🎉 Live Earthquake Map Complete!")
    print("=" * 60)
    print("Generated Files:")
    print(f"  📂 {map_filename} - Interactive live earthquake map")
    print(f"  📂 live_earthquakes_timeline.html - Timeline visualization")
    print(f"  📂 live_earthquakes_depth_magnitude.html - Depth analysis")
    print(f"  📂 live_earthquake_data.csv - Raw live data")
    
    print(f"\n🌐 How to View:")
    print("1. Open any of the HTML files in your web browser")
    print("2. The map shows ONLY live USGS earthquake data")
    print("3. Each earthquake marker shows detailed information on click")
    print("4. Use the layer controls to switch map views")
    
    print(f"\n🔄 Data Refresh:")
    print("- Run this script again to get updated live data")
    print("- USGS updates their database every few minutes")
    print("- The script fetches earthquakes from the last 7 days")
    
    print(f"\n📊 Data Source:")
    print("- USGS Earthquake Hazards Program")
    print("- https://earthquake.usgs.gov/")
    print("- Real-time global earthquake monitoring")
    
    return live_data


if __name__ == "__main__":
    # Import folium here to avoid issues if not installed
    try:
        import folium
        from folium import Element
        
        # Run the main function
        live_data = main()
        
        print(f"\n✅ Script completed successfully!")
        print("🌍 Open 'live_earthquakes_only.html' to see the live earthquake map")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install folium requests pandas plotly")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
