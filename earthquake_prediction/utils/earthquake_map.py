"""
Interactive Earthquake Map with Live Data Visualization
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from io import StringIO
import os
import time


class EarthquakeMapVisualizer:
    """
    Interactive earthquake map visualizer with live data integration
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the map visualizer
        
        Args:
            api_key (str, optional): API key for external earthquake data sources
        """
        self.api_key = api_key
        self.usgs_base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        self.default_center = [28.6139, 77.2090]  # New Delhi coordinates
        
    def fetch_live_earthquake_data(self, days=7, min_magnitude=4.0, max_results=1000):
        """
        Fetch live earthquake data from USGS API
        
        Args:
            days (int): Number of days to look back
            min_magnitude (float): Minimum magnitude to include
            max_results (int): Maximum number of results
            
        Returns:
            pd.DataFrame: Live earthquake data
        """
        try:
            # Calculate date range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Format dates for USGS API
            start_date = start_time.strftime('%Y-%m-%d')
            end_date = end_time.strftime('%Y-%m-%d')
            
            # Build API request
            params = {
                'format': 'geojson',
                'starttime': start_date,
                'endtime': end_date,
                'minmagnitude': min_magnitude,
                'limit': max_results,
                'orderby': 'time-desc'
            }
            
            print(f"Fetching live earthquake data from {start_date} to {end_date}...")
            
            # Make API request
            response = requests.get(self.usgs_base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            earthquakes = []
            
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                earthquake = {
                    'id': props.get('ids', '').split(',')[0] if props.get('ids') else '',
                    'magnitude': props.get('mag', 0),
                    'place': props.get('place', 'Unknown'),
                    'time': pd.to_datetime(props.get('time', 0), unit='ms'),
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'depth_km': coords[2] if len(coords) > 2 else 0,
                    'magType': props.get('magType', 'unknown'),
                    'rms': props.get('rms', 0),
                    'gap': props.get('gap', 0),
                    'dmin': props.get('dmin', 0),
                    'tsunami': props.get('tsunami', 0),
                    'alert': props.get('alert', None),
                    'status': props.get('status', 'unknown'),
                    'updated': pd.to_datetime(props.get('updated', 0), unit='ms'),
                    'url': props.get('url', ''),
                    'felt': props.get('felt', 0),
                    'cdi': props.get('cdi', 0),
                    'mmi': props.get('mmi', 0)
                }
                earthquakes.append(earthquake)
            
            df = pd.DataFrame(earthquakes)
            print(f"✅ Successfully fetched {len(df)} earthquakes")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching live data: {str(e)}")
            print("Using synthetic data as fallback...")
            return self._generate_sample_earthquake_data()
    
    def _generate_sample_earthquake_data(self):
        """
        Generate sample earthquake data for demonstration
        
        Returns:
            pd.DataFrame: Sample earthquake data
        """
        np.random.seed(42)
        n_samples = 100
        
        # Generate sample data focused around major seismic zones
        seismic_zones = [
            {'lat': 35.0, 'lon': 140.0, 'name': 'Japan'},
            {'lat': -33.9, 'lon': -71.6, 'name': 'Chile'},
            {'lat': 37.8, 'lon': -122.4, 'name': 'California'},
            {'lat': 28.6, 'lon': 77.2, 'name': 'Northern India'},
            {'lat': 41.0, 'lon': 29.0, 'name': 'Turkey'},
        ]
        
        earthquakes = []
        for i in range(n_samples):
            # Choose a random seismic zone
            zone = np.random.choice(seismic_zones)
            
            # Add some variation around the zone
            lat = zone['lat'] + np.random.normal(0, 5)
            lon = zone['lon'] + np.random.normal(0, 5)
            
            # Generate realistic earthquake parameters
            magnitude = np.random.exponential(2) + 4.0  # Most earthquakes are small
            magnitude = min(magnitude, 9.0)  # Cap at realistic maximum
            
            depth = np.random.exponential(15)  # Most earthquakes are shallow
            depth = min(depth, 700)  # Maximum realistic depth
            
            earthquake = {
                'id': f'sample_{i:04d}',
                'magnitude': round(magnitude, 1),
                'place': f'{zone["name"]} region',
                'time': datetime.utcnow() - timedelta(hours=np.random.randint(0, 168)),
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
                'depth_km': round(depth, 1),
                'magType': np.random.choice(['mb', 'ml', 'mw', 'ms']),
                'alert': np.random.choice([None, 'green', 'yellow', 'orange', 'red'], p=[0.7, 0.15, 0.1, 0.04, 0.01]),
                'tsunami': np.random.choice([0, 1], p=[0.95, 0.05]),
                'status': 'reviewed',
                'felt': np.random.poisson(magnitude * 10) if magnitude > 5 else 0
            }
            earthquakes.append(earthquake)
        
        return pd.DataFrame(earthquakes)
    
    def create_interactive_map(self, earthquake_data, center=None, zoom_start=2):
        """
        Create interactive earthquake map using Folium
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data to visualize
            center (list, optional): Map center coordinates [lat, lon]
            zoom_start (int): Initial zoom level
            
        Returns:
            folium.Map: Interactive map object
        """
        if center is None:
            center = self.default_center
        
        # Create base map
        m = folium.Map(
            location=center,
            zoom_start=zoom_start,
            tiles='CartoDB Positron'
        )
        
        # Add different tile layers
        folium.TileLayer('OpenStreetMap').add_to(m)
        folium.TileLayer('CartoDB Dark_Matter').add_to(m)
        # Stamen Terrain with proper attribution
        folium.TileLayer(
            tiles='Stamen Terrain',
            attr='Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
            name='Stamen Terrain'
        ).add_to(m)
        
        # Color mapping for magnitude
        def get_color_from_magnitude(mag):
            if mag < 4.5:
                return 'green'
            elif mag < 5.5:
                return 'yellow'
            elif mag < 6.5:
                return 'orange'
            elif mag < 7.5:
                return 'red'
            else:
                return 'darkred'
        
        # Size mapping for magnitude
        def get_size_from_magnitude(mag):
            return max(3, mag * 3)
        
        # Add earthquake markers
        for _, earthquake in earthquake_data.iterrows():
            # Create popup content
            popup_html = f"""
            <div style="width: 250px;">
                <h4>🌍 Magnitude {earthquake['magnitude']}</h4>
                <hr>
                <b>📍 Location:</b> {earthquake['place']}<br>
                <b>📅 Time:</b> {earthquake['time'].strftime('%Y-%m-%d %H:%M:%S')} UTC<br>
                <b>📏 Coordinates:</b> {earthquake['latitude']:.3f}, {earthquake['longitude']:.3f}<br>
                <b>⬇️ Depth:</b> {earthquake['depth_km']:.1f} km<br>
                <b>🔢 Type:</b> {earthquake.get('magType', 'unknown')}<br>
                {'<b>🌊 Tsunami:</b> Yes<br>' if earthquake.get('tsunami', 0) else ''}
                {'<b>⚠️ Alert:</b> ' + earthquake['alert'].upper() + '<br>' if earthquake.get('alert') else ''}
                {'<b>👥 Felt Reports:</b> ' + str(earthquake['felt']) + '<br>' if earthquake.get('felt', 0) > 0 else ''}
            </div>
            """
            
            # Add marker to map
            folium.CircleMarker(
                location=[earthquake['latitude'], earthquake['longitude']],
                radius=get_size_from_magnitude(earthquake['magnitude']),
                color='black',
                weight=1,
                fillColor=get_color_from_magnitude(earthquake['magnitude']),
                fillOpacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"M{earthquake['magnitude']} - {earthquake['place']}"
            ).add_to(m)
        
        # Add magnitude legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 140px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    ">
        <p style="margin: 10px; font-weight: bold;">Earthquake Magnitude</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:green"></i> &lt; 4.5</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:yellow"></i> 4.5 - 5.5</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:orange"></i> 5.5 - 6.5</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:red"></i> 6.5 - 7.5</p>
        <p style="margin: 10px;"><i class="fa fa-circle" style="color:darkred"></i> &gt; 7.5</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add heatmap layer
        if len(earthquake_data) > 0:
            heat_data = [[row['latitude'], row['longitude'], row['magnitude']] 
                        for idx, row in earthquake_data.iterrows()]
            
            heatmap = plugins.HeatMap(heat_data, radius=15, blur=10)
            heatmap.add_to(m)
        
        # Add cluster markers for better performance with many points
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add measure control
        plugins.MeasureControl().add_to(m)
        
        # Add full screen control
        plugins.Fullscreen().add_to(m)
        
        return m
    
    def create_magnitude_timeline(self, earthquake_data):
        """
        Create interactive timeline of earthquake magnitudes
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data
            
        Returns:
            plotly.graph_objects.Figure: Timeline figure
        """
        if len(earthquake_data) == 0:
            return go.Figure()
        
        # Sort by time
        data = earthquake_data.sort_values('time').copy()
        
        # Create timeline plot
        fig = go.Figure()
        
        # Color mapping for magnitude
        colors = []
        for mag in data['magnitude']:
            if mag < 4.5:
                colors.append('green')
            elif mag < 5.5:
                colors.append('yellow')
            elif mag < 6.5:
                colors.append('orange')
            elif mag < 7.5:
                colors.append('red')
            else:
                colors.append('darkred')
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=data['time'],
            y=data['magnitude'],
            mode='markers',
            marker=dict(
                size=data['magnitude'] * 5,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=data['place'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Magnitude: %{y:.1f}<br>' +
                         'Time: %{x}<br>' +
                         '<extra></extra>',
            name='Earthquakes'
        ))
        
        fig.update_layout(
            title='Earthquake Magnitude Timeline',
            xaxis_title='Date/Time',
            yaxis_title='Magnitude',
            hovermode='closest',
            height=400
        )
        
        return fig
    
    def create_depth_magnitude_plot(self, earthquake_data):
        """
        Create depth vs magnitude scatter plot
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure
        """
        if len(earthquake_data) == 0:
            return go.Figure()
        
        fig = px.scatter(
            earthquake_data,
            x='depth_km',
            y='magnitude',
            color='magnitude',
            size='magnitude',
            hover_data=['place', 'time'],
            color_continuous_scale='Viridis',
            title='Earthquake Depth vs Magnitude'
        )
        
        fig.update_layout(
            xaxis_title='Depth (km)',
            yaxis_title='Magnitude',
            height=400
        )
        
        return fig
    
    def create_magnitude_distribution(self, earthquake_data):
        """
        Create magnitude distribution histogram
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data
            
        Returns:
            plotly.graph_objects.Figure: Histogram figure
        """
        if len(earthquake_data) == 0:
            return go.Figure()
        
        fig = px.histogram(
            earthquake_data,
            x='magnitude',
            nbins=20,
            title='Earthquake Magnitude Distribution',
            color_discrete_sequence=['lightblue']
        )
        
        fig.update_layout(
            xaxis_title='Magnitude',
            yaxis_title='Count',
            height=400
        )
        
        return fig
    
    def create_geographic_distribution(self, earthquake_data):
        """
        Create geographic distribution plot
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data
            
        Returns:
            plotly.graph_objects.Figure: Geographic plot
        """
        if len(earthquake_data) == 0:
            return go.Figure()
        
        fig = px.scatter_geo(
            earthquake_data,
            lat='latitude',
            lon='longitude',
            color='magnitude',
            size='magnitude',
            hover_data=['place', 'time', 'depth_km'],
            color_continuous_scale='Viridis',
            title='Global Earthquake Distribution'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def save_map(self, map_obj, filepath):
        """
        Save interactive map to HTML file
        
        Args:
            map_obj (folium.Map): Map object to save
            filepath (str): Output file path
        """
        try:
            map_obj.save(filepath)
            print(f"✅ Map saved to {filepath}")
        except Exception as e:
            print(f"❌ Error saving map: {str(e)}")
    
    def create_comprehensive_dashboard(self, earthquake_data, save_path=None):
        """
        Create comprehensive earthquake dashboard
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data
            save_path (str, optional): Path to save dashboard
            
        Returns:
            dict: Dictionary containing all visualizations
        """
        dashboard = {}
        
        # Create interactive map
        dashboard['map'] = self.create_interactive_map(earthquake_data)
        
        # Create timeline
        dashboard['timeline'] = self.create_magnitude_timeline(earthquake_data)
        
        # Create depth vs magnitude plot
        dashboard['depth_magnitude'] = self.create_depth_magnitude_plot(earthquake_data)
        
        # Create magnitude distribution
        dashboard['magnitude_dist'] = self.create_magnitude_distribution(earthquake_data)
        
        # Create geographic distribution
        dashboard['geo_dist'] = self.create_geographic_distribution(earthquake_data)
        
        # Save dashboard if path provided
        if save_path:
            # Save map
            map_path = save_path.replace('.html', '_map.html')
            dashboard['map'].save(map_path)
            
            # Save plotly figures
            for name, fig in dashboard.items():
                if name != 'map' and hasattr(fig, 'write_html'):
                    fig_path = save_path.replace('.html', f'_{name}.html')
                    fig.write_html(fig_path)
        
        return dashboard
    
    def get_earthquake_statistics(self, earthquake_data):
        """
        Get comprehensive earthquake statistics
        
        Args:
            earthquake_data (pd.DataFrame): Earthquake data
            
        Returns:
            dict: Statistics summary
        """
        if len(earthquake_data) == 0:
            return {}
        
        stats = {
            'total_earthquakes': len(earthquake_data),
            'magnitude_stats': {
                'min': earthquake_data['magnitude'].min(),
                'max': earthquake_data['magnitude'].max(),
                'mean': earthquake_data['magnitude'].mean(),
                'median': earthquake_data['magnitude'].median(),
                'std': earthquake_data['magnitude'].std()
            },
            'depth_stats': {
                'min': earthquake_data['depth_km'].min(),
                'max': earthquake_data['depth_km'].max(),
                'mean': earthquake_data['depth_km'].mean(),
                'median': earthquake_data['depth_km'].median()
            },
            'magnitude_ranges': {
                'minor (< 4.0)': len(earthquake_data[earthquake_data['magnitude'] < 4.0]),
                'light (4.0-4.9)': len(earthquake_data[(earthquake_data['magnitude'] >= 4.0) & 
                                                     (earthquake_data['magnitude'] < 5.0)]),
                'moderate (5.0-5.9)': len(earthquake_data[(earthquake_data['magnitude'] >= 5.0) & 
                                                        (earthquake_data['magnitude'] < 6.0)]),
                'strong (6.0-6.9)': len(earthquake_data[(earthquake_data['magnitude'] >= 6.0) & 
                                                      (earthquake_data['magnitude'] < 7.0)]),
                'major (7.0-7.9)': len(earthquake_data[(earthquake_data['magnitude'] >= 7.0) & 
                                                     (earthquake_data['magnitude'] < 8.0)]),
                'great (8.0+)': len(earthquake_data[earthquake_data['magnitude'] >= 8.0])
            },
            'time_range': {
                'earliest': earthquake_data['time'].min(),
                'latest': earthquake_data['time'].max(),
                'span_days': (earthquake_data['time'].max() - earthquake_data['time'].min()).days
            }
        }
        
        return stats


def create_streamlit_app():
    """
    Create Streamlit web application for earthquake visualization
    """
    st.set_page_config(
        page_title="🌍 Earthquake Live Map Dashboard",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌍 Earthquake Live Map Dashboard")
    st.markdown("Interactive visualization of earthquake data with live updates")
    
    # Initialize map visualizer
    map_viz = EarthquakeMapVisualizer()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Live USGS Data", "Sample Data", "Upload CSV"]
    )
    
    # Parameters for live data
    if data_source == "Live USGS Data":
        days = st.sidebar.slider("Days to look back", 1, 30, 7)
        min_magnitude = st.sidebar.slider("Minimum magnitude", 1.0, 6.0, 4.0, 0.1)
        max_results = st.sidebar.slider("Maximum results", 100, 2000, 1000)
    
    # Load data button
    if st.sidebar.button("🔄 Load/Refresh Data"):
        with st.spinner("Loading earthquake data..."):
            if data_source == "Live USGS Data":
                earthquake_data = map_viz.fetch_live_earthquake_data(
                    days=days, 
                    min_magnitude=min_magnitude, 
                    max_results=max_results
                )
            elif data_source == "Sample Data":
                earthquake_data = map_viz._generate_sample_earthquake_data()
            
            # Store in session state
            st.session_state.earthquake_data = earthquake_data
    
    # File upload option
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file", 
            type="csv",
            help="CSV should have columns: latitude, longitude, magnitude, time, place, depth_km"
        )
        
        if uploaded_file is not None:
            earthquake_data = pd.read_csv(uploaded_file)
            st.session_state.earthquake_data = earthquake_data
    
    # Check if data exists
    if 'earthquake_data' not in st.session_state:
        st.info("👆 Please load earthquake data using the sidebar controls")
        return
    
    earthquake_data = st.session_state.earthquake_data
    
    # Display statistics
    stats = map_viz.get_earthquake_statistics(earthquake_data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Earthquakes", stats.get('total_earthquakes', 0))
    with col2:
        st.metric("Max Magnitude", f"{stats.get('magnitude_stats', {}).get('max', 0):.1f}")
    with col3:
        st.metric("Avg Magnitude", f"{stats.get('magnitude_stats', {}).get('mean', 0):.1f}")
    with col4:
        st.metric("Time Span (days)", stats.get('time_range', {}).get('span_days', 0))
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Map", "📊 Timeline", "📈 Analysis", "📋 Data"])
    
    with tab1:
        st.subheader("Interactive Earthquake Map")
        
        # Create map
        earthquake_map = map_viz.create_interactive_map(earthquake_data)
        
        # Display map using streamlit-folium
        try:
            import streamlit_folium as st_folium
            st_folium.folium_static(earthquake_map, width=1200, height=600)
        except ImportError:
            st.warning("streamlit-folium not installed. Saving map to HTML file instead.")
            map_path = "temp_earthquake_map.html"
            earthquake_map.save(map_path)
            st.success(f"Map saved to {map_path}")
            
            # Display HTML file download link
            with open(map_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            
            st.download_button(
                label="📥 Download Interactive Map",
                data=html_data,
                file_name="earthquake_map.html",
                mime="text/html"
            )
    
    with tab2:
        st.subheader("Earthquake Timeline")
        timeline_fig = map_viz.create_magnitude_timeline(earthquake_data)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        st.subheader("Magnitude Distribution")
        dist_fig = map_viz.create_magnitude_distribution(earthquake_data)
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Depth vs Magnitude Analysis")
        depth_fig = map_viz.create_depth_magnitude_plot(earthquake_data)
        st.plotly_chart(depth_fig, use_container_width=True)
        
        st.subheader("Global Distribution")
        geo_fig = map_viz.create_geographic_distribution(earthquake_data)
        st.plotly_chart(geo_fig, use_container_width=True)
    
    with tab4:
        st.subheader("Raw Data")
        st.dataframe(earthquake_data)
        
        # Download data button
        csv = earthquake_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="earthquake_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    # Test the earthquake map visualizer
    map_viz = EarthquakeMapVisualizer()
    
    print("🌍 Testing Earthquake Map Visualizer")
    print("=" * 50)
    
    # Fetch live data
    earthquake_data = map_viz.fetch_live_earthquake_data(days=7, min_magnitude=4.0)
    
    # Display statistics
    stats = map_viz.get_earthquake_statistics(earthquake_data)
    print(f"\n📊 Statistics:")
    print(f"Total earthquakes: {stats.get('total_earthquakes', 0)}")
    print(f"Max magnitude: {stats.get('magnitude_stats', {}).get('max', 0):.1f}")
    print(f"Time span: {stats.get('time_range', {}).get('span_days', 0)} days")
    
    # Create comprehensive dashboard
    dashboard = map_viz.create_comprehensive_dashboard(earthquake_data, 'earthquake_dashboard.html')
    
    print(f"\n✅ Dashboard created with {len(dashboard)} components")
    print("Generated files:")
    print("- earthquake_dashboard_map.html")
    print("- earthquake_dashboard_timeline.html") 
    print("- earthquake_dashboard_depth_magnitude.html")
    print("- earthquake_dashboard_magnitude_dist.html")
    print("- earthquake_dashboard_geo_dist.html")
