# 🌍 Interactive Earthquake Map Feature

This enhancement adds comprehensive interactive mapping capabilities to your earthquake prediction project, featuring live data integration and web-based visualization.

## 🚀 New Features Added

### 1. **Interactive Map Visualizations**
- **Folium-based interactive maps** with multiple tile layers
- **Color-coded earthquake markers** by magnitude
- **Detailed popup information** for each earthquake
- **Heatmap overlay** for spatial patterns
- **Cluster markers** for better performance

### 2. **Live Data Integration**
- **Real-time USGS API integration** for current earthquake data
- **Automatic fallback** to synthetic data if API is unavailable
- **Configurable date ranges** and magnitude filters
- **Comprehensive earthquake statistics**

### 3. **Web Dashboard**
- **Streamlit-based web application** for interactive exploration
- **Multiple visualization tabs** (Map, Timeline, Analysis, Data)
- **Real-time data loading** and refresh capabilities
- **Data export features** (CSV downloads)

### 4. **Enhanced Analytics**
- **Timeline visualizations** of earthquake activity
- **Depth vs magnitude analysis**
- **Geographic distribution plots**
- **Statistical summaries** and breakdowns

## 📁 Files Added

```
earthquake_prediction/
├── utils/
│   └── earthquake_map.py         # Core mapping functionality
├── earthquake_map_app.py          # Streamlit web app launcher
├── demo_earthquake_map.py         # Demonstration script
└── MAP_FEATURE_README.md          # This documentation
```

## 📦 Dependencies Added

The following packages have been added to `requirements.txt`:

```txt
folium>=0.12.0              # Interactive mapping
streamlit>=1.20.0           # Web dashboard framework
streamlit-folium>=0.11.0    # Streamlit-Folium integration
```

## 🎯 Quick Start

### 1. Install New Dependencies
```bash
pip install folium streamlit streamlit-folium
```

### 2. Run the Demo
```bash
python demo_earthquake_map.py
```

### 3. Launch Web Dashboard
```bash
streamlit run earthquake_map_app.py
```

### 4. Integrate with Main Pipeline
```bash
python main.py
```

## 🔧 Usage Examples

### Basic Map Creation
```python
from utils.earthquake_map import EarthquakeMapVisualizer

# Initialize visualizer
map_viz = EarthquakeMapVisualizer()

# Fetch live earthquake data
earthquake_data = map_viz.fetch_live_earthquake_data(days=7, min_magnitude=4.0)

# Create interactive map
earthquake_map = map_viz.create_interactive_map(earthquake_data)
earthquake_map.save('earthquake_map.html')
```

### Integration with Existing Project
```python
from utils.visualizer import EarthquakeVisualizer

visualizer = EarthquakeVisualizer()

# Create earthquake map from project data
map_path = visualizer.create_earthquake_map(your_dataframe)

# Create live earthquake dashboard
live_dashboard = visualizer.create_live_earthquake_dashboard()
```

### Web Dashboard
```python
# Launch the Streamlit app
streamlit run earthquake_map_app.py
```

## 🗺️ Map Features

### **Interactive Elements**
- **Pan and zoom** functionality
- **Multiple base layers** (OpenStreetMap, CartoDB, Stamen)
- **Marker clustering** for performance
- **Full-screen mode**
- **Measurement tools**

### **Data Visualization**
- **Color coding** by magnitude:
  - 🟢 Green: < 4.5
  - 🟡 Yellow: 4.5 - 5.5
  - 🟠 Orange: 5.5 - 6.5
  - 🔴 Red: 6.5 - 7.5
  - 🔴 Dark Red: > 7.5

### **Information Display**
- **Detailed popups** with earthquake metadata
- **Hover tooltips** for quick information
- **Legend** for magnitude ranges
- **Statistics panel** with key metrics

## 📊 Dashboard Components

### **🗺️ Interactive Map Tab**
- Main earthquake map with all interactive features
- Real-time data display
- Export functionality

### **📊 Timeline Tab**
- Temporal analysis of earthquake activity
- Magnitude distribution histograms
- Interactive plotly charts

### **📈 Analysis Tab**
- Depth vs magnitude scatter plots
- Global distribution visualization
- Advanced analytics

### **📋 Data Tab**
- Raw data table with sorting and filtering
- CSV export functionality
- Data statistics

## 🌐 Live Data Sources

### **USGS Earthquake API**
- **Real-time data** from the US Geological Survey
- **Global coverage** with detailed metadata
- **Configurable parameters**:
  - Date range (1-30 days)
  - Minimum magnitude (1.0-6.0)
  - Maximum results (100-2000)

### **Data Fields**
- Magnitude, location, time
- Coordinates (latitude, longitude)
- Depth, magnitude type
- Alert levels, tsunami warnings
- Felt reports, intensity measurements

## 🔧 Configuration Options

### **Map Settings**
```python
# Customize map appearance
map_viz = EarthquakeMapVisualizer()
earthquake_map = map_viz.create_interactive_map(
    earthquake_data,
    center=[lat, lon],      # Custom center point
    zoom_start=5            # Initial zoom level
)
```

### **Data Fetching**
```python
# Configure live data parameters
earthquake_data = map_viz.fetch_live_earthquake_data(
    days=14,                # Look back 14 days
    min_magnitude=3.5,      # Include smaller earthquakes
    max_results=500         # More results
)
```

## 🎨 Integration with Existing Visualizer

The map functionality has been seamlessly integrated into your existing `EarthquakeVisualizer` class:

### **New Methods Added**
- `create_earthquake_map(df)` - Create map from project data
- `create_live_earthquake_dashboard()` - Real-time dashboard
- `_prepare_data_for_mapping(df)` - Data preparation helper

### **Enhanced Main Pipeline**
Your `main.py` now automatically:
1. Creates interactive maps from synthetic data
2. Fetches and visualizes live earthquake data
3. Generates comprehensive dashboards
4. Saves all visualizations

## 🚀 Advanced Features

### **Custom Data Integration**
```python
# Use your own earthquake data
custom_data = pd.DataFrame({
    'latitude': [...],
    'longitude': [...],
    'magnitude': [...],
    'time': [...],
    'place': [...],
    'depth_km': [...]
})

map_viz = EarthquakeMapVisualizer()
custom_map = map_viz.create_interactive_map(custom_data)
```

### **Comprehensive Dashboard**
```python
# Create full dashboard with all visualizations
dashboard = map_viz.create_comprehensive_dashboard(
    earthquake_data,
    save_path='my_dashboard.html'
)
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Missing Dependencies**
   ```bash
   pip install folium streamlit streamlit-folium
   ```

2. **API Connection Issues**
   - The system automatically falls back to synthetic data
   - Check internet connection for live data

3. **Streamlit-Folium Issues**
   - Install: `pip install streamlit-folium`
   - Alternative: Maps are saved as HTML files

### **Data Requirements**
For mapping, your data needs these columns:
- `latitude` (required)
- `longitude` (required) 
- `magnitude` (required)
- `time` (auto-generated if missing)
- `place` (auto-generated if missing)
- `depth_km` (auto-generated if missing)

## 📈 Performance Considerations

- **Marker clustering** automatically handles large datasets
- **Data limiting** prevents browser overload
- **Progressive loading** for better user experience
- **Efficient rendering** with optimized HTML output

## 🎯 Next Steps

### **Immediate Actions**
1. Install new dependencies
2. Run the demo script
3. Launch the web dashboard
4. Integrate with your existing workflow

### **Advanced Usage**
1. Customize map styling and colors
2. Add prediction overlays
3. Implement real-time updates
4. Create custom data sources

### **Integration Ideas**
1. **Prediction Visualization** - Show model predictions on the map
2. **Risk Assessment** - Color-code areas by predicted risk
3. **Real-time Monitoring** - Auto-refresh with new data
4. **Alert System** - Notifications for significant events

## 🎉 Benefits

✅ **Enhanced Visualization** - Interactive maps are more engaging than static plots
✅ **Real-time Data** - Stay current with live earthquake information  
✅ **Web-based Interface** - Easy sharing and presentation
✅ **Comprehensive Analytics** - Multiple visualization perspectives
✅ **Seamless Integration** - Works with your existing project structure
✅ **Professional Output** - Publication-ready interactive visualizations

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify internet connection for live data features
4. Review the demo script for usage examples

The map feature significantly enhances your earthquake prediction project by adding interactive visualization capabilities and live data integration, making it more comprehensive and professional for presentations, analysis, and real-world applications.
