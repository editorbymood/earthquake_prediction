"""
Simple HTTP Server to View Earthquake Maps
This creates a local web server to easily view all your earthquake visualizations
"""

import http.server
import socketserver
import webbrowser
import os
import threading
import time
from pathlib import Path


class EarthquakeMapServer:
    """Simple HTTP server for viewing earthquake maps"""
    
    def __init__(self, port=8000):
        self.port = port
        self.server = None
        self.thread = None
        
    def start_server(self):
        """Start the HTTP server in a separate thread"""
        try:
            # Change to the project directory
            os.chdir(Path(__file__).parent)
            
            # Create server
            handler = http.server.SimpleHTTPRequestHandler
            self.server = socketserver.TCPServer(("", self.port), handler)
            
            print(f"🌐 Starting local web server at http://localhost:{self.port}")
            print("📂 Serving files from:", os.getcwd())
            
            # Start server in background thread
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop the HTTP server"""
        if self.server:
            self.server.shutdown()
            print("🛑 Server stopped")
    
    def list_available_maps(self):
        """List all available earthquake map files"""
        map_files = []
        
        # Look for HTML files
        html_files = [f for f in os.listdir('.') if f.endswith('.html')]
        
        # Categorize the files
        categories = {
            'Live Data Maps': [],
            'Project Dashboard': [],
            'Analysis Plots': [],
            'Other': []
        }
        
        for file in html_files:
            if 'live_earthquake' in file:
                categories['Live Data Maps'].append(file)
            elif 'dashboard' in file:
                categories['Project Dashboard'].append(file)
            elif any(keyword in file for keyword in ['timeline', 'magnitude', 'depth', 'geo']):
                categories['Analysis Plots'].append(file)
            else:
                categories['Other'].append(file)
        
        return categories


def main():
    """Main function to start the map viewer"""
    print("🌍 Earthquake Map Viewer")
    print("=" * 50)
    
    # Initialize server
    server = EarthquakeMapServer(port=8001)  # Use 8001 to avoid conflicts
    
    # List available maps
    print("📋 Scanning for earthquake maps...")
    map_categories = server.list_available_maps()
    
    total_maps = sum(len(files) for files in map_categories.values())
    if total_maps == 0:
        print("❌ No HTML map files found!")
        print("💡 Try running one of these first:")
        print("   - python create_live_earthquake_map.py")
        print("   - python demo_earthquake_map.py")
        print("   - python main.py")
        return
    
    print(f"✅ Found {total_maps} earthquake maps:")
    print()
    
    # Display available maps
    for category, files in map_categories.items():
        if files:
            print(f"📁 {category}:")
            for i, file in enumerate(files, 1):
                file_size = os.path.getsize(file) / 1024  # KB
                print(f"   {i}. {file} ({file_size:.1f} KB)")
            print()
    
    # Start server
    if server.start_server():
        print("🚀 Server started successfully!")
        print()
        print("🌐 Available Maps:")
        print("-" * 30)
        
        # Generate direct links
        for category, files in map_categories.items():
            if files:
                print(f"\n📁 {category}:")
                for file in files:
                    url = f"http://localhost:{server.port}/{file}"
                    print(f"   🔗 {url}")
        
        print("\n" + "=" * 50)
        print("🎯 Quick Access:")
        print("=" * 50)
        
        # Highlight the most important maps
        if map_categories['Live Data Maps']:
            live_map = map_categories['Live Data Maps'][0]
            print(f"🌍 Live Earthquake Map:")
            print(f"   http://localhost:{server.port}/{live_map}")
            
            # Auto-open the live map
            print("\n⏳ Opening live earthquake map in your browser...")
            time.sleep(2)
            webbrowser.open(f"http://localhost:{server.port}/{live_map}")
        
        # Instructions
        print(f"\n📖 Instructions:")
        print("1. Click any link above to view earthquake maps")
        print("2. Maps will open in your default web browser")
        print("3. Interactive maps support zooming, clicking markers, and layer switching")
        print("4. Press Ctrl+C to stop the server when done")
        
        try:
            print(f"\n🔄 Server running... Press Ctrl+C to stop")
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n👋 Shutting down server...")
            server.stop_server()
            print("✅ Server stopped. Goodbye!")
    
    else:
        print("❌ Failed to start server")


if __name__ == "__main__":
    main()
