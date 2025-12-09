import requests
from typing import Dict, Any, Optional

class WeatherLoader:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    def get_weather(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Fetch current weather data for a specific location.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "weather_code", "wind_speed_10m"],
            "timezone": "auto"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Formatted response
            current = data.get('current', {})
            return {
                "temperature": current.get('temperature_2m'),
                "humidity": current.get('relative_humidity_2m'),
                "precipitation": current.get('precipitation'),
                "wind_speed": current.get('wind_speed_10m'),
                "condition_code": current.get('weather_code'),
                "units": data.get('current_units', {})
            }
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return {
                "error": str(e),
                "temperature": None,
                "humidity": None,
                "precipitation": None,
                "wind_speed": None,
                "condition_code": None
            }

    def get_weather_for_default_location(self):
        # Default to a seismic active zone or a generic location (e.g., Tokyo or LA)
        # Let's use San Francisco as a default example
        return self.get_weather(37.7749, -122.4194)
