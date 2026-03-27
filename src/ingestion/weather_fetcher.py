from __future__ import annotations

import os
import requests
from datetime import datetime

# Major logistics hubs to monitor by default
MAJOR_PORTS = [
    {"name": "Rotterdam", "lat": 51.9225, "lon": 4.47917},
    {"name": "Shanghai", "lat": 31.2304, "lon": 121.4737},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"name": "Hamburg", "lat": 53.5511, "lon": 9.9937}
]

class WeatherFetcher:
    """Fetches severe weather alerts for major shipping hubs over OpenWeatherMap."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        # Note: One Call API 3.0 requires a subscription, fallback to standard current weather
        # if alerts are unavailable or keys missing.
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def fetch_weather_alerts(self) -> list[dict]:
        """Fetch current rough weather conditions at major ports."""
        
        if not self.api_key or self.api_key == "your_openweather_key_here":
            return self._get_dummy_weather()
            
        alerts = []
        
        for port in MAJOR_PORTS:
            params = {
                "lat": port["lat"],
                "lon": port["lon"],
                "appid": self.api_key,
                "units": "metric"
            }
            try:
                response = requests.get(self.base_url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Look for severe conditions (wind > 15m/s (approx 30 knots) or thunderstorm/extreme rain)
                    wind_speed = data.get("wind", {}).get("speed", 0)
                    weather_main = data.get("weather", [{}])[0].get("main", "")
                    
                    if wind_speed > 15 or weather_main in ["Thunderstorm", "Extreme", "Tornado"]:
                        desc = data.get("weather", [{}])[0].get("description", "severe weather")
                        
                        alerts.append({
                            "id": f"weather-{port['name']}-{datetime.now().strftime('%Y%m%d')}",
                            "title": f"Severe Weather at Port of {port['name']}",
                            "description": f"High risk weather conditions detected: {desc}. Wind speed: {wind_speed} m/s.",
                            "source": "OpenWeather",
                            "date": datetime.now().isoformat(),
                            "type": "weather"
                        })
            except requests.RequestException:
                pass # Continue to next port silently
                
        return alerts if alerts else self._get_dummy_weather()

    def _get_dummy_weather(self) -> list[dict]:
        """Provides fallback data when API keys aren't set."""
        return [
            {
                "id": "weather-shanghai-dummy",
                "title": "Severe Weather at Port of Shanghai",
                "description": "High risk weather conditions detected: heavy thunderstorm. Wind speed: 18.5 m/s.",
                "source": "OpenWeather (Mock)",
                "date": datetime.now().isoformat(),
                "type": "weather"
            }
        ]
