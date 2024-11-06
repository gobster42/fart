import numpy as np
from typing import Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class WeatherAnalyzer:
    def __init__(self):
        self.condition_weights = {
            'rain': -0.2,
            'wind': -0.15,
            'temperature': 0.1,
            'visibility': 0.05
        }

    def analyze_weather_impact(self, weather_data: Dict) -> Dict:
        """Analyze weather impact on match conditions"""
        impact_scores = {}
        
        # Rain impact
        rain_intensity = weather_data.get('precipitation', 0)
        impact_scores['rain'] = self._calculate_rain_impact(rain_intensity)
        
        # Wind impact
        wind_speed = weather_data.get('wind_speed', 0)
        wind_direction = weather_data.get('wind_direction', 0)
        impact_scores['wind'] = self._calculate_wind_impact(wind_speed, wind_direction)
        
        # Temperature impact
        temperature = weather_data.get('temperature', 15)
        impact_scores['temperature'] = self._calculate_temperature_impact(temperature)
        
        # Visibility impact
        visibility = weather_data.get('visibility', 10)
        impact_scores['visibility'] = self._calculate_visibility_impact(visibility)
        
        # Calculate overall impact
        total_impact = sum(
            score * self.condition_weights[condition]
            for condition, score in impact_scores.items()
        )
        
        return {
            'total_impact': total_impact,
            'components': impact_scores,
            'recommendation': self._generate_weather_recommendation(impact_scores)
        }
    
    def _calculate_rain_impact(self, intensity: float) -> float:
        """Calculate impact of rain on match conditions"""
        # Scale: 0 (no rain) to 1 (heavy rain)
        return min(intensity / 10.0, 1.0)
    
    def _calculate_wind_impact(self, speed: float, direction: float) -> float:
        """Calculate impact of wind on match conditions"""
        # Scale: 0 (calm) to 1 (strong wind)
        return min(speed / 30.0, 1.0)
    
    def _calculate_temperature_impact(self, temp: float) -> float:
        """Calculate impact of temperature on match conditions"""
        # Optimal temperature range: 15-25°C
        if 15 <= temp <= 25:
            return 0.0
        return abs(temp - 20) / 20
    
    def _calculate_visibility_impact(self, visibility: float) -> float:
        """Calculate impact of visibility on match conditions"""
        # Scale: 0 (clear) to 1 (poor visibility)
        return max(0, 1 - visibility / 10.0)
    
    def _generate_weather_recommendation(self, impacts: Dict) -> str:
        """Generate weather-based betting recommendation"""
        if impacts['rain'] > 0.7 or impacts['wind'] > 0.7:
            return "High-risk conditions - consider reducing stake"
        elif impacts['temperature'] > 0.5:
            return "Sub-optimal temperature - monitor team adaptation"
        return "Favorable conditions for normal betting strategy"