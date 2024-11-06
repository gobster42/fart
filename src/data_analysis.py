import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_elo_rating(self, team_data: Dict) -> float:
        """Calculate ELO rating based on match history"""
        K = 32  # K-factor
        base_rating = 1500
        current_rating = base_rating
        
        for match in team_data['history']:
            opponent_rating = match['opponent_rating']
            expected_score = 1 / (1 + 10 ** ((opponent_rating - current_rating) / 400))
            actual_score = match['result']
            current_rating += K * (actual_score - expected_score)
            
        return current_rating
    
    def analyze_head_to_head(self, matches: List[Dict]) -> Dict:
        """Advanced head-to-head analysis"""
        stats = {
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': [],
            'goals_conceded': [],
            'win_streak': 0,
            'form_rating': 0,
            'dominance_index': 0
        }
        
        current_streak = 0
        total_matches = len(matches)
        
        for idx, match in enumerate(matches):
            # Basic stats
            if match['result'] == 'W':
                stats['wins'] += 1
                current_streak += 1
            elif match['result'] == 'D':
                stats['draws'] += 1
                current_streak = 0
            else:
                stats['losses'] += 1
                current_streak = 0
                
            stats['win_streak'] = max(stats['win_streak'], current_streak)
            stats['goals_scored'].append(match['goals_for'])
            stats['goals_conceded'].append(match['goals_against'])
            
            # Calculate recency-weighted form
            recency_weight = 1 - (idx / total_matches)
            stats['form_rating'] += match['performance_rating'] * recency_weight
            
        # Calculate dominance index
        goals_scored_array = np.array(stats['goals_scored'])
        goals_conceded_array = np.array(stats['goals_conceded'])
        stats['dominance_index'] = np.mean(goals_scored_array - goals_conceded_array)
        
        return stats
    
    def calculate_performance_metrics(self, team_data: Dict) -> Dict:
        """Calculate advanced performance metrics"""
        metrics = {
            'attack_strength': 0,
            'defense_stability': 0,
            'momentum_score': 0,
            'consistency_rating': 0,
            'pressure_handling': 0
        }
        
        # Attack strength calculation
        goals_scored = np.array(team_data['goals_scored'])
        xG = np.array(team_data['expected_goals'])
        metrics['attack_strength'] = np.mean(goals_scored - xG)
        
        # Defense stability
        goals_conceded = np.array(team_data['goals_conceded'])
        xGA = np.array(team_data['expected_goals_against'])
        metrics['defense_stability'] = np.mean(xGA - goals_conceded)
        
        # Momentum score
        recent_form = team_data['recent_form']
        weights = np.exp(np.linspace(-1, 0, len(recent_form)))
        metrics['momentum_score'] = np.average(recent_form, weights=weights)
        
        # Consistency rating
        performance_ratings = np.array(team_data['performance_ratings'])
        metrics['consistency_rating'] = 1 / (np.std(performance_ratings) + 1)
        
        # Pressure handling
        high_pressure_matches = team_data['high_pressure_performances']
        metrics['pressure_handling'] = np.mean(high_pressure_matches)
        
        return metrics
    
    def analyze_team_style(self, match_data: List[Dict]) -> Dict:
        """Analyze team playing style and patterns"""
        style_metrics = {
            'possession_avg': [],
            'pass_accuracy': [],
            'shots_per_game': [],
            'shot_conversion_rate': [],
            'pressing_intensity': [],
            'defensive_line_height': []
        }
        
        for match in match_data:
            style_metrics['possession_avg'].append(match['possession'])
            style_metrics['pass_accuracy'].append(match['pass_accuracy'])
            style_metrics['shots_per_game'].append(len(match['shots']))
            style_metrics['shot_conversion_rate'].append(match['goals'] / max(1, len(match['shots'])))
            style_metrics['pressing_intensity'].append(match['pressing_actions'])
            style_metrics['defensive_line_height'].append(match['avg_defensive_line'])
            
        return {k: np.mean(v) for k, v in style_metrics.items()}