import numpy as np
from typing import Dict, List
import pandas as pd
from scipy.stats import exponweighting
import logging

logger = logging.getLogger(__name__)

class FormAnalyzer:
    def __init__(self):
        self.form_window = 10  # Last 10 matches
        self.recent_weight = 2.0  # Weight for most recent matches
        
    def analyze_team_form(self, match_history: List[Dict]) -> Dict:
        """Analyze team form using advanced metrics"""
        recent_matches = match_history[-self.form_window:]
        
        # Calculate weighted form
        weights = np.exp(np.linspace(-1, 0, len(recent_matches)))
        weighted_scores = [
            match['performance_score'] * weight
            for match, weight in zip(recent_matches, weights)
        ]
        
        # Performance trends
        performance_trend = self._calculate_performance_trend(recent_matches)
        
        # Scoring patterns
        scoring_patterns = self._analyze_scoring_patterns(recent_matches)
        
        # Opposition strength consideration
        adjusted_form = self._adjust_for_opposition(recent_matches)
        
        return {
            'weighted_form': np.average(weighted_scores, weights=weights),
            'trend': performance_trend,
            'scoring_patterns': scoring_patterns,
            'adjusted_form': adjusted_form,
            'consistency': self._calculate_consistency(recent_matches),
            'momentum': self._calculate_momentum(recent_matches)
        }
    
    def _calculate_performance_trend(self, matches: List[Dict]) -> Dict:
        """Calculate performance trend over recent matches"""
        scores = [match['performance_score'] for match in matches]
        trend = np.polyfit(range(len(scores)), scores, 1)[0]
        
        return {
            'direction': 'improving' if trend > 0.1 else 'declining' if trend < -0.1 else 'stable',
            'magnitude': abs(trend),
            'raw_trend': trend
        }
    
    def _analyze_scoring_patterns(self, matches: List[Dict]) -> Dict:
        """Analyze team's scoring patterns"""
        goals_scored = [match['goals_scored'] for match in matches]
        goals_conceded = [match['goals_conceded'] for match in matches]
        
        return {
            'avg_goals_scored': np.mean(goals_scored),
            'avg_goals_conceded': np.mean(goals_conceded),
            'scoring_consistency': np.std(goals_scored),
            'defensive_consistency': np.std(goals_conceded),
            'clean_sheets': sum(1 for g in goals_conceded if g == 0),
            'scoring_streak': self._calculate_streak(goals_scored)
        }
    
    def _adjust_for_opposition(self, matches: List[Dict]) -> float:
        """Adjust form based on opposition strength"""
        adjusted_scores = []
        
        for match in matches:
            opposition_strength = match['opposition_ranking'] / 20.0  # Normalize to 0-1
            performance_score = match['performance_score']
            adjusted_scores.append(performance_score * (1 + opposition_strength))
            
        return np.mean(adjusted_scores)
    
    def _calculate_consistency(self, matches: List[Dict]) -> float:
        """Calculate team's consistency rating"""
        performances = [match['performance_score'] for match in matches]
        return 1 / (np.std(performances) + 1)  # Normalize to 0-1
    
    def _calculate_momentum(self, matches: List[Dict]) -> float:
        """Calculate team's momentum score"""
        recent_results = [match['result'] for match in matches[-3:]]  # Last 3 matches
        momentum_scores = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        
        weighted_momentum = sum(
            momentum_scores[result] * weight
            for result, weight in zip(recent_results, [3, 2, 1])  # More weight to recent matches
        ) / 6  # Normalize to 0-1
        
        return weighted_momentum
    
    def _calculate_streak(self, goals: List[int]) -> int:
        """Calculate current scoring/clean sheet streak"""
        streak = 0
        for goal in reversed(goals):
            if goal > 0:
                streak += 1
            else:
                break
        return streak