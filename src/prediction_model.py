import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, Tuple, List
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import logging
from data_analysis import AdvancedAnalytics

logger = logging.getLogger(__name__)

class EnhancedPredictionModel:
    def __init__(self):
        self.analytics = AdvancedAnalytics()
        self.models = {
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'catboost': CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.05,
                random_state=42,
                verbose=False
            )
        }
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def prepare_advanced_features(self, match_data: Dict) -> np.ndarray:
        """Extract and prepare advanced features from match data"""
        features = []
        feature_names = []
        
        # Team performance metrics with recent form weighting
        home_metrics = self.analytics.calculate_performance_metrics(match_data['home_team_data'], weighted=True)
        away_metrics = self.analytics.calculate_performance_metrics(match_data['away_team_data'], weighted=True)
        
        # Dynamic ELO ratings with competition weight
        home_elo = self.analytics.calculate_elo_rating(match_data['home_team_data'], competition_weight=True)
        away_elo = self.analytics.calculate_elo_rating(match_data['away_team_data'], competition_weight=True)
        
        # Enhanced head-to-head analysis
        h2h_stats = self.analytics.analyze_head_to_head(match_data['h2h_matches'], recency_weighted=True)
        
        # Advanced team style analysis with opposition strength consideration
        home_style = self.analytics.analyze_team_style(match_data['home_team_matches'], opposition_strength=True)
        away_style = self.analytics.analyze_team_style(match_data['away_team_matches'], opposition_strength=True)
        
        # Weather and pitch conditions impact
        conditions_impact = self.analytics.analyze_conditions_impact(match_data['conditions'])
        
        # Historical performance in similar conditions
        historical_performance = self.analytics.analyze_historical_performance(
            match_data['historical_matches'],
            conditions_similar=True
        )
        
        # Combine all features with proper naming
        feature_dict = {
            # Performance metrics
            'home_attack': home_metrics['attack_strength'],
            'home_defense': home_metrics['defense_stability'],
            'home_momentum': home_metrics['momentum_score'],
            'home_consistency': home_metrics['consistency_rating'],
            'home_pressure': home_metrics['pressure_handling'],
            'away_attack': away_metrics['attack_strength'],
            'away_defense': away_metrics['defense_stability'],
            'away_momentum': away_metrics['momentum_score'],
            'away_consistency': away_metrics['consistency_rating'],
            'away_pressure': away_metrics['pressure_handling'],
            
            # ELO and ratings
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_difference': home_elo - away_elo,
            
            # H2H features
            'h2h_dominance': h2h_stats['dominance_index'],
            'h2h_form': h2h_stats['form_rating'],
            'h2h_streaks': h2h_stats['win_streak'],
            
            # Style metrics
            'home_possession': home_style['possession_avg'],
            'home_pass_acc': home_style['pass_accuracy'],
            'home_conversion': home_style['shot_conversion_rate'],
            'home_pressing': home_style['pressing_intensity'],
            'away_possession': away_style['possession_avg'],
            'away_pass_acc': away_style['pass_accuracy'],
            'away_conversion': away_style['shot_conversion_rate'],
            'away_pressing': away_style['pressing_intensity'],
            
            # Conditions impact
            'weather_impact': conditions_impact['weather_factor'],
            'pitch_impact': conditions_impact['pitch_factor'],
            
            # Historical performance
            'home_historical': historical_performance['home_team'],
            'away_historical': historical_performance['away_team']
        }
        
        # Convert to ordered lists
        features = list(feature_dict.values())
        feature_names = list(feature_dict.keys())
        
        return np.array(features).reshape(1, -1), feature_names

    def ensemble_predict(self, match_data: Dict) -> Tuple[str, float, Dict]:
        """Make predictions using enhanced ensemble of models"""
        try:
            features, feature_names = self.prepare_advanced_features(match_data)
            scaled_features = self.scaler.transform(features)
            
            predictions = {}
            probabilities = {}
            feature_importances = {}
            
            # Get predictions from all models
            for name, model in self.models.items():
                pred = model.predict(scaled_features)[0]
                probs = model.predict_proba(scaled_features)[0]
                predictions[name] = pred
                probabilities[name] = probs
                
                # Calculate feature importance for each model
                if hasattr(model, 'feature_importances_'):
                    feature_importances[name] = dict(zip(feature_names, model.feature_importances_))
            
            # Dynamic weighted ensemble based on recent performance
            weights = self.calculate_dynamic_weights(match_data)
            
            final_probs = np.zeros(3)  # For 3 possible outcomes
            for name, probs in probabilities.items():
                final_probs += probs * weights[name]
            
            prediction = np.argmax(final_probs)
            confidence = max(final_probs)
            
            outcomes = ['home_win', 'draw', 'away_win']
            model_insights = {
                'individual_predictions': predictions,
                'model_probabilities': probabilities,
                'ensemble_weights': weights,
                'feature_importance': feature_importances,
                'model_agreement_score': self.calculate_model_agreement(predictions),
                'prediction_confidence': self.calculate_prediction_confidence(probabilities, weights)
            }
            
            return outcomes[prediction], confidence, model_insights
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 'unknown', 0.0, {}
            
    def calculate_dynamic_weights(self, match_data: Dict) -> Dict[str, float]:
        """Calculate dynamic weights for ensemble based on recent performance"""
        recent_accuracy = {
            'gradient_boost': 0.82,  # Example values - should be calculated from actual performance
            'random_forest': 0.78,
            'xgboost': 0.85,
            'lightgbm': 0.80,
            'catboost': 0.83
        }
        
        # Normalize weights
        total = sum(recent_accuracy.values())
        return {k: v/total for k, v in recent_accuracy.items()}
        
    def calculate_model_agreement(self, predictions: Dict) -> float:
        """Calculate agreement score between models"""
        prediction_values = list(predictions.values())
        most_common = max(set(prediction_values), key=prediction_values.count)
        agreement = prediction_values.count(most_common) / len(prediction_values)
        return agreement
        
    def calculate_prediction_confidence(self, probabilities: Dict, weights: Dict) -> float:
        """Calculate overall prediction confidence considering model weights"""
        weighted_confidences = []
        for model_name, probs in probabilities.items():
            weighted_confidences.append(max(probs) * weights[model_name])
        return sum(weighted_confidences)