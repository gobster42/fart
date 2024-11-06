import numpy as np
from typing import Dict, Tuple, List
import logging
from scipy.stats import poisson

logger = logging.getLogger(__name__)

class ValueBetAnalyzer:
    def __init__(self):
        self.min_value_threshold = 0.1
        self.max_odds_threshold = 10.0
        
    def calculate_true_probabilities(
        self,
        match_data: Dict,
        model_confidence: float,
        model_insights: Dict
    ) -> Dict[str, float]:
        """Calculate true probabilities using multiple factors"""
        
        # Base probabilities from model
        base_probs = model_insights['ensemble_probabilities']
        
        # Adjust for historical accuracy
        historical_accuracy = self.get_historical_accuracy(match_data)
        adjusted_probs = base_probs * historical_accuracy
        
        # Adjust for market efficiency
        market_probs = self.calculate_market_probabilities(match_data['odds'])
        efficiency_weight = 0.3  # Weight given to market probabilities
        
        final_probs = {
            outcome: (1 - efficiency_weight) * adj_prob + efficiency_weight * market_prob
            for outcome, (adj_prob, market_prob) in 
            zip(base_probs.keys(), zip(adjusted_probs, market_probs.values()))
        }
        
        return final_probs
    
    def kelly_criterion_full(
        self,
        true_prob: float,
        odds: float,
        bankroll: float,
        kelly_fraction: float = 0.5
    ) -> Tuple[float, Dict]:
        """Calculate optimal bet size using fractional Kelly Criterion"""
        if odds <= 1 or true_prob <= 0:
            return 0.0, {'reason': 'Invalid odds or probability'}
            
        q = 1 - true_prob
        p = true_prob
        b = odds - 1
        
        if p * b <= q:  # No edge
            return 0.0, {'reason': 'No edge found'}
            
        kelly_bet = ((p * b - q) / b) * kelly_fraction
        optimal_bet = kelly_bet * bankroll
        
        insights = {
            'edge': p * b - q,
            'kelly_percentage': kelly_bet * 100,
            'recommended_bet': optimal_bet,
            'expected_value': (p * odds - 1) * 100
        }
        
        return optimal_bet, insights
    
    def analyze_value_bet(
        self,
        prediction: str,
        confidence: float,
        odds: Dict,
        match_data: Dict,
        model_insights: Dict,
        bankroll: float = 1000
    ) -> Dict:
        """Comprehensive value bet analysis"""
        try:
            # Calculate true probabilities
            true_probs = self.calculate_true_probabilities(match_data, confidence, model_insights)
            
            results = {}
            for outcome, prob in true_probs.items():
                if outcome in odds:
                    # Get optimal bet size and insights
                    bet_size, insights = self.kelly_criterion_full(prob, odds[outcome], bankroll)
                    
                    results[outcome] = {
                        'true_probability': prob,
                        'offered_odds': odds[outcome],
                        'optimal_bet': bet_size,
                        'kelly_insights': insights,
                        'value_rating': (prob * odds[outcome] - 1) * 100,
                        'confidence_score': model_insights['model_probabilities'][outcome]
                    }
            
            # Add overall analysis
            results['analysis'] = {
                'best_value': max(results.items(), key=lambda x: x[1]['value_rating'])[0],
                'market_efficiency': self.calculate_market_efficiency(odds),
                'model_consensus': model_insights['model_agreement_score'],
                'risk_assessment': self.assess_risk(results)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Value bet analysis error: {e}")
            return {}