import numpy as np
from typing import Dict, List
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    def __init__(self):
        self.risk_thresholds = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }

    def calculate_volatility_score(self, historical_data: List[Dict]) -> float:
        """Calculate volatility score based on historical performance"""
        returns = np.array([match['return'] for match in historical_data])
        return np.std(returns)

    def assess_market_liquidity(self, odds_history: List[Dict]) -> float:
        """Assess market liquidity based on odds movement"""
        odds_changes = np.diff([odds['value'] for odds in odds_history])
        return np.mean(np.abs(odds_changes))

    def calculate_risk_score(
        self,
        prediction_confidence: float,
        model_agreement: float,
        market_efficiency: float,
        volatility: float
    ) -> Dict:
        """Calculate comprehensive risk score"""
        base_risk = 1 - prediction_confidence
        
        # Weight factors
        weights = {
            'model_agreement': 0.3,
            'market_efficiency': 0.3,
            'volatility': 0.4
        }
        
        risk_components = {
            'model_uncertainty': 1 - model_agreement,
            'market_risk': 1 - market_efficiency,
            'volatility_risk': min(volatility, 1.0)
        }
        
        weighted_risk = sum(
            risk * weights[factor]
            for factor, risk in risk_components.items()
        )
        
        final_risk = (base_risk + weighted_risk) / 2
        
        risk_level = (
            'low' if final_risk < self.risk_thresholds['low']
            else 'medium' if final_risk < self.risk_thresholds['medium']
            else 'high'
        )
        
        return {
            'score': final_risk,
            'level': risk_level,
            'components': risk_components,
            'confidence': 1 - final_risk
        }