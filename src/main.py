import asyncio
import logging
from datetime import datetime
import json
from typing import Dict, List

from data_fetcher import DataFetcher
from prediction_model import PredictionModel
from notifications import NotificationManager
from config import NI_PREMIERSHIP_ID, PREDICTION_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BettingAnalyzer:
    def __init__(self):
        self.model = PredictionModel()
        self.notification_manager = NotificationManager()

    async def analyze_matches(self):
        """Main analysis loop"""
        async with DataFetcher() as fetcher:
            while True:
                try:
                    # Fetch upcoming matches
                    matches = await fetcher.fetch_matches(NI_PREMIERSHIP_ID)
                    
                    for match in matches:
                        # Fetch odds for the match
                        odds = await fetcher.fetch_odds(match['fixture']['id'])
                        
                        # Prepare match data for prediction
                        match_data = self.prepare_match_data(match, odds)
                        
                        # Get prediction and confidence
                        prediction, confidence = self.model.predict_match(match_data)
                        
                        if confidence >= PREDICTION_THRESHOLD:
                            # Evaluate if it's a value bet
                            is_value, value = self.model.evaluate_value_bet(
                                prediction, confidence, odds['bookmakers'][0]['bets'][0]['values']
                            )
                            
                            if is_value:
                                # Create and send notification
                                embed = self.notification_manager.create_prediction_embed(
                                    match_data, prediction, confidence, value, odds
                                )
                                await self.notification_manager.send_prediction(embed)
                                
                                logger.info(f"Sent prediction for {match_data['home_team']} vs {match_data['away_team']}")
                
                except Exception as e:
                    logger.error(f"Error in analysis loop: {e}")
                
                # Wait for 1 hour before next analysis
                await asyncio.sleep(3600)

    @staticmethod
    def prepare_match_data(match: Dict, odds: Dict) -> Dict:
        """Prepare match data for prediction"""
        return {
            'home_team': match['teams']['home']['name'],
            'away_team': match['teams']['away']['name'],
            'date': datetime.fromtimestamp(match['fixture']['timestamp']).strftime('%Y-%m-%d %H:%M'),
            'league': match['league']['name'],
            'venue': match['fixture']['venue']['name'],
            'home_team_form': match['teams']['home']['form'],
            'away_team_form': match['teams']['away']['form'],
            'odds': odds
        }

if __name__ == "__main__":
    analyzer = BettingAnalyzer()
    asyncio.run(analyzer.analyze_matches())