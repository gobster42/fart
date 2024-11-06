from discord_webhook import DiscordWebhook, DiscordEmbed
from datetime import datetime
from typing import Dict, List
import logging
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO

from config import DISCORD_WEBHOOK_URL, NOTIFICATION_COLORS

logger = logging.getLogger(__name__)

class EnhancedNotificationManager:
    def __init__(self, webhook_url: str = DISCORD_WEBHOOK_URL):
        self.webhook_url = webhook_url

    def create_prediction_visualization(
        self,
        match_data: Dict,
        model_insights: Dict
    ) -> str:
        """Create visualization of model predictions"""
        # Create probability distribution plot
        fig = go.Figure()
        
        outcomes = ['Home Win', 'Draw', 'Away Win']
        for model, probs in model_insights['model_probabilities'].items():
            fig.add_trace(go.Bar(
                name=model,
                x=outcomes,
                y=probs,
                text=[f'{p:.1%}' for p in probs]
            ))
            
        fig.update_layout(
            title='Model Predictions Comparison',
            barmode='group',
            yaxis_title='Probability',
            height=400
        )
        
        # Save plot to bytes
        img_bytes = BytesIO()
        fig.write_image(img_bytes, format='png')
        img_bytes.seek(0)
        
        return base64.b64encode(img_bytes.read()).decode()

    def create_advanced_embed(
        self,
        match_data: Dict,
        prediction: str,
        confidence: float,
        value_analysis: Dict,
        model_insights: Dict
    ) -> DiscordEmbed:
        """Create a detailed Discord embed with advanced analytics"""
        
        confidence_level = (
            'high_confidence' if confidence > 0.75
            else 'medium_confidence' if confidence > 0.6
            else 'low_confidence'
        )
        
        embed = DiscordEmbed(
            title=f"🎯 Advanced Betting Analysis: {match_data['home_team']} vs {match_data['away_team']}",
            color=NOTIFICATION_COLORS[confidence_level]
        )

        # Match Information
        embed.add_embed_field(
            name="📅 Match Details",
            value=f"Date: {match_data['date']}\nLeague: {match_data['league']}\nVenue: {match_data['venue']}",
            inline=False
        )

        # Advanced Prediction Details
        prediction_details = (
            f"Primary Outcome: {prediction.replace('_', ' ').title()}\n"
            f"Model Confidence: {confidence:.2%}\n"
            f"Ensemble Agreement: {model_insights['model_agreement_score']:.2%}\n"
            f"Market Efficiency: {value_analysis['analysis']['market_efficiency']:.2%}"
        )
        embed.add_embed_field(
            name="🔮 Advanced Prediction",
            value=prediction_details,
            inline=True
        )

        # Value Analysis
        best_value = value_analysis['analysis']['best_value']
        value_details = (
            f"Best Value Bet: {best_value}\n"
            f"True Probability: {value_analysis[best_value]['true_probability']:.2%}\n"
            f"Value Rating: {value_analysis[best_value]['value_rating']:.1f}%\n"
            f"Kelly Stake: {value_analysis[best_value]['optimal_bet']:.2f} units"
        )
        embed.add_embed_field(
            name="💰 Value Analysis",
            value=value_details,
            inline=True
        )

        # Team Performance Metrics
        performance_details = (
            f"Home Attack Strength: {match_data['home_metrics']['attack_strength']:.2f}\n"
            f"Home Defense Stability: {match_data['home_metrics']['defense_stability']:.2f}\n"
            f"Away Attack Strength: {match_data['away_metrics']['attack_strength']:.2f}\n"
            f"Away Defense Stability: {match_data['away_metrics']['defense_stability']:.2f}"
        )
        embed.add_embed_field(
            name="📊 Performance Metrics",
            value=performance_details,
            inline=False
        )

        # Add visualization
        viz_image = self.create_prediction_visualization(match_data, model_insights)
        embed.set_image(url=f"attachment://prediction_viz.png")

        # Risk Assessment
        risk_details = (
            f"Risk Level: {value_analysis['analysis']['risk_assessment']['level']}\n"
            f"Volatility: {value_analysis['analysis']['risk_assessment']['volatility']:.2f}\n"
            f"Confidence Score: {value_analysis['analysis']['risk_assessment']['confidence']:.2f}"
        )
        embed.add_embed_field(
            name="⚠️ Risk Assessment",
            value=risk_details,
            inline=True
        )

        embed.set_footer(text="⚠️ Bet responsibly. This is AI-generated advice.")
        embed.set_timestamp(datetime.now().timestamp())

        return embed

    async def send_prediction(self, embed: DiscordEmbed, viz_image: str = None) -> bool:
        """Send prediction to Discord channel"""
        try:
            webhook = DiscordWebhook(url=self.webhook_url)
            webhook.add_embed(embed)
            
            if viz_image:
                webhook.add_file(
                    file=base64.b64decode(viz_image),
                    filename="prediction_viz.png"
                )
                
            response = webhook.execute()
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False