import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

# League IDs for Northern Ireland competitions
NI_PREMIERSHIP_ID = 128
NI_CHAMPIONSHIP_ID = 129

# Model configuration
PREDICTION_THRESHOLD = 0.65
MIN_ODDS_VALUE = 1.5
MAX_ODDS_VALUE = 7.0

# Discord notification settings
NOTIFICATION_COLORS = {
    'high_confidence': 0x00ff00,  # Green
    'medium_confidence': 0xffff00,  # Yellow
    'low_confidence': 0xff0000,    # Red
}