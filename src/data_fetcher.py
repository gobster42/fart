import aiohttp
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, List

from config import FOOTBALL_API_KEY, ODDS_API_KEY, NI_PREMIERSHIP_ID

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_matches(self, league_id: int) -> List[Dict]:
        """Fetch upcoming matches from the Football API"""
        url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures"
        headers = {
            "X-RapidAPI-Key": FOOTBALL_API_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        params = {
            "league": league_id,
            "season": datetime.now().year,
            "next": 10
        }

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                return data.get('response', [])
        except Exception as e:
            logger.error(f"Error fetching matches: {e}")
            return []

    async def fetch_odds(self, fixture_id: int) -> Dict:
        """Fetch odds from multiple bookmakers"""
        url = f"https://api-football-v1.p.rapidapi.com/v3/odds"
        headers = {
            "X-RapidAPI-Key": FOOTBALL_API_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
        params = {"fixture": fixture_id}

        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                return data.get('response', [{}])[0]
        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            return {}