# Northern Ireland Sports Betting Analyzer

An advanced sports betting analysis system for Northern Ireland sports, featuring machine learning predictions, value bet detection, and Discord notifications.

## Features

- Real-time match data fetching
- Advanced prediction model using Random Forest algorithm
- Value bet detection using Kelly Criterion
- Detailed Discord notifications with match analysis
- Automated monitoring of Northern Ireland Premiership matches
- Historical data analysis and form tracking

## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on `.env.example` and add your API keys:
   - Get a Football API key from [API-Football](https://www.api-football.com/)
   - Create a Discord webhook URL for notifications

4. Run the analyzer:
   ```bash
   python src/main.py
   ```

## Discord Notifications

The system sends detailed match predictions including:
- Match details and timing
- Prediction confidence levels
- Value bet analysis
- Current best odds
- Team form analysis
- Head-to-head statistics

## Responsible Betting Notice

This tool is for informational purposes only. Please bet responsibly and be aware of the risks involved in sports betting.

## Maintenance

- Regularly update the model with new training data
- Monitor API usage limits
- Check Discord webhook functionality
- Review prediction accuracy and adjust thresholds as needed