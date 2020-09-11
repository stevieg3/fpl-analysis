# fpl-analysis

Using machine learning to optimise Fantasy Premier League team selection.

In this repository we model points scored by Premier League players in the popular Fantasy Premier League game (https://fantasy.premierleague.com/). We then use this to make predictions for upcoming fixtures.

The main model is `DeepFantasyFootball`, a LSTM which uses Fantasy Football Scout and Odds Portal data to make player points predictions for the next 5 gameweeks. The main notebooks used to develop this model can be found in `notebooks/DeepFantasyFootball`.

We use SHAP values to explain model predictions (see `notebooks/SHAP values 2020-21.ipynb`)

Fantasy Football Scout does not have 100% coverage over Premier League players so we also use a fallback LSTM which uses just the features from the official FPL API. This also incorporates 'market' features such as cost and ownership. In practice this is only required for players who play very few minutes during the course of a season.

## API guide

Note: Requires relevant S3 buckets and files (intended for personal use)

__Parameters__:

`previous_gw`: Gameweek prior to the one you want to make a selection for

`prediction_season_order`: Season order number (2020/21 season is 5)

`live_run`: Boolean. Set to True to use FPL API to make predictions using latest data. Set to False for retrospective predictions

`double_gw_teams` (_optional_): List of teams with a double gameweek in the next gameweek. Next gameweek predictions for these teams are multiplied by 2. Only needs to be specified for FPL-based model. DeepFantasyFootball incorporates this as a feature.

Current teams:
'Manchester City', 'Liverpool', 'Arsenal', 'Wolverhampton Wanderers', 'Everton', 'Aston Villa', 'Leicester City', 
'Manchester United', 'Southampton', 'Tottenham Hotspur', 'Chelsea', 'Burnley', 'West Ham United', 'Crystal Palace', 
'Sheffield United', 'West Bromwich Albion', 'Leeds', 'Fulham', 'Brighton & Hove Albion', 'Newcastle United'

__Run locally using gunicorn__:
```bash
export $(xargs <.env)

gunicorn app.fpl_app:app --bind 0.0.0.0:5000 --timeout 2000

curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data '{"previous_gw": 38, "prediction_season_order": 3, "live_run": false, "double_gw_teams": ["Arsenal"]}'
```

__Run locally using Docker__:
```bash
docker-compose up

curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data '{"previous_gw": 38, "prediction_season_order": 3, "live_run": false, "double_gw_teams": ["Arsenal"]}'
```

## Data sources

- Fantasy Football Scout for player-level statistics (https://www.fantasyfootballscout.co.uk/)
- OddsPortal for historical and live match outcome odds (https://www.oddsportal.com/soccer/england/premier-league/)
- Official historical FPL data from 2016-17 to 2018-19 (https://github.com/vaastav/Fantasy-Premier-League)