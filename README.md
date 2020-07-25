# fpl-analysis

## User guide

__Parameters__:

`previous_gw`: Gameweek prior to the one you want to make a selection for

`prediction_season_order`: Season order number (2019/20 season is 4)

`live_run`: Boolean. Set to True to use FPL API to make predictions using latest data. Set to False for retrospective predictions

`double_gw_teams` (_optional_): List of teams with a double gameweek in the next gameweek. Next gameweek predictions for these teams are multiplied by 2

Current teams:
'Manchester City', 'Liverpool', 'Arsenal', 'Wolverhampton Wanderers', 'Everton', 'Aston Villa', 'Leicester City', 
'Manchester United', 'Southampton', 'Tottenham Hotspur', 'Chelsea', 'Burnley', 'West Ham United', 'Crystal Palace', 
'Sheffield United', 'Watford', 'Norwich City', 'Bournemouth', 'Brighton & Hove Albion', 'Newcastle United'

__Run locally using gunicorn__:
```bash
export $(xargs <.env)

gunicorn app.fpl_app:app --bind 0.0.0.0:5000 --timeout 2000

curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data '{"previous_gw":"29","prediction_season_order":"4","live_run":"True"}'
```

__Run locally using Docker__:
```bash
docker-compose up

curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data '{"previous_gw":"29","prediction_season_order":"4","live_run":"True"}'
```
