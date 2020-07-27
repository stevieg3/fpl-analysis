import json

from flask import Flask, request, Response

from src.interface import fpl_scorer

app = Flask(__name__)

CURRENT_SEASON_TEAMS = [
    'Manchester City', 'Liverpool', 'Arsenal',
    'Wolverhampton Wanderers', 'Everton', 'Aston Villa',
    'Leicester City', 'Manchester United', 'Southampton',
    'Tottenham Hotspur', 'Chelsea', 'Burnley', 'West Ham United',
    'Crystal Palace', 'Sheffield United', 'Watford', 'Norwich City',
    'Bournemouth', 'Brighton & Hove Albion', 'Newcastle United'
]

NON_OPTIONAL_PAYLOAD_PARAMS = ['previous_gw', 'prediction_season_order', 'live_run']


@app.route('/api', methods=['GET'])
def api():
    content = request.get_json()
    print(content)

    # Non-optional parameters:
    for param in NON_OPTIONAL_PAYLOAD_PARAMS:
        assert param in content.keys(), f'{param} not provided in payload'

    previous_gw = content['previous_gw']
    prediction_season_order = content['prediction_season_order']
    live_run = content['live_run']
    assert isinstance(live_run, bool), 'live_run not a Boolean'

    # Optional parameters:
    try:
        double_gw_teams = content['double_gw_teams']
        assert all(team in CURRENT_SEASON_TEAMS for team in double_gw_teams), "Invalid team in double_gw_teams"
    except KeyError:
        double_gw_teams = []

    try:
        previous_gw_was_double_gw = content['previous_gw_was_double_gw']
    except KeyError:
        previous_gw_was_double_gw = False

    fpl_scorer(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order,
        live_run=live_run,
        double_gw_teams=double_gw_teams,
        previous_gw_was_double_gw=previous_gw_was_double_gw
    )

    return Response(status=200)


if __name__ == '__main__':
    # need debug and threaded parameters to prevent TensorFlow error
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
