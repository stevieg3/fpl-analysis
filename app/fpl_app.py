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


@app.route('/api', methods=['GET'])
def api():
    content = request.get_json()

    previous_gw = int(content['previous_gw'])
    prediction_season_order = int(content['prediction_season_order'])
    live_run_text = content['live_run']
    try:
        double_gw_teams = list(content['double_gw_teams'])
        assert all(team in CURRENT_SEASON_TEAMS for team in double_gw_teams), "Invalid team in double_gw_teams"
    except KeyError:
        double_gw_teams = []

    if live_run_text == 'False':
        live_run = False
    elif live_run_text == 'True':
        live_run = True
    else:
        raise ValueError("Invalid value passed for live_run. Must be 'True' or 'False'")

    fpl_scorer(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order,
        live_run=live_run,
        double_gw_teams=double_gw_teams
    )

    return Response(status=200)


if __name__ == '__main__':
    # need debug and threaded parameters to prevent TensorFlow error
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
