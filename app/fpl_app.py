from flask import Flask, request, Response

from src.interface import fpl_scorer

app = Flask(__name__)


@app.route('/api', methods=['GET'])
def api():
    content = request.get_json()

    previous_gw = int(content['previous_gw'])
    prediction_season_order = int(content['prediction_season_order'])
    live_run_text = content['live_run']

    if live_run_text == 'False':
        live_run = False
    else:
        live_run = True

    fpl_scorer(
        previous_gw=previous_gw,
        prediction_season_order=prediction_season_order,
        live_run=live_run
    )

    return Response(status=201)


if __name__ == '__main__':
    # need debug and threaded parameters to prevent TensorFlow error
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

"""
curl -X GET "http://0.0.0.0:5000/api" -H "Content-Type: application/json" --data '{"previous_gw":"3","prediction_season_order":"2","live_run":"False"}'

gunicorn app.fpl_app:app --bind 0.0.0.0:5000

docker build -t stevengeorge3/fpl-analysis -f app/Dockerfile .

docker run -p 5000:5000 stevengeorge3/fpl-analysis
"""
