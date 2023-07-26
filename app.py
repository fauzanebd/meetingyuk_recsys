import json
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request
from threading import Thread
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from werkzeug.exceptions import BadRequest

from collaborative_filtering import run_sgd_background
from db_connections import meetingyuk_mongo_collection, get_new_ratings_data
from for_you_recommendation import for_you_recommendation
from nearest_recommendation import nearest_recommendation, find_k
import os


load_dotenv(".env")
app = Flask(__name__)

# sched = BackgroundScheduler(daemon=True)
# sched.add


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/near_recs/<string:user_latitude>,<string:user_longitude>', methods=['GET'])
def _nearest_recommendation(user_latitude, user_longitude):
    max_returns = request.args.get('max_returns', default=25, type=int)
    max_radius = request.args.get('max_radius', default=1000, type=float)
    if user_latitude is None or user_longitude is None:
        abort(400)
    else:
        try:
            return (
                jsonify({
                    "success": True,
                    "recommendations": json.loads(
                        nearest_recommendation(user_latitude, user_longitude, max_returns, from_api=True, max_radius=max_radius)
                    )
                })
            )
        except Exception as e:
            return (
                jsonify({
                    "error": e
                }), 500
            )

@app.route('/recommendation/<string:user_id>', methods=['GET'])
def _recommendation(user_id):

    max_returns = request.args.get('max_returns', default=25, type=int)
    include_rated = request.args.get('include_rated', default=False, type=bool)
    reqbody = None
    try:
        reqbody = request.get_json()
    except Exception as e:
        pass
    latitude = None
    longitude = None
    max_radius = None
    if reqbody:
        if 'location' in reqbody:
            latitude = reqbody['location']['latitude']
            longitude = reqbody['location']['longitude']
            max_radius = reqbody['location'].get('max_radius', 10000)
            max_radius = float(max_radius)

    if user_id is None:
        abort(400)
    else:
        try:
            return (
                jsonify({
                    "success": True,
                    "recommendations": json.loads(
                        for_you_recommendation(user_id, max_returns, latitude, longitude, include_rated, max_radius)
                    )
                })
            )
        except Exception as e:
            return (
                jsonify({
                    "error": e
                }), 500
            )


@app.route('/kmeans/train_model', methods=['GET'])
def _train_kmeans():
    # Call this api every time new place data is inserted
    # to retrain the kmeans model
    thread = Thread(target=find_k)
    thread.start()
    return jsonify({"message": "K-means training started in background."})


@app.route('/sgd/train_new_data', methods=['GET'])
def _train_new_data_sgd():
    # Call this api every time new ratings data is inserted
    # to retrain the sgd model
    try:
        reqdata = request.get_json()
    except BadRequest:
        abort(400)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 400

    if reqdata is None:
        abort(400)

    if 'new_data_ids' not in reqdata:
        abort(400)

    new_data_ids = reqdata['new_data_ids']
    new_data = get_new_ratings_data(new_data_ids)

    thread = Thread(target=run_sgd_background, args=(new_data,))
    thread.start()

    return jsonify({"message": "SGD training started in background."})

def start_app():
    print("Starting app...")
    load_dotenv(".env")
    # run_sgd_background()
    find_k()

start_app()
if __name__ == '__main__':
    app.run()