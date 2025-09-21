# For Custom Stat Predictor 
from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.CustomGetPredictedStats import *
from scripts.pygetPlayerSkills import *
from scripts.WeightInputRecruitMeasureable import *
from RecruitSkillPredictorCol.scripts.getPred_4 import *
import json

app = Flask(__name__)
CORS(app)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    print("Received JSON:", data)

    position = data["Position"]
    del data["Position"]

    print(data, "XXXXX")

    predicted_stats = givePlayerStats(data, position)
    #print(predicted_stats)
    return jsonify({"status": "success", "received": predicted_stats})


@app.route('/player-url-submit', methods=["POST"])
def playerUrlSubmit():

    data = request.get_json()
    print("Received JSON_URL", data)

    player_url = data.get("url")

    #Check if last String in player_url split is a number (playerCode) or not (could be development page for ex: /D)
    last_string = player_url.split("/")[-1]
    player_code = last_string if last_string.isnumeric() else player_url.split("/")[-2]


    wanted_year = data.get("player_year")
    if wanted_year == "Current":

        player_skills = get_player_info(player_url)
    else:
        predicted_measureables = getPredictedMeasureables(player_code)
        wanted_year = wanted_year[-1]
        predicted_skills = getPredictedSkills(player_code,wanted_year)

        player_skills = {**predicted_measureables, **predicted_skills}



    print(player_skills)

    return jsonify({"message": "Player Skills received successfully", "player_skills": player_skills})

if __name__ == '__main__':
    app.run(debug=True,port=2000)




    '''
    # Dummy Data
    data = {'Position':'PF','IS': 14, 'IQ': 11, 'OS': 13, 'Pass': 3, 'Rng': 14, 'Hnd': 14, 'Fin': 11, 'Drv': 11, 'Reb': 15, 
    'Str': 13, 'IDef': 16, 'Spd': 9, 'PDef': 9, 'Sta': 12, 'height': 82.0, 'wingspan': 90.0, 'weight': 215.0, 
    'vertical': 33.5}
    '''
