# For Custom Stat Predictor 
from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.CustomGetPredictedStats import *
import json

app = Flask(__name__)
CORS(app)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    print("Received JSON:", data)

    position = data["Position"]
    del data["Position"]

    predicted_stats = givePlayerStats(data, position)
    #print(predicted_stats)
    return jsonify({"status": "success", "received": predicted_stats})

if __name__ == '__main__':
    app.run(debug=True)




    '''
    # Dummy Data
    data = {'Position':'PF','IS': 14, 'IQ': 11, 'OS': 13, 'Pass': 3, 'Rng': 14, 'Hnd': 14, 'Fin': 11, 'Drv': 11, 'Reb': 15, 
    'Str': 13, 'IDef': 16, 'Spd': 9, 'PDef': 9, 'Sta': 12, 'height': 82.0, 'wingspan': 90.0, 'weight': 215.0, 
    'vertical': 33.5}
    '''
