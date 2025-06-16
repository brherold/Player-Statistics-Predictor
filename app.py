from flask import Flask, render_template, request, redirect, url_for
from scripts.flaskGetPredictedStats import *

import time
import requests
import threading


app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        try:
            player_url = request.form["url"]
            position = request.form.get("position")
            predicted_stats = givePlayerStats(player_url,position)
            return render_template("predictorPage.html", player_url = player_url, player_name = predicted_stats[0], position = position, stats = predicted_stats[1] )
        except:
            return render_template("error.html")

    return render_template("home.html")

    #return 

#Self-Pings every 12 minutes to keep web-app open in Render
def keep_alive():
    while True:
        time.sleep(720)  
        try:
            requests.get("https://player-statistics-predictor.onrender.com")  
            print("Self-ping successful")
        except requests.exceptions.RequestException as e:
            print(f"Failed to ping: {e}")

# Starts background task when app starts
thread = threading.Thread(target=keep_alive, daemon=True)
thread.start()




if __name__ == '__main__':
    #app.run()
    app.run(port=5002,debug=True)