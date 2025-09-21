from flask import Flask, render_template, request, redirect, url_for

from scripts.flaskGetPredictedStats import *

import time
import requests
import threading


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            position = request.form.get("position")
            #predicted_stats = None
            #player_url = None

            file = request.files.get("htmlFile")
            url_input = request.form.get("url", "").strip()

        

            # Case 1: URL provided
            if url_input:
                
                player_name, predicted_stats, playerID = givePlayerStats(player_url, position)
                player_url = f"https://onlinecollegebasketball.org/player/{playerID}"

            # Case 2: file uploaded takes priority if not empty
            elif file and file.filename:
                
                file_content = file.read().decode("utf-8", errors="ignore")
                #print(file_content)
                #givePlayerStats("","","")
                
                try:
                    player_name, predicted_stats, playerID = givePlayerStats(file_content, position, from_file=True)
                    
                    #print(player_url)
                    player_url = f"https://onlinecollegebasketball.org/player/{playerID}"

                except Exception as e:
                    print(e)
                    raise    
            

            
            
            else:
                return render_template("error.html", error="Please provide a URL or upload a file")

            #print(predicted_stats)

            return render_template(
                "predictorPage.html",
                player_url=player_url,
                player_name=player_name,
                position=position,
                stats=predicted_stats
            )

        except Exception as e:
            return render_template("error.html", error=str(e))

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
    app.run()
    #app.run(port=5002,debug=True)