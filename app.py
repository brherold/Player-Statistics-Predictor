from flask import Flask, render_template, request, redirect, url_for

from scripts.flaskGetPredictedStats import *
from scripts.TeamEPMContent.getTeamPlayerStatsFlask import *
import time
import requests
import threading

'''
modes: predict (player stat predictor) | view (team EPM viewer)
'''

app = Flask(__name__)

#Color wheel for Player Stat EPMS
def epm_to_rgb(value, min_val=-10, max_val=10):
    """
    (OPM and DPM have min_val = -5, max_val = 5)
    (EPM has min_val = -10, max_val = 10)
    """

    v = max(min_val, min(max_val, value))
    
    gray = (80, 80, 85)       
    red = (255, 80, 80)       
    green = (0, 220, 120)    
    
    if v < 0:
        t = abs(v) / abs(min_val)
        r = int(gray[0] + (red[0] - gray[0]) * t)
        g = int(gray[1] + (red[1] - gray[1]) * t)
        b = int(gray[2] + (red[2] - gray[2]) * t)
    elif v > 0:
        t = v / max_val
        r = int(gray[0] + (green[0] - gray[0]) * t)
        g = int(gray[1] + (green[1] - gray[1]) * t)
        b = int(gray[2] + (green[2] - gray[2]) * t)
    else:
        r, g, b = gray
    
    return f'rgb({r},{g},{b})'



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get the selected mode
            mode = request.form.get("mode")  # 'predict' or 'view'
            position = request.form.get("position")  # only relevant for player

            file = request.files.get("htmlFile")
            url_input = request.form.get("url", "").strip()

            if url_input:
                # Mode check: only allow URL input if player predictor
                if mode == "predict":
                    player_name, predicted_stats, playerID = givePlayerStats(url_input, position)
                    player_url = f"https://onlinecollegebasketball.org/player/{playerID}"
                    return render_template(
                        "predictorPage.html",
                        player_url=player_url,
                        player_name=player_name,
                        position=position,
                        stats=predicted_stats
                    )
                else:
                    return render_template("error.html", error="Team EPM mode only supports file upload.")

            elif file and file.filename:
                file_content = file.read().decode("utf-8", errors="ignore")

                if mode == "predict":
                    # Process player HTML
                    player_name, predicted_stats, playerID = givePlayerStats(file_content, position, from_file=True)
                    player_url = f"https://onlinecollegebasketball.org/player/{playerID}"
                    return render_template(
                        "predictorPage.html",
                        player_url=player_url,
                        player_name=player_name,
                        position=position,
                        stats=predicted_stats
                    )
                else:
                    # Process team HTML
                    header_text, team_id, season, team_stats, player_epms = get_team_player_stats(file_content)

                    
                    return render_template(
                        "teamStatPage.html",
                        header_text = header_text,
                        team_id = team_id,
                        season = season, 
                        team_stats=team_stats,
                        player_epms=player_epms,
                        epm_to_rgb=epm_to_rgb
                    )
            else:
                return render_template("error.html", error="Please provide a URL or upload a file")

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