from flask import Flask, render_template, request, redirect, url_for
from scripts.flaskGetPredictedStats import *

app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        player_url = request.form["url"]
        position = request.form.get("position")
        predicted_stats = givePlayerStats(player_url,position)
        return render_template("predictorPage.html", player_url = player_url, player_name = predicted_stats[0], position = position, stats = predicted_stats[1] )

    return render_template("home.html")

    #return 





if __name__ == '__main__':
    app.run(port=5002, debug=True)
