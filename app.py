from flask import Flask, render_template, request
import pandas as pd
import os
import json
from recommender import recommend

app = Flask(__name__)

data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "support_tickets_history.csv"))
dataset = pd.read_csv(data_file_path)
    

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", relevant_profiles=None)

@app.route("/recommend", methods=["POST"])
def recommendation():
    if request.method == "POST":
        # Get the user input (Description)
        user_input = request.form['issue_description']
        recommendations = recommend(dataset, user_input)


    return render_template("index.html", recommendations=recommendations)

if __name__ == "__main__":
     app.run(host="127.0.0.1", port=8080, debug=True)
