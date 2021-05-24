from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
from model import pdt_recommendation

app = Flask(__name__)

model_load = pdt_recommendation()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        username = request.form.get("user_name")
        top_5_pdt = model_load.predict(str(username)) 

        return render_template('index.html', top_5_products= top_5_pdt )
    else :
        return render_template('index.html')


if __name__ == '__main__':
    app.run()