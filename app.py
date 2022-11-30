from flask import Flask,request, url_for, redirect, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

filename = 'modelo_entrenado.pkl'
model = pickle.load(open(filename, 'rb'))
cols = ['quilates','profundidadt','table','longitud','anchura','profundidad','claridad','color','corte']

@app.route('/')
def home():
    return render_template("front_analitica.html")

@app.route('/predict',methods=['POST'])
def predict():
    data = {
        "Claridad": request.form.get("claridad"),
        "color": request.form.get("color"),
        "corte": request.form.get("corte")
    }
    int_features = [x for x in request.form.values()]

    print('lista: ',int_features)
    final = np.array(int_features).reshape(1, -1)
    prediction = model.predict(final)
    prediction = int(prediction[0])
    return render_template('front_analitica.html',pred='El precio del diamante es ${}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)