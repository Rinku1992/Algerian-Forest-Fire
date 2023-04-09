import pickle
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
scaler_model = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    # data = [float(x) for x in request.json['data'].values()]
    print(data)
    new_data = [list(data.values())]
    # new_data = [list(data)]
    # final_features = [np.array(data)]
    print(new_data)
    scaled_data = scaler_model.transform(new_data)
    output = float(model.predict(scaled_data)[0])

    output1 = str(np.where(output == 0, 'not fire', 'fire'))

    return jsonify(output1)

@app.route('/predict', methods = ['POST'])

def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [list(data)]
    scaled_data = scaler_model.transform(final_features)
    # final_features = final_features.reshape(1,-1)
    print(final_features)
   
    output = float(model.predict(scaled_data)[0])
    output1 = str(np.where(output == 0, 'not fire', 'fire'))

    return render_template('home.html', prediction_text = "Algerian Forest will have {}".format(output1))

@app.route('/eda', methods = ['GET'])
def eda():

    return render_template('Algerian_fire.html')

if __name__ == "__main__":
    app.run(host = "0.0.0.0",port = 5000)