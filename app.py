import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#Starting point
app = Flask(__name__)
#load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data = request.json['data'] #the request that come to the api will have the variable called data having the data in json format
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    #new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    new_data = np.array(list(data.values())).reshape(1,-1)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug= True)


