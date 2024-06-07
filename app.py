import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model=pickle.load(open('mlmodel.pkl','rb'))

@app.route('/') #default
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data'] #whenever predict_api is used, i/p is taken in json format and captured in data key
    #capture it using request.json
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    input_data = np.array(list(data.values())).reshape(1, -1)
    #new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(input_data)
    print(output)
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)