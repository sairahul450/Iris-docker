# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)

pickle_in=open('model.pkl','rb')

model= pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome all"

@app.route('/predict')
def predict_note_authentication():
    sepal_len=request.args.get('sepal_len') 
    sepal_wid=request.args.get('sepal_wid')
    petal_len=request.args.get('petal_len')
    petal_wid=request.args.get('petal_wid')
    prediction=model.predict([sepal_len,sepal_wid,petal_len,petal_wid])
    return "Predicted value :" +str(prediction)


if __name__=='__main__':
    app.run()
   # http://127.0.0.1:5000/predict?sepal_len=5&sepal_wid=3&petal_len=1&petal_wid=0