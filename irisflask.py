#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:20:14 2022

@author: rahulpandiri
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in=open('model.pkl','rb')

model= pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "welcome all"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """ Let's predecit the iris dataset
    This is using docstrings for specifications.
    
    ---
    parameters:
      - name: sepal_len
        in: query
        type: number
        required: true
      - name: sepal_wid
        in: query
        type: number
        required: true
      - name: petal_len
        in: query
        type: number
        required: true
      - name: petal_wid
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """


    sepal_len=request.args.get('sepal_len') 
    sepal_wid=request.args.get('sepal_wid')
    petal_len=request.args.get('petal_len')
    petal_wid=request.args.get('petal_wid')
    prediction=model.predict([[sepal_len,sepal_wid,petal_len,petal_wid]])
    return "Predicted value :" +str(prediction)


if __name__=='__main__':
    app.run()
   # http://127.0.0.1:5000/predict?sepal_len=5&sepal_wid=3&petal_len=1&petal_wid=0