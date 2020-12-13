# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:26:50 2020

@author: KSK
"""
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template,url_for
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.naive_bayes import MultinomialNB

#load the model fron the desk
clf=pickle.load(open('nlp_model.pkl','rb'))
cv=pickle.load(open('transform.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.html',prediction=my_prediction)

if __name__=='__main__':
    app.run(debug=True)