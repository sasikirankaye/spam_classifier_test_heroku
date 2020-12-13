# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:00:13 2020

@author: KSK
"""
import numpy as np
import pandas as pd
import nltk
# loading the dataset
df=pd.read_csv('spam.csv',encoding="latin-1")
df.isnull().sum()
#dropping the unnamed2,3,4 features
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

#seperating the X and y variables
df['label']=df['class'].map({'ham':0,'spam':1})
X=df['message']
y=df['label']
## calling the countvectorizer
from sklearn.feature_extraction.text import  CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(X)

import pickle
pickle.dump(cv,open('transform.pkl','wb'))

#calling the train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

# calling the naive bayes algorithm
from sklearn.naive_bayes import  MultinomialNB
clf=MultinomialNB()
clf.fit(X,y)
# creating the pickle file

pickle.dump(clf,open('nlp_model.pkl','wb'))