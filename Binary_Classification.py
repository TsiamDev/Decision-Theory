# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:16:30 2022

@author: Konstantinos Tsiamitros
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
import csv

# creates the model
def create_baseline():
    # initializes an empty sequential model
    model = Sequential()
    # adds an input layer with 13 inputs: (without the "target" column)
    model.add(Input(13))
    # adds a dense layer (fully connected) with 13 inputs, activation function of ReLU and 250 outputs
    # experiemented a bit with number of outputs in the hidden layer - 250 seems to be the sweetspot
    model.add(Dense(250, activation='relu'))
    # adds a dense layer with one output and sigmoid activation function 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

# Parsing of the dataset file - pretty much self explanatory
headers = []
labels = []

with open('Dataset_1.csv') as file:
    data = csv.reader(file, delimiter=',')
    
    data = list(data)
    #string correction - remove weird character causing trouble :)
    data[0][0] = "age"
    headers = data[0][:]
    del data[0]
    
    temp = []
    for row in data:
        temp_row = []
        for j in range(0, len(row)):
            if j == (len(row) - 1):
                # store the "target" column - should be int
                labels.append(int(row[j]))
            else:
                # store the rest of the columns
                temp_row.append(float(row[j]))
        temp.append(temp_row)
    #convert parsed data to numpy array
    data = np.array(temp)
        
    
#"""
# evaluate model with dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=8000, verbose=1)
# 10-fold validator: splits the dataset into 10 train-test sets
kfold = StratifiedKFold(n_splits=10, shuffle=True)
# added early stopping to avoid overfitting - restores best weights
es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=30, restore_best_weights=True)
# get the results
results = cross_val_score(estimator, data, labels, cv=kfold, fit_params={'callbacks':es})
# display results
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#"""
 
   