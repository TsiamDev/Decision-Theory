# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:16:30 2022

@author: Konstantinos Tsiamitros
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
import csv

def create_baseline():
    model = Sequential()
    model.add(Dense(13, input_dim=13, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

headers = []
labels = []

#with open('Dataset_small.csv') as file:
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
                labels.append(int(row[j]))
            else:
                temp_row.append(float(row[j]))
        temp.append(temp_row)
    data = np.array(temp)
        

# evaluate model with dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=8000, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=5, restore_best_weights=True)
results = cross_val_score(estimator, data, labels, cv=kfold, fit_params={'callbacks':es})
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
 
   