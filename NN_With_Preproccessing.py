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

from matplotlib import pyplot as plt
import numpy as np
import csv
import math

# creates the model
def create_baseline():
    # initializes an empty sequential model
    model = Sequential()
    # adds an input layer with 13 inputs: (without the "target" column)
    model.add(Input(13))
    # adds a dense layer (fully connected) with 13 inputs, activation function of ReLU and 250 outputs
    # experiemented a bit with number of outputs in the hidden layer - 250 seems to be the sweetspot
    #model.add(Dense(14, activation='leaky_relu'))
    model.add(Dense(100, activation='leaky_relu'))
    # adds a dense layer with one output and sigmoid activation function 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model

# Parsing of the dataset file - pretty much self explanatory
headers = []
labels = []

# Read the file, load, parse and preprocess the data
with open('Dataset_1.csv') as file:
    # Load Data
    data = csv.reader(file, delimiter=',')
    
    # Parse Data
    data = list(data)
    #string correction - remove weird character causing trouble :)
    data[0][0] = "age"
    headers = data[0][:]
    # delete the headers from the data object
    del data[0]
    
    # Preprocess Data
    # calculate mean values and store them in <m>:
    m = []
    for i in range(0, len(data[0])):
        d = 0 # sum
        cnt = 0 # number of data points
        for j in range(0, len(data)):
            d = d + float(data[j][i])
            cnt = cnt + 1
        m.append(d / cnt)
        #print(m)
    
    temp = []
    for row in data:
        temp_row = []
        # preprocess data
        for j in range(0, len(row)):
            #zero_cnt = 0
            # non binary columns
            if j in [0, 3, 4, 7, 9]:
                if j == (len(row) - 1):
                    # store the "target" column - should be int
                    labels.append(int(row[j]))
                else:
                    f = float(row[j])
                    # remove outliers - ignore values if they are more than two times the mean
                    if (f > m[j]) and (f < (2* m[j])):
                        #standardization
                        # value = (value - μ) / σ
                        temp_row.append(f - m[j])
                    else:
                        temp_row.append(0)
                        #zero_cnt = zero_cnt + 1
            else:
                #binary columns
                if j == (len(row) - 1):
                    # store the "target" column - should be int
                    labels.append(int(row[j]))
                else:
                    temp_row.append(float(row[j]))
        # downsampling - remove outliers
        cnt = 0
        if temp_row[0] == 0:
            cnt = cnt + 1
        if temp_row[3] == 0:
            cnt = cnt + 1
        if temp_row[4] == 0:
            cnt = cnt + 1
        if temp_row[7] == 0:
            cnt = cnt + 1
        if temp_row[9] == 0:
            cnt = cnt + 1
            
        if cnt >= 3:
            del labels[-1]
        else:
            temp.append(temp_row)
    
    #convert parsed data to numpy array
    data = np.array(temp)
      

# Testbench - self explanatory for the most part
means = []
stds = []
for i in range(0, 10):
    #"""
    # evaluate model with dataset
    estimator = KerasClassifier(build_fn=create_baseline, epochs=200, batch_size=8000, verbose=1)
    # 10-fold validator: splits the dataset into 10 train-test sets
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # added early stopping to avoid overfitting - restores best weights
    es = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=30, restore_best_weights=True)
    # get the results
    results = cross_val_score(estimator, data, labels, cv=kfold, fit_params={'callbacks':es}, error_score='raise')
    # display 
    mn = results.mean()*100
    st = results.std()*100
    print("Baseline: %.2f%% (%.2f%%)" % (mn, st))
    means.append(mn)
    stds.append(st)
    #"""

#plt.title("With preprocessing, Standardization & downsampling - 20 Neurons in hidden layer")
plt.title("With preprocessing - 100 Neurons in hidden layer")
plt.plot(means)
plt.plot(stds)
plt.savefig('WithP_S_D_and_100_N')	#save the figure in the current directory
plt.show()    

   