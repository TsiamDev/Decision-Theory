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
    model.add(Dense(50, activation='leaky_relu'))
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
    
    m = []
    # mean values:
    for i in range(0, len(data[0])):
        d = 0
        cnt = 0

        for j in range(0, len(data)):
        
            d = d + float(data[j][i])
            cnt = cnt + 1
        m.append(d / cnt)
        print(m)
    
    temp = []
    for row in data:
        temp_row = []
        #"""
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
        if temp_row[0] == 0:
            del labels[-1]
        else:
            temp.append(temp_row)
        #"""
        """
        for j in range(0, len(row)):
            if j == (len(row) - 1):
                # store the "target" column - should be int
                labels.append(int(row[j]))
            else:
                # store the rest of the columns
                temp_row.append(float(row[j]))
        temp.append(temp_row)
        #"""
    
    #convert parsed data to numpy array
    data = np.array(temp)
      
    #fig = plt.figure(1)	#identifies the figure 

    """
    for i in [0, 3, 4, 7, 9]:
    #for i in range(0, len(data[0])):
        x = data[:,i]
        x = [math.log10(i + 0.000001) for i in x]
        #x = sorted(x, reverse=True)
        y = [math.log10(i + 0.000001) for i in labels]
        x, y = zip(*sorted(zip(x, y), reverse=True))
        plt.title(headers[i], fontsize='16')	#title
        plt.scatter(range(0, len(x)), x)	#plot the points
        plt.scatter(range(0, len(x)), y)	#plot the points
        #break
        

    
    #plt.xlabel("X",fontsize='13')	#adds a label in the x axis
    #plt.ylabel("Y",fontsize='13')	#adds a label in the y axis
    #plt.legend(('YvsX'),loc='best')	#creates a legend to identify the plot
    #plt.savefig('Y_X.png')	#saves the figure in the present directory
        plt.grid()	#shows a grid under the plot
        plt.show()
    """

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

plt.title("With preprocessing, Standardization & downsampling - 50 Neurons in hidden layer")
#plt.title("No preprocessing - 14 Neurons in hidden layer")
plt.plot(means)
plt.plot(stds)
plt.savefig('WithP_S_D_and_50_N')	#saves the figure in the present directory
plt.show()    

   