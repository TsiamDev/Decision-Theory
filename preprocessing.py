# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 18:16:30 2022

@author: Konstantinos Tsiamitros
"""

import csv

#with open('Dataset_small.csv') as file:
with open('Dataset_1.csv') as file:
    data = csv.reader(file, delimiter=',')
    
    data = list(data)
    
    #string correction - remove weird character causing trouble :)
    data[0][0] = "age"
    
            
 
    
