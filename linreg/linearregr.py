#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 21:25:24 2019

@group: Monday 13-15 pm Group 4

@Tutor name: Anirban Saha

@Authors: Chandan Bhat(229746), Rashmi Koparde(230322), Suraj Shashidhar(230052)



Description: The program has to parse input data along with the expected results.
    It should come up with its own implementation of multi variable linear equation.
    It should calculate the appropriate weights to reduce the errors in predicted results.
    
    The program should be able to run on any dataset and should be able to take 
    learning rate and error threshold from the command line.
    
    User can then tune the learning threshold and threshold from command line.

Run_Format: 
python3 linearregr.py --data "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/linreg/random.csv" --learningRate "0.0001" --threshold "0.0001" 

python3 linearregr.py --data "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/linreg/yacht.csv --learningRate "0.0001" --threshold "0.0001"
"""

import numpy as np
import pandas as pd
import argparse
import os
"""
Importing Numpy for array calculation, Importing pandas for file I/O, Import argparse for argument parsing
os for getting base path for output location
"""
ap = argparse.ArgumentParser()

# Add the arguments to the parser and parse the arguments from command line
ap.add_argument( "--data", required=True, help="first operand")
ap.add_argument("--learningRate", required=True, help="second operand")
ap.add_argument("--threshold", required=True, help="third operand")
#ap.add_argument("--output", required=True, help="fourth operand")
args = vars(ap.parse_args())

filepath = args["data"]
learning_rate = float(args["learningRate"])
threshold = float(args["threshold"])
#output_filepath = args["output"]

"""
Function name: readInputFile
input: filepath
output: dataframe with features and expected value , with number of weights
"""
def readInputFile(filepath):
    tmpDf = pd.read_csv(filepath);
    print("Record count is: " + str(tmpDf.shape[0]))
    numOfCols = tmpDf.shape[1]
    collist = []
    base_columnname = "x"  
    t = base_columnname
    
    for i in range(1,numOfCols):
        t = "x"
        t = t + str(i)
        collist.append(t)
        
    collist.append("expected_value")
    
    print(*collist)
    tmpDf = pd.read_csv(filepath,  index_col = None, names = collist);
    tmpDf.insert(0, "x0", 1.0)
    return numOfCols, tmpDf;

# get the input dataframe along with number of weights need to be calculated 
numOfWeights, inputDf = readInputFile(filepath)

#convert dataframe into an numpy array
inputNpArray = inputDf.rename_axis('ID').values

# extract the features and expected values into separate Numpy arrays
featureNpArray = inputNpArray[:,:-1]
expectedvalueNpArray = inputNpArray[:,-1:]
record_count = expectedvalueNpArray.shape[0]

#Function takes record_count and numOfWeights as ip and returns
#initilized weights and predicted array
def initialize_required_arrays(record_count,numOfWeights):    
    weightsNpArray = np.zeros(shape=(1, numOfWeights));
    predictedvalueNpArray = np.zeros(shape=(record_count,1));  
    errorNpArray =  np.zeros(shape=(record_count,1));
    gradientNpArray = np.zeros(shape=(1, numOfWeights));
    return weightsNpArray, predictedvalueNpArray, errorNpArray, gradientNpArray

#initializing the wights and predicted value numpy arrays
weightsNpArray, predictedvalueNpArray, errorNpArray, gradientNpArray = initialize_required_arrays(record_count,numOfWeights);


#Calculate the predicted value for example: f(x) = w0 + w1x1 + w2x2 + w3x3
def calculate_predicted_value(weightsNpArray, featureNpArray):
    return np.dot(featureNpArray,weightsNpArray.T);


"""
Function name: update_weights
input: earning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray
output: numpy array of weights

Function takes input parameters, First it calculates predicted value
error = expected value - predicted value
gradient = error * feature
new weight = old weight + learning rate * gradient
"""
def update_weights(learning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray):
    predictedvalueNpArray = calculate_predicted_value(weightsNpArray, featureNpArray);
    errorNpArray = expectedvalueNpArray - predictedvalueNpArray;
    tmpNpArray = errorNpArray*featureNpArray
    gradientNpArray = np.sum(tmpNpArray, axis = 0)
    weightsNpArray = weightsNpArray + learning_rate * gradientNpArray
    return weightsNpArray


# This function calculates cost by first squaring entire error array and summing the array to single value
def calculate_costfunction(featureNpArray, expectedvalueNpArray, weightsNpArray):
    predicted_values = calculate_predicted_value(weightsNpArray, featureNpArray)
    error = expectedvalueNpArray - predicted_values;
    sq_error = np.array(error)**2
    return round(sq_error.sum(),4)

#initialization for job run
output_weight_lst = []
output_cost_lst = []
cnt = 0
prev_error = 0.0
flag = 0

#Start the updation process
"""
First calculate the error / cost
use the error to update the weights for the next batch
check if the threshold is met or we have already reached the minima and cannot further minimize the error
if thats the case then break from loop
else use the new weights and again do the above things
"""
while(1):
    cost = calculate_costfunction(featureNpArray, expectedvalueNpArray, weightsNpArray);
    output_cost_lst.append(cost)
    weightsNpArray = update_weights(learning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray)
    output_weight_lst.append(weightsNpArray.tolist())
    cnt = cnt + 1        
    if(abs(prev_error - cost) < threshold ):
        break;
    prev_error = cost
     
    

print("\n ---------------------------------------------------- \n")


#Process to just collect the updated weights and append it into a dataframe. 
#After dataframe has been created just print its head and tail and create a output file from it
output_lst = []

for i in range(cnt):
    tmp_lst = []
    for j in range(len(output_weight_lst[0])):
        for k in range(len(output_weight_lst[0][0])):            
            tmp_lst.append(output_weight_lst[i][j][k])
    tmp_lst.append(output_cost_lst[i])
    output_lst.append(tmp_lst)
    

columnlist = []
for i in range(numOfWeights):
    basename = "weight"
    columnname = "weight" + str(i)
    columnlist.append(columnname)

columnlist.append("sum_of_squared_errors")    

output_df = pd.DataFrame(output_lst, columns = columnlist, index = None)
output_df.index.name = "iteration_number"

print(output_df.head(5))
print(output_df.tail(5))
print(output_df.shape)


#Creating a output file consisting of weights and sum of squared errors from dataframe

#print(os.path.basename(filepath))
output_filename = os.path.basename(filepath)
output_filename = "p01_" + output_filename
output_df.to_csv(output_filename, sep=',', encoding='utf-8')


 
    


    
    
    
    
    









    


































