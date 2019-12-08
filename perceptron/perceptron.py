#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:12:32 2019

@author: surajshashidhar


python perceptron.py --data "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/perceptron/Example.tsv" --output "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/perceptron/op_Example_perceptron.tsv"


python perceptron.py --data "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/perceptron/Gauss2.tsv" --output "/Users/surajshashidhar/Desktop/ovgu/semester_1/machine_learning/programming/perceptron/op_Gauss2_perceptron.tsv"

"""

import numpy as np
import pandas as pd
import argparse
import os

ap = argparse.ArgumentParser()

# Add the arguments to the parser and parse the arguments from command line
ap.add_argument( "--data", required=True, help="first operand")
ap.add_argument("--output", required=True, help="second operand")

args = vars(ap.parse_args())

filepath = args["data"]
output_filepath = args["output"]
learning_rate = 1
number_of_iterations = 100

#output_filepath = args["output"]

"""
Function name: readInputFile
input: filepath
output: dataframe with features and expected value , with number of weights
also remove any column that has "nan" values. Rxamples file had one
"""

def readInputFile(filepath):
    flag = 0
    myfile = os.path.basename(filepath)
    if(myfile == "Example.tsv"):
        flag = 1
    
    df = pd.read_csv(filepath, sep='\t');
    
    #Data had an extra column with all the null values, hence need to cleanup before import
    #Anomoly found in examples file
    df = df.dropna(axis='columns', how = 'all')
    
    print("Record count is: " + str(df.shape[0]))
    print("Shape is: " + str(df.shape))
    numOfCols = df.shape[1]
    collist = []
    base_columnname = "x"  
    t = base_columnname
    
    collist.append("expected_value")
    for i in range(1,numOfCols):
        t = "x"
        t = t + str(i)
        collist.append(t)
        
    print("----")    
    print(*collist)
    if(flag == 1):
        collist.append("x3")
    tmpDf = pd.read_csv(filepath,  index_col = None, names = collist, sep='\t');
    if(flag > 0 ):
        if("x3" in collist):
            tmpDf.drop("x3", axis=1, inplace=True)
    
    #Apply a fuction to convert A to 1 and B to 0 ie A is + and B is -
    tmpDf['expected_value'] = [1 if x == 'A' else 0 for x in tmpDf['expected_value']]
    cols = tmpDf.columns.tolist()
    print(*cols)
    cols =  cols[1:] + cols[0:1]
    
    #Rearranging the columns and making the expected_value column as last column.
    #This s needed as in our below code we are considering that last column will be expected value
    tmpDf = tmpDf[cols]
    
    #Inserting bias column with initalization as 1
    tmpDf.insert(0, "x0", 1.0)
    print(*tmpDf.columns.tolist())
    return numOfCols, tmpDf;

# get the input dataframe along with number of weights need to be calculated 
numOfWeights, inputDf = readInputFile(filepath);


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
#Also find out whether the weighted sum will activate the perceptron or not
#if the net sum > 0 then output will be made as 1 else it will be 0
def calculate_predicted_value(weightsNpArray, featureNpArray):
    tmpNpArray = np.dot(featureNpArray,weightsNpArray.T);
    
    #Percetron Activation function(step function) condition
    tmpNpArray = np.where(tmpNpArray > 0.0, 1.0, 0.0)
    return tmpNpArray



"""
Function name: update_weights
input: learning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray
output: numpy array of weights

Function takes input parameters, First it calculates predicted value
error = expected value - predicted value
errorcount = count[error[i] != 0]
gradient = error * feature
new weight = old weight + learning rate * gradient
"""
def update_weights(learning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray):
    predictedvalueNpArray = calculate_predicted_value(weightsNpArray, featureNpArray);
    errorNpArray = expectedvalueNpArray - predictedvalueNpArray;
    
    #if there is some error then the differene between predicted and expected value will be other than zero
    errorcount = np.count_nonzero(errorNpArray != 0.0)    
    tmpNpArray = errorNpArray*featureNpArray
    gradientNpArray = np.sum(tmpNpArray, axis = 0)
    weightsNpArray = weightsNpArray + learning_rate * gradientNpArray
    return weightsNpArray,errorcount


# This function calculates cost by first squaring entire error array and summing the array to single value
def calculate_costfunction(featureNpArray, expectedvalueNpArray, weightsNpArray):
    predicted_values = calculate_predicted_value(weightsNpArray, featureNpArray)
    error = expectedvalueNpArray - predicted_values;
    sq_error = np.array(error)**2
    return round(sq_error.sum(),4)

#initialization for job run
output_weight_lst = []
cnt = 0
error_count_lst = []


#Start the updation process for normal learning process
"""
First calculate the error
use the error to update the weights for the next batch
check if have reached 100th iteration and break if condition is satisfied
if thats the case then break from loop
else use the new weights and again do the above things
"""
while(cnt <= number_of_iterations):
    weightsNpArray,errorcount = update_weights(learning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray)
    output_weight_lst.append(weightsNpArray.tolist())
    error_count_lst.append(errorcount)
    cnt = cnt + 1        
    
    
output_weight_annealing_lst = []
annealing_cnt = 0
error_count_annealing_lst = []
annealing_denominator = 1
annealing_learning_rate = learning_rate

#Start the updation process for annealing learning process and einitialize the NP arrays

weightsNpArray, predictedvalueNpArray, errorNpArray, gradientNpArray = initialize_required_arrays(record_count,numOfWeights);

"""
First update the annealing learning rate according to iteration number
then calculate the error
use the error to update the weights for the next batch
check if have reached 100th iteration and break if condition is satisfied
if thats the case then break from loop
else use the new weights and again do the above things
"""
while(annealing_cnt <= number_of_iterations):
    annealing_learning_rate = learning_rate/annealing_denominator
    weightsNpArray,errorcount = update_weights(annealing_learning_rate, weightsNpArray, featureNpArray, expectedvalueNpArray)
    output_weight_annealing_lst.append(weightsNpArray.tolist())
    error_count_annealing_lst.append(errorcount)
    annealing_cnt = annealing_cnt + 1
    annealing_denominator = annealing_denominator + 1


##Here we simple pivot out=r list elemnt to add it as a column in the data frame, later we will dump this into csv
outputDf = pd.DataFrame()
for i in range(len(error_count_lst)):  
    normal_element = error_count_lst[i];
    annealing_element = error_count_annealing_lst[i]
    outputDf.insert(i, i, [normal_element,annealing_element] , True)


outputDf.to_csv(output_filepath, sep="\t", header = False, index = False)   



      