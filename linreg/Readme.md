Write a program that implements a (batch) linear regression using the gradient descent method. 
Use the following gradient calculation:

N gradient=􏰁x⃗i(yi −f(x⃗i))
i=1
w⃗ ← w⃗ + η · gradient

where x⃗i is one data point (with N being the size of the data set), η the learning rate, yi is the target output and f (x⃗i ) is the linear function defined as f (⃗x) = w⃗ T ⃗x or equivalently f(⃗x) = 􏰀i wi · xi. Whereas w⃗ and ⃗x include the bias/intercept, i.e. w0 and x0 = 1. All weights should be initialized as 0.
Given are the two data sets1 named yacht and random as csv files. Your program should be able to read both data sets and treat the last value of each line as the target output. Your task is to correctly implement the gradient descent method and return for each iteration the weights and sum of squared errors until a given threshold of change in the error is reached. The output of your algorithm should look like this:
iteration_number,weight0,weight1,weight2,...,weightN,sum_of_squared_errors
