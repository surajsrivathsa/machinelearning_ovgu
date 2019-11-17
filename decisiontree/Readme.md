Write a Python3 program that implements a decision tree using the ID3 algorithm presented in the lecture. 
Use the following entropy calculation:
C entropy(S) = − 􏰉 pilogC (pi)
i=1
where pi is the proportion of class i (with C being all classes in the data set). 
Use Information Gain as your decision measure and treat all features as discrete multinomial distributions.
Given are the two data sets1 named car and nursery as csv files. 
Your program should be able to read both data sets and treat the last value of each line as the class. 
Your task is to correctly implement the ID3 algorithm and return the final tree without stopping early (both data sets can 
be learned perfectly, i.e. all leaves have an entropy of 0). 
The output of your algorithm should look like the example XML solution given for the car data set. 
With that, you can check the correctness of your solution. 
All features are unnamed on purpose, please number them according to the column starting from 0 (e.g. att0).
