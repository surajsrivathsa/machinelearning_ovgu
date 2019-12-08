Write a program that implements a single perceptron using the delta rule presented in the lecture. Use the following activation function:
􏰀1 ifw⃗T⃗x>0 0 else
where w⃗ is the vector of weights including the bias (w0). Treat all attributes and weights
as double-precision values.
Given are the two data sets1 named Example and Gauss2 as tsv (tabular separated values)
files. Your program should be able to read both data sets and treat the first value of each
line as the class (A or B). In order to get the same results, class A is to be treated as the
positive class, hence y = 1, and class B as the negative one (y = 0). All weights are to
be initialized with 0. Your task is to correctly implement the perceptron learning rule in
batch mode with a constant (ηt = η0) and an annealing (ηt = η0 ) learning rate (in both
 cases η0 = 1), i.e:
t
yˆ =
 w⃗t+1 ←w⃗t + 􏰁 ηt(y−yˆ)⃗x ⃗x∈Y (⃗x,w⃗ )
where Y(⃗x,w⃗) is the set of samples which are misclassified. Please use the number of misclassified points as your error rate (i.e. |Y(⃗x,w⃗)|). The output of your algorithm should be a single tsv file, which contains exactly two rows after 100 iterations (per variant):
1. The first row contains the tabular separated values for the error of each iteration (starting from iteration 0) with the constant learning rate.
2. The second row follows the same format, but with the annealing learning rate.
The iteration number and any other information should not be inside the output file, only the error values. 
